"""
Regression tests for problems found in the pre-0.2.0 audit.
"""

import pickle
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from conceptflow import (
    Concept,
    ConceptLattice,
    ExplorationBuilder,
    FormalContext,
    IncompleteContext,
    ManyValuedContext,
)
from conceptflow.algorithms import (
    compute_hasse_edges,
    compute_luxenburger_basis,
    enumerate_concepts,
)
from conceptflow.cluster import ConceptLatticeEstimator
from conceptflow.metrics import implication_exceptions
from conceptflow.preprocessing import ConceptualScaler, NominalScale, OrdinalScale

# ---------------------------------------------------------------------------
# NumPy integer indices
# ---------------------------------------------------------------------------


def test_hasse_edges_with_numpy_integer_indices_beyond_64_objects():
    rng = np.random.default_rng(0)
    ctx = FormalContext.from_array(rng.random((80, 4)) < 0.5)
    concepts = enumerate_concepts(ctx)
    numpy_concepts = [
        Concept(
            extent=[np.int64(i) for i in c.extent],
            intent=[np.int64(i) for i in c.intent],
        )
        for c in concepts
    ]
    expected = set(compute_hasse_edges(concepts, context=ctx))

    assert set(compute_hasse_edges(numpy_concepts, context=ctx)) == expected
    assert set(compute_hasse_edges(numpy_concepts)) == expected
    assert numpy_concepts == concepts


def test_numpy_integers_accepted_as_indices():
    ctx = FormalContext.from_array([[1, 1], [1, 0]], attributes=["a", "b"])

    assert implication_exceptions(ctx, [np.int64(0)], [np.int64(1)]) == {1}
    assert compute_luxenburger_basis(ctx, min_support=np.int64(1))
    assert compute_luxenburger_basis(ctx, min_support=np.float64(0.5))


def test_incomplete_context_index_validation():
    ctx = IncompleteContext.from_array([[1, None], [0, 1]])

    assert ctx.certain_extent([np.int64(1)]) == {1}
    with pytest.raises(IndexError):
        ctx.certain_extent([-1])
    with pytest.raises(IndexError):
        ctx.certain_extent([2])
    with pytest.raises(KeyError):
        ctx.certain_extent([True])


# ---------------------------------------------------------------------------
# Missing values never become crosses
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "build",
    [
        lambda: FormalContext.from_dataframe(pd.DataFrame({"linear": [1.0, np.nan]})),
        lambda: FormalContext.from_array([[1.0, np.nan]]),
        lambda: FormalContext.from_array(np.array([[True, None]], dtype=object)),
        lambda: FormalContext.from_dataframe(
            pd.DataFrame({"linear": [True, pd.NA]}, dtype="boolean")
        ),
        lambda: FormalContext.from_dataframe(pd.DataFrame({"linear": ["x", None]})),
    ],
)
def test_formal_context_reads_empty_cells_as_no_cross(build):
    # bool(NaN) is True: converting directly would turn blanks into crosses.
    assert build().incidence.ravel().tolist() == [True, False]


def test_formal_context_rejects_unknown_marker():
    with pytest.raises(ValueError, match="IncompleteContext"):
        FormalContext.from_array(np.array([[True, "?"]], dtype=object))


def test_estimators_reject_nan_but_point_to_cross_table_reading():
    frame = pd.DataFrame({"a": [1.0, np.nan], "b": [1.0, 1.0]})

    with pytest.raises(ValueError, match="from_dataframe"):
        ConceptLatticeEstimator().fit(frame)

    ConceptLatticeEstimator().fit(FormalContext.from_dataframe(frame))


def test_estimators_explain_incomplete_input():
    incomplete = IncompleteContext.from_array([[1, None], [0, 1]])

    with pytest.raises(ValueError, match="certain_context"):
        ConceptLatticeEstimator().fit(incomplete)

    frame = incomplete.to_dataframe()
    with pytest.raises(ValueError, match="certain_context"):
        ConceptLatticeEstimator().fit(frame)

    complete = IncompleteContext.from_array([[1, 0], [0, 1]])
    ConceptLatticeEstimator().fit(complete)


def test_question_mark_is_missing_in_scaler_too():
    df = pd.DataFrame({"c": ["x", "?"]}, index=["a", "b"])

    with pytest.raises(ValueError, match="NaN"):
        ConceptualScaler([NominalScale("c")]).fit(df)

    scaler = ConceptualScaler([NominalScale("c")], missing="false").fit(df)
    assert list(scaler.get_feature_names_out()) == ["c=x"]


def test_unknown_respects_scale_structure():
    df = pd.DataFrame({"size": ["S", "L", None]}, index=["a", "b", "c"])
    ctx = ConceptualScaler(
        [OrdinalScale("size", ["S", "M", "L"])], output="context", missing="unknown"
    ).fit_transform(df)

    # Every level is >= S, so that attribute is known even for c.
    assert ctx.implication_status([], ["size>=S"]) == "valid"
    assert ctx.certain[2].tolist() == [True, False, False]
    assert ctx.unknown[2].tolist() == [False, True, True]
    # Unknown cells are independent, so implications that follow from the
    # scale itself can still show as "possible" (documented limitation).
    assert ctx.implication_status(["size>=L"], ["size>=M"]) == "possible"


def test_params_are_revalidated_at_transform_time():
    scaler = ConceptualScaler([NominalScale("c")]).fit(pd.DataFrame({"c": ["x", "y"]}))

    scaler.set_params(missing="bogus")
    with pytest.raises(ValueError, match="missing"):
        scaler.transform(pd.DataFrame({"c": ["x"]}))

    scaler.set_params(missing="unknown", output="sparse")
    with pytest.raises(ValueError, match="sparse"):
        scaler.transform(pd.DataFrame({"c": ["x"]}))


def test_scaler_pickled_without_missing_param_still_works():
    df = pd.DataFrame({"c": ["x", "y"]})
    scaler = ConceptualScaler([NominalScale("c")]).fit(df)
    # Simulate a 0.1.x pickle, written before the parameter existed.
    del scaler.__dict__["missing"]

    restored = pickle.loads(pickle.dumps(scaler))

    assert restored.missing == "error"
    assert restored.transform(df).tolist() == [[True, False], [False, True]]


def test_exploration_builder_missing_hint_names_the_builder():
    mvc = ManyValuedContext.from_dataframe(
        pd.DataFrame({"c": ["x", None]}, index=["a", "b"])
    )

    with pytest.raises(ValueError, match="ExplorationBuilder") as info:
        ExplorationBuilder(mvc).root("r", [NominalScale("c")])
    assert "unknown" not in str(info.value)


# ---------------------------------------------------------------------------
# DimFlux bridge and import cost
# ---------------------------------------------------------------------------


def test_import_conceptflow_does_not_load_dimflux():
    code = "import sys, conceptflow; print('dim_flux' in sys.modules)"
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "False"


@pytest.mark.parametrize("timeout_ms", [-1, 1.5, True])
def test_dimflux_rejects_invalid_timeouts(timeout_ms):
    from conceptflow.visualization import lattice_to_graph_data_dimflux

    lattice = ConceptLattice.from_context(FormalContext.from_array([[1, 0], [1, 1]]))
    with pytest.raises(ValueError, match="timeout_ms"):
        lattice_to_graph_data_dimflux(lattice, timeout_ms=timeout_ms)


def test_dimflux_warns_on_large_lattices(monkeypatch):
    from conceptflow.visualization import dimflux_layout

    lattice = ConceptLattice.from_context(
        FormalContext.from_array(np.eye(8, dtype=bool) == 0)  # 2^8 concepts
    )
    monkeypatch.setattr(
        dimflux_layout,
        "_dimflux_coordinates",
        lambda context, timeout_ms: {c.extent: np.zeros(2) for c in lattice.concepts},
    )

    with pytest.warns(UserWarning, match="layout=\"none\""):
        dimflux_layout.lattice_to_graph_data_dimflux(lattice)


# ---------------------------------------------------------------------------
# Second review
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "data",
    [
        [["x", "?"]],
        np.array([["x", "?"]]),
        np.array([[b"x", b"?"]]),
        np.array([["x", "?"]], dtype=object),
    ],
)
def test_unknown_marker_rejected_for_every_string_dtype(data):
    with pytest.raises(ValueError, match="IncompleteContext"):
        FormalContext.from_array(data)


def test_formal_context_does_not_alias_caller_array():
    source = np.array([[True, False]])
    ctx = FormalContext.from_array(source)
    source[0, 1] = True

    assert ctx.incidence.tolist() == [[True, False]]


def test_contains_sets_memory_scales_with_set_sizes():
    import tracemalloc

    from conceptflow.algorithms._membership import contains_sets

    sets = [frozenset(range(i % 50, i % 50 + 5)) for i in range(20000)]
    tracemalloc.start()
    contains_sets(np.zeros((1, 1000), dtype=bool), sets)
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()

    # A dense sets x attributes matrix alone would be 80 MB.
    assert peak < 20e6


def test_fractional_min_support_is_read_as_its_decimal_value():
    from conceptflow.algorithms.implications import _resolve_min_support

    assert _resolve_min_support(np.float32(0.1), 30) == 3
    assert _resolve_min_support(0.1, 30) == 3
    assert _resolve_min_support(0.07, 300_000_000) == 21_000_000


def test_luxenburger_concepts_argument_is_validated_and_deduplicated():
    ctx = FormalContext.from_array(np.random.default_rng(0).random((10, 5)) < 0.5)
    concepts = enumerate_concepts(ctx)

    assert len(compute_luxenburger_basis(ctx, concepts=concepts + concepts)) == len(
        compute_luxenburger_basis(ctx, concepts=concepts)
    )
    with pytest.raises(ValueError, match="outside the context"):
        compute_luxenburger_basis(ctx, concepts=[Concept({0, 1, 50}, {0})])
