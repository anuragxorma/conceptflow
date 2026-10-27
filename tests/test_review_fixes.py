"""
Regression tests for problems found while reviewing the 0.3.0 features.
"""

import re
import shutil
import subprocess
import warnings

import numpy as np
import pandas as pd
import pytest

from conceptflow import ConceptLattice, FormalContext, plot_lattice
from conceptflow.algorithms import enumeration
from conceptflow.cluster import ConceptLatticeEstimator
from conceptflow.core import direct_product, direct_sum
from conceptflow.exploration import nested_view
from conceptflow.io import context_from_json, read_csv
from conceptflow.rules import ImplicationBasisEstimator, ImplicationOutlierDetector
from conceptflow.visualization import latex_escape, lattice_to_dot, plot_nested

# ---------------------------------------------------------------------------
# Reading cell values: one rule per cell, whatever the dtype
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "frame",
    [
        pd.DataFrame({"a": ["1", "0"], "b": ["x", ""]}),
        pd.DataFrame({"a": ["1", "0"], "b": ["1", "0"]}),
    ],
)
def test_string_cells_do_not_depend_on_other_cells(frame):
    assert FormalContext.from_dataframe(frame).incidence[:, 0].tolist() == [True, False]


def test_string_cells_same_for_every_dtype():
    for array in [np.array([["1", "0"]]), np.array([["1", "0"]], dtype=object)]:
        assert FormalContext.from_array(array).incidence.tolist() == [[True, False]]


@pytest.mark.parametrize("value", ["yes", "no", "maybe"])
def test_ambiguous_strings_are_rejected(value):
    with pytest.raises(ValueError, match="Cannot read"):
        FormalContext.from_array(np.array([[value]], dtype=object))


def test_indices_are_range_checked():
    ctx = FormalContext.from_array([[1, 0]])
    for call in [
        lambda: ctx.remove_objects([-1]),
        lambda: ctx.remove_objects([5]),
        lambda: ctx.subcontext(attributes=[2]),
        lambda: ctx.attribute_derivation([-1]),
    ]:
        with pytest.raises(IndexError):
            call()
    with pytest.raises(KeyError, match="Unknown attribute"):
        ctx.attribute_derivation(["nope"])


def test_empty_incidence_for_zero_objects():
    assert FormalContext(objects=[], attributes=["a"], incidence=[]).incidence.shape == (0, 1)


# ---------------------------------------------------------------------------
# Iceberg lattices and the size guard
# ---------------------------------------------------------------------------


def test_iceberg_always_has_its_top_concept():
    ctx = FormalContext.from_array([[1, 0], [0, 1]])
    lattice = ConceptLattice.from_context(ctx, min_support=5)
    assert lattice.top().extent == {0, 1}

    est = ConceptLatticeEstimator(min_support=5).fit(ctx.incidence)
    assert est.transform(ctx.incidence).shape == (2, 1)


def test_max_concepts_accepts_numpy_integers_and_silences_the_warning(monkeypatch):
    monkeypatch.setattr(enumeration, "LARGE_LATTICE_WARNING_SIZE", 3)
    ctx = FormalContext.from_array(np.eye(4))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ConceptLattice.from_context(ctx, max_concepts=np.int64(1000))
    assert not [w for w in caught if issubclass(w.category, enumeration.LargeLatticeWarning)]


# ---------------------------------------------------------------------------
# I/O and context operations
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, message",
    [
        ('[{"ObjNames":["g"],"Params":{"AttrNames":["a"]}},'
         '{"Count":1,"Data":[{"Count":1,"Inds":[-1]}]}]', "not in"),
        ('[{"ObjNames":["g","h"],"Params":{"AttrNames":["a"]}},'
         '{"Count":1,"Data":[{"Count":1,"Inds":[0]}]}]', "rows"),
        ('[{"ObjNames":["g"],"Params":{"AttrNames":["a"]}},'
         '{"Count":1,"Data":[{"Count":1}]}]', "Unrecognised"),
    ],
)
def test_fcart_json_is_validated(text, message):
    with pytest.raises(ValueError, match=message):
        context_from_json(text)


def test_csv_rows_longer_than_the_header(tmp_path):
    path = tmp_path / "ctx.csv"
    path.write_text(",a\no1,1,1\n")
    with pytest.raises(ValueError, match="2 values"):
        read_csv(path)


def test_operation_name_clashes_are_explicit():
    first = FormalContext.from_array([[1], [1]], objects=["a|b", "a"], attributes=["m"])
    second = FormalContext.from_array([[1], [1]], objects=["c", "b|c"], attributes=["n"])
    with pytest.raises(ValueError, match="separator"):
        direct_product(first, second)
    with pytest.raises(ValueError, match="differ"):
        direct_sum(first, second, prefixes=("p", "p"))


def test_nested_view_builds_only_the_inner_lattice_it_draws():
    objects = [f"o{i}" for i in range(6)]
    rng = np.random.default_rng(0)
    outer = FormalContext.from_array(rng.random((6, 3)) < 0.5, objects, ["a", "b", "c"])
    inner = FormalContext.from_array(rng.random((6, 2)) < 0.5, objects, ["p", "q"])

    view = nested_view(outer, inner)
    assert list(view.children) == [view.lattice.top().stable_id()]


def test_plot_nested_checks_arguments_first():
    ctx = FormalContext.from_array([[1]], objects=["o"], attributes=["a"])
    with pytest.raises(TypeError, match="inner="):
        plot_nested(ctx)
    with pytest.raises(ValueError, match="backend"):
        plot_nested(ctx, inner=ctx, backend="nope")


# ---------------------------------------------------------------------------
# Drawings
# ---------------------------------------------------------------------------


def test_layered_iceberg_drawings_are_not_rotated():
    ctx = FormalContext.from_array(
        [[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1], [1, 0, 1, 0]]
    )
    lattice = ConceptLattice.from_context(ctx, min_support=2)
    graph = plot_lattice(lattice, backend="graph_data", layout="layered")
    fig = plot_lattice(lattice, backend="matplotlib", layout="layered")

    drawn_y = {round(y, 6) for y in fig.axes[0].collections[1].get_offsets()[:, 1]}
    assert len(drawn_y) == len({node.y for node in graph.nodes})


@pytest.mark.skipif(shutil.which("dot") is None, reason="Graphviz not installed")
def test_dot_labels_keep_backslashes_quotes_and_empty_names():
    ctx = FormalContext.from_array([[1, 0], [1, 1]], attributes=["", "x\\Gy"])
    dot = lattice_to_dot(ConceptLattice.from_context(ctx), title='end\\ "q"')

    result = subprocess.run(["dot", "-Tsvg"], input=dot.encode(), capture_output=True)
    assert result.returncode == 0
    texts = re.findall(r"<text[^>]*>([^<]*)</text>", result.stdout.decode())
    assert "x\\Gy" in texts
    assert any(t.startswith("end\\ ") for t in texts)


def test_latex_escape_turns_control_characters_into_spaces():
    assert latex_escape("a\n\nb\tc") == "a  b c"


def test_tikz_title():
    ctx = FormalContext.from_array([[1, 0], [0, 1]])
    fig = plot_lattice(
        ConceptLattice.from_context(ctx), backend="tikz", layout="layered", title="T_1"
    )
    assert "T\\_1" in fig.source


# ---------------------------------------------------------------------------
# Estimators
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "estimator",
    [ConceptLatticeEstimator(), ImplicationBasisEstimator(), ImplicationOutlierDetector()],
)
def test_columns_must_match_fit_time_names(estimator):
    X = pd.DataFrame(np.random.default_rng(0).random((30, 3)) < 0.5, columns=["a", "b", "c"])
    estimator.fit(X)
    use = estimator.transform if hasattr(estimator, "transform") else estimator.predict

    use(X)
    use(X.to_numpy())  # plain arrays are not checked
    with pytest.raises(ValueError, match="different order"):
        use(X[["c", "b", "a"]])
    with pytest.raises(ValueError, match="Unseen"):
        use(X.rename(columns={"a": "z"}))


def test_n_concepts_without_select_is_ignored_for_grid_searches():
    X = np.random.default_rng(1).random((20, 4)) < 0.5
    est = ConceptLatticeEstimator(n_concepts=2).fit(X)
    assert len(est.concepts_) == est.get_lattice().n_concepts
