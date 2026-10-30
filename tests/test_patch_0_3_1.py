"""
Regression tests for problems found in the post-0.3.0 audit.
"""

import json
import re
from fractions import Fraction

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest
import scipy.sparse

from conceptflow import (
    ConceptLattice,
    FormalContext,
    IncompleteContext,
    ManyValuedContext,
    plot_lattice,
)
from conceptflow.algorithms import compute_luxenburger_basis
from conceptflow.decomposition import ExactOrdinalTwoFactorizer
from conceptflow.feature_extraction import ConceptMembershipEncoder
from conceptflow.io import read_cxt, write_cxt
from conceptflow.io.formats import context_from_json
from conceptflow.metrics import concept_entropy, implication_confidence
from conceptflow.preprocessing import ConceptualScaler, ContranominalScale, NominalScale
from conceptflow.rules import ImplicationBasisEstimator
from conceptflow.validation import check_binary_context_input
from conceptflow.visualization import plot_nested

# ---------------------------------------------------------------------------
# Contexts
# ---------------------------------------------------------------------------


def test_context_tables_are_read_only():
    ctx = FormalContext.from_array([[1, 0], [0, 1]])
    with pytest.raises(ValueError, match="read-only"):
        ctx.incidence[0, 0] = False

    certain = np.zeros((2, 2), bool)
    unknown = np.zeros((2, 2), bool)
    partial = IncompleteContext(("a", "b"), ("x", "y"), certain, unknown)
    certain[0, 0] = unknown[0, 0] = True  # the caller's arrays, not the context's
    assert not partial.certain[0, 0] and not partial.unknown[0, 0]
    with pytest.raises(ValueError, match="read-only"):
        partial.unknown[0, 0] = True


def test_to_dataframe_is_writable():
    frame = FormalContext.from_array([[1, 0]]).to_dataframe()
    frame.iloc[0, 0] = False
    assert not frame.iloc[0, 0]


def test_many_valued_context_equality_and_hash():
    first = ManyValuedContext.from_dataframe(pd.DataFrame({"a": [1, 2]}))
    second = ManyValuedContext.from_dataframe(pd.DataFrame({"a": [1, 2]}))
    other = ManyValuedContext.from_dataframe(pd.DataFrame({"a": [1, 3]}))
    assert first == second and hash(first) == hash(second)
    assert first != other


def test_add_objects_and_attributes_check_the_shape():
    ctx = FormalContext.from_array([[1, 0], [0, 1], [1, 1]])
    with pytest.raises(ValueError, match="shape"):
        ctx.add_attributes(["d", "e"], [[1, 0, 1], [0, 1, 1]])
    with pytest.raises(ValueError, match="shape"):
        ctx.add_objects(["p", "q"], [[1, 0, 1, 1]])

    assert ctx.add_objects(["p"], [1, 0]).incidence[-1].tolist() == [True, False]
    assert ctx.add_attributes(["d"], [1, 0, 1]).incidence[:, -1].tolist() == [True, False, True]


def test_single_string_is_one_name():
    ctx = FormalContext.from_array([[1, 1, 0, 1], [1, 1, 1, 0]], attributes=["a", "b", "ab", "c"])
    assert ctx.attribute_derivation("ab") == ctx.attribute_derivation(["ab"])
    assert implication_confidence(ctx, "ab", "c") == implication_confidence(ctx, ["ab"], ["c"])


def test_sparse_input_to_formal_context():
    matrix = scipy.sparse.csr_matrix(np.eye(2))
    assert FormalContext.from_array(matrix) == FormalContext.from_array(np.eye(2))


def test_incomplete_context_unknown_name_message():
    partial = IncompleteContext.from_dataframe(pd.DataFrame({"x": [True, None]}))
    with pytest.raises(KeyError, match="Unknown attribute 'zz'"):
        partial.certain_extent(["zz"])


def test_lattices_from_different_algorithms_are_equal():
    ctx = FormalContext.from_array(np.random.default_rng(0).random((8, 5)) < 0.5)
    first = ConceptLattice.from_context(ctx)
    second = ConceptLattice.from_context(ctx, algorithm="closebyone")
    assert first == second and hash(first) == hash(second)


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------


def test_write_cxt_rejects_line_breaks_in_names(tmp_path):
    ctx = FormalContext.from_array([[1]], objects=["a\nb"], attributes=["m"])
    with pytest.raises(ValueError, match="line break"):
        write_cxt(ctx, tmp_path / "x.cxt")


def test_read_cxt_accepts_trailing_blank_lines_and_bom(tmp_path):
    ctx = FormalContext.from_array([[1, 0], [0, 1]], objects=["g", "h"], attributes=["m", "n"])
    path = tmp_path / "x.cxt"
    write_cxt(ctx, path)
    text = path.read_text(encoding="utf-8")

    path.write_text(text + "\n\n", encoding="utf-8")
    assert read_cxt(path) == ctx
    path.write_text(text, encoding="utf-8-sig")
    assert read_cxt(path) == ctx


def test_cxt_round_trip_without_attributes(tmp_path):
    ctx = FormalContext.from_array(np.zeros((2, 0)))
    write_cxt(ctx, tmp_path / "x.cxt")
    assert read_cxt(tmp_path / "x.cxt") == ctx


def test_malformed_json_raises_value_error():
    with pytest.raises(ValueError, match="incidence"):
        context_from_json('{"objects": ["a"], "attributes": ["x"]}')
    layout = [{"ObjNames": ["a"], "Params": {"AttrNames": ["x"]}}, {"Data": [{"Inds": 5}]}]
    with pytest.raises(ValueError, match="list of attribute indices"):
        context_from_json(json.dumps(layout))


# ---------------------------------------------------------------------------
# Algorithms and measures
# ---------------------------------------------------------------------------


def test_fraction_min_support():
    ctx = FormalContext.from_array([[1, 1], [1, 0], [0, 1], [0, 0]])
    assert ConceptLattice.from_context(ctx, min_support=Fraction(1, 2)) == (
        ConceptLattice.from_context(ctx, min_support=0.5)
    )


def test_invalid_algorithm_and_confidence_raise_value_error():
    ctx = FormalContext.from_array([[1, 0]])
    with pytest.raises(ValueError, match="algorithm"):
        ConceptLattice.from_context(ctx, algorithm=None)
    with pytest.raises(ValueError, match="min_confidence"):
        compute_luxenburger_basis(ctx, min_confidence=None)


def test_entropy_of_pure_concepts_is_positive_zero():
    lattice = ConceptLattice.from_context(FormalContext.from_array([[1, 0], [1, 1]]))
    values = concept_entropy(lattice, [0, 0])
    assert all(np.copysign(1.0, v) == 1.0 for v in values)


# ---------------------------------------------------------------------------
# Estimators
# ---------------------------------------------------------------------------


def test_string_arrays_as_estimator_input():
    expected = FormalContext.from_array([[True, False], [False, True]])
    assert check_binary_context_input([["x", "."], [".", "x"]]) == expected
    assert check_binary_context_input(np.array([["1", "0"], ["0", "1"]])) == expected
    with pytest.raises(ValueError, match="NaN"):
        check_binary_context_input(np.array([["x", None], [".", "x"]], dtype=object))


@pytest.mark.parametrize("scale", [NominalScale, ContranominalScale])
@pytest.mark.parametrize(
    "fit_values, transform_values",
    [([1, 2, 3], [1, np.nan]), ([1.0, np.nan, 3.0], [1, 3])],
)
def test_nominal_values_match_across_dtypes(scale, fit_values, transform_values):
    scaler = ConceptualScaler([scale("c")], missing="false").fit(pd.DataFrame({"c": fit_values}))
    reference = scaler.transform(pd.DataFrame({"c": [transform_values[0]]}))
    scaled = scaler.transform(pd.DataFrame({"c": transform_values}))
    assert scaled[0].tolist() == reference[0].tolist()


def test_scaler_refuses_unnamed_input_after_named_fit():
    frame = pd.DataFrame({"c": ["a", "b"]})
    scaler = ConceptualScaler([NominalScale("c")]).fit(frame)
    with pytest.raises(ValueError, match="feature names should match"):
        scaler.transform(frame.to_numpy())


def test_encoder_from_lattice_checks_feature_names():
    ctx = FormalContext.from_array([[1, 0, 1], [0, 1, 1]], attributes=list("abc"))
    encoder = ConceptMembershipEncoder.from_lattice(ConceptLattice.from_context(ctx))
    with pytest.raises(ValueError, match="feature names should match"):
        encoder.transform(pd.DataFrame([[1, 0, 1]], columns=list("xyz")))


def test_implication_feature_names_are_unique():
    X = pd.DataFrame(
        [[1, 0, 0, 0], [0, 1, 0, 0], [1, 1, 0, 1], [0, 0, 1, 1]],
        columns=["a", "b", "a, b", "c"],
    )
    names = ImplicationBasisEstimator().fit(X).get_feature_names_out()
    assert len(set(names)) == len(names)


def test_ordinal_factorizer_on_a_grid_is_fast():
    # The complement lattice is a 6 x 6 grid: about 10^14 linear extensions,
    # which the 0.3.0 brute-force search enumerated.
    k = 6
    complement = [
        [i >= a for a in range(1, k)] + [j >= b for b in range(1, k)]
        for i in range(k)
        for j in range(k)
    ]
    X = 1 - np.array(complement, dtype=int)
    assert ExactOrdinalTwoFactorizer().fit(X).coverage_ == 1.0


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def _graph_json(html: str) -> dict:
    return json.loads(re.search(r"const graphData = (.*?);\n", html).group(1))


def test_html_pages_escape_names_and_titles():
    ctx = FormalContext.from_array([[1, 0], [0, 1]], ["</script>", 'a"b'], ["<b>x", "c\\d"])
    html = plot_lattice(ConceptLattice.from_context(ctx), title="</div><b>").html
    assert html.count("</script>") == 2  # the d3 import and the page script
    assert "</div><b>" not in html
    assert any(n["metadata"]["extent"] == ["</script>"] for n in _graph_json(html)["nodes"])

    outer = FormalContext.from_array([[1], [0], [1]], ["g1", "g2", "g3"], ["x"])
    inner = FormalContext.from_array([[1], [1], [0]], ["g1", "g2", "g3"], ["y"])
    nested = plot_nested(outer, inner=inner, title='say "hi" \\ </script>').html
    assert nested.count("</script>") == 4  # d3, jsPDF, svg2pdf and the page script
    literal = re.search(r"rootData \? (.*?) : currentView.name", nested).group(1)
    assert json.loads(literal) == 'say "hi" \\ </script>'


def test_d3_without_layout_falls_back_to_layered():
    lattice = ConceptLattice.from_context(FormalContext.from_array([[1, 0], [0, 1]]))
    nodes = _graph_json(plot_lattice(lattice, layout="none").html)["nodes"]
    assert all(node["x"] is not None and node["y"] is not None for node in nodes)
    assert len({(node["x"], node["y"]) for node in nodes}) == len(nodes)


def test_two_concept_lattice_is_not_drawn_as_one_point():
    lattice = ConceptLattice.from_context(FormalContext.from_array([[1], [0]]))
    nodes = plot_lattice(lattice, backend="graph_data").nodes
    (x1, y1), (x2, y2) = [(node.x, node.y) for node in nodes]
    assert abs(y1 - y2) > 1e-3


def test_iceberg_object_labels_only_at_drawn_object_concepts():
    ctx = FormalContext.from_array([[1, 1], [1, 0], [1, 0]], ["g1", "g2", "g3"], ["a", "b"])
    lattice = ConceptLattice.from_context(ctx, min_support=2)
    source = plot_lattice(lattice, backend="tikz", layout="layered").source
    assert "g2" in source and "g1" not in source


def test_matplotlib_draws_dollar_signs_literally(tmp_path):
    lattice = ConceptLattice.from_context(FormalContext.from_array([[1]], ["$\\foo$"], ["m"]))
    axes = plot_lattice(lattice, backend="matplotlib", layout="layered")
    figure = getattr(axes, "figure", axes)
    figure.savefig(tmp_path / "x.png")


def test_figure_writers_create_folders_and_check_suffixes(tmp_path):
    lattice = ConceptLattice.from_context(FormalContext.from_array([[1, 0], [0, 1]]))
    tikz = plot_lattice(lattice, backend="tikz")
    assert tikz.write(tmp_path / "new" / "x.tex").exists()
    with pytest.raises(ValueError, match=".tex or .pdf"):
        tikz.write(tmp_path / "x.png")
    dot = plot_lattice(lattice, backend="graphviz")
    assert dot.write(tmp_path / "other" / "x.dot").exists()


# ---------------------------------------------------------------------------
# Audit leftovers fixed in 0.4.0
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "estimator",
    ["ConceptLatticeEstimator", "ConceptMembershipEncoder", "ImplicationBasisEstimator"],
)
def test_output_is_checked_again_at_transform(estimator):
    from conceptflow.cluster import ConceptLatticeEstimator  # noqa: F401
    from conceptflow.feature_extraction import ConceptMembershipEncoder  # noqa: F401
    from conceptflow.rules import ImplicationBasisEstimator  # noqa: F401

    X = np.random.default_rng(0).random((10, 4)) < 0.5
    model = locals()[estimator]().fit(X).set_params(output="list")
    with pytest.raises(ValueError, match="output must be"):
        model.transform(X)


def test_estimators_reject_infinity_in_dataframes():
    frame = pd.DataFrame({"a": [1.0, np.inf], "b": [0.0, 1.0]})
    with pytest.raises(ValueError, match="infinity"):
        check_binary_context_input(frame)
    # A FormalContext still reads inf as a cross (numbers > 0 are crosses).
    assert FormalContext.from_dataframe(frame).incidence[1, 0]


def test_integer_scale_source_is_a_column_position():
    integer_named = pd.DataFrame({0: ["a", "b"], 1: [1.0, 5.0]})
    scaler = ConceptualScaler([NominalScale(0)], output="dataframe")
    assert scaler.fit_transform(integer_named).columns.tolist() == ["0=a", "0=b"]

    named = pd.DataFrame({"color": ["a", "b"], "size": [1.0, 5.0]})
    scale = NominalScale(np.int64(0))
    out = ConceptualScaler([scale], output="dataframe").fit_transform(named)
    assert out.columns.tolist() == ["color=a", "color=b"]
    assert scale.source_attribute == "x0"  # the user's scale is not modified

    with pytest.raises(IndexError, match="column position 5"):
        ConceptualScaler([NominalScale(5)]).fit(named)
