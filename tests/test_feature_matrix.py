"""
Every feature on every kind of input.

The inputs are the cases that break code written for "ordinary" contexts:
no objects, no attributes, a single concept, duplicates, names with markup
and TeX characters, integer names, iceberg lattices, sparse input, pickled
lattices, and more than 64 attributes. The features are the layouts and
plotting backends, the drawing and lattice measures, the implication bases,
nested diagrams, file formats and the scikit-learn estimators (alone, in a
Pipeline and in a ColumnTransformer).

A feature has to work on each input and give a result consistent with the
lattice (one node per concept, one edge per cover, ...), or fail with the
documented ``ValueError``; never with an internal error.
"""

import io
import json
import math
import pickle
import re
import shutil
import warnings

import numpy as np
import pandas as pd
import pytest
import scipy.sparse
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from conceptflow import ConceptLattice, FormalContext, plot_lattice
from conceptflow.algorithms import (
    compute_canonical_basis,
    compute_luxenburger_basis,
    compute_minimal_generators,
    compute_proper_premise_basis,
)
from conceptflow.cluster import ConceptLatticeEstimator
from conceptflow.decomposition import ASSO, GreConD
from conceptflow.exploration import nested_combinations, nested_view
from conceptflow.feature_extraction import ConceptMembershipEncoder
from conceptflow.io import read_csv, read_cxt, read_json, write_csv, write_cxt, write_json
from conceptflow.metrics import (
    concept_levels,
    concept_stability,
    delta_measure,
    distributivity_index,
    drawing_quality_scores,
    incomparable_pairs,
    lattice_height,
    lattice_width,
    linearity_index,
    stability_bounds,
)
from conceptflow.rules import ImplicationBasisEstimator, ImplicationOutlierDetector
from conceptflow.visualization import plot_nested

SPECIAL_NAMES = [
    "a&b", "50%", "$x$", "snake_case", "{brace}", "back\\slash", "#hash", "~tilde",
    "^caret", '"quoted"', "<script>alert(1)</script>", "Ωmega ∧ ä", "😀", " spaced ",
    "", "tab\tname", "line\nbreak", "x,y;z",
]


def _context(incidence, objects=None, attributes=None):
    return FormalContext.from_array(np.asarray(incidence, dtype=bool), objects, attributes)


def _random(shape, seed, density=0.5):
    return np.random.default_rng(seed).random(shape) < density


def _special_names():
    incidence = _random((9, 9), seed=3)
    return _context(incidence, SPECIAL_NAMES[:9], SPECIAL_NAMES[9:])


def _numeric_names():
    frame = pd.DataFrame(_random((4, 3), seed=1), index=[10, 20, 30, 40], columns=[0, 1, 2])
    return FormalContext.from_dataframe(frame)


def _sparse():
    return FormalContext.from_array(scipy.sparse.csr_matrix(_random((6, 5), seed=4)))


def _duplicates():
    incidence = _random((4, 3), seed=5)
    return _context(incidence[[0, 1, 1, 2, 3, 3]][:, [0, 1, 1, 2, 2]])


# name -> (context, lattice options)
INPUTS = {
    "empty": (lambda: _context(np.zeros((0, 0))), {}),
    "no objects": (lambda: _context(np.zeros((0, 3))), {}),
    "no attributes": (lambda: _context(np.zeros((3, 0))), {}),
    "one concept": (lambda: _context(np.ones((3, 3))), {}),
    "no crosses": (lambda: _context(np.zeros((3, 3))), {}),
    "chain": (lambda: _context(np.tril(np.ones((4, 4)))), {}),
    "boolean algebra": (lambda: _context(~np.eye(4, dtype=bool)), {}),
    "duplicates": (_duplicates, {}),
    "numeric names": (_numeric_names, {}),
    "special names": (_special_names, {}),
    "sparse input": (_sparse, {}),
    "iceberg": (lambda: _context(_random((12, 6), seed=6)), {"min_support": 4}),
    "iceberg of one concept": (lambda: _context(_random((5, 4), seed=7)), {"min_support": 5}),
    "wide": (lambda: _context(_random((5, 70), seed=8, density=0.7)), {}),
    "medium": (lambda: _context(_random((30, 9), seed=9, density=0.4)), {}),
}
INPUT_NAMES = list(INPUTS)


def make_lattice(name):
    build, options = INPUTS[name]
    context = build()
    lattice = ConceptLattice.from_context(context, **options)
    if name == "sparse input":
        lattice = pickle.loads(pickle.dumps(lattice))  # also a pickled lattice
    return lattice


@pytest.fixture(params=INPUT_NAMES)
def lattice(request):
    return make_lattice(request.param)


# ---------------------------------------------------------------------------
# Layouts and drawings
# ---------------------------------------------------------------------------


def check_graph_data(graph_data, lattice, *, positioned, upward=True):
    nodes = graph_data.node_map()
    assert len(nodes) == lattice.n_concepts == len(graph_data.nodes)
    assert {n.concept for n in graph_data.nodes} == set(lattice.concepts)
    by_concept = {n.concept: n.node_id for n in graph_data.nodes}
    assert {(e.source, e.target) for e in graph_data.edges} == {
        (by_concept[lower], by_concept[upper]) for lower, upper in lattice.edges
    }
    if positioned:
        for node in graph_data.nodes:
            assert math.isfinite(node.x) and math.isfinite(node.y), node.node_id
        if upward:
            for edge in graph_data.edges:
                assert nodes[edge.target].y < nodes[edge.source].y, edge


@pytest.mark.parametrize("layout", ["dimflux", "layered", "none"])
def test_layouts(lattice, layout):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # e.g. "DimFlux of an iceberg is slow"
        graph_data = plot_lattice(lattice, backend="graph_data", layout=layout, timeout_ms=200)
    check_graph_data(graph_data, lattice, positioned=layout != "none")


@pytest.mark.parametrize("layout", ["dimflux", "layered"])
def test_drawing_quality_scores(lattice, layout):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        graph_data = plot_lattice(lattice, backend="graph_data", layout=layout, timeout_ms=200)
    scores = drawing_quality_scores(graph_data, worst_case=True)
    for name, value in scores.items():
        if name.startswith("vertical_axis_balance") and lattice.is_iceberg:
            continue  # documented: NaN without a unique top and bottom
        assert math.isfinite(value) and 0.0 <= value <= 1.0, (name, value)


def test_d3_page(lattice):
    html = plot_lattice(lattice, backend="d3", layout="layered").html
    # Names cannot end the script block or inject markup.
    assert html.count("</script>") == html.count("<script")
    assert "<script>alert(1)</script>" not in html
    # The data is valid JSON with one node per concept.
    data = json.loads(re.search(r"const graphData = (.*?);\n", html).group(1))
    assert len(data["nodes"]) == lattice.n_concepts
    assert len(data["edges"]) == len(lattice.edges)


# DejaVu Sans has no glyph for the tab in one of SPECIAL_NAMES.
@pytest.mark.filterwarnings("ignore:Glyph 9")
@pytest.mark.parametrize("fmt", ["svg", "pdf", "png"])
def test_matplotlib(lattice, fmt):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure = plot_lattice(lattice, backend="matplotlib", layout="layered")
    buffer = io.BytesIO()
    figure.savefig(buffer, format=fmt)
    plt.close(figure)
    assert buffer.getbuffer().nbytes > 100


def test_graphviz(lattice, tmp_path):
    figure = plot_lattice(lattice, backend="graphviz")
    assert figure.source.count("->") == len(lattice.edges)
    if shutil.which("dot"):
        svg = figure.render("svg").decode("utf-8")
        assert svg.count('class="node"') == lattice.n_concepts


def test_tikz(lattice, tmp_path):
    figure = plot_lattice(lattice, backend="tikz", layout="layered")
    body = re.sub(r"\\[{}]", "", figure.source)  # escaped braces
    assert body.count("{") == body.count("}")
    if shutil.which("pdflatex") or shutil.which("lualatex"):
        pdf = figure.write(tmp_path / "lattice.pdf")
        assert pdf.read_bytes().startswith(b"%PDF")


# ---------------------------------------------------------------------------
# Measures and bases
# ---------------------------------------------------------------------------


def test_lattice_measures(lattice):
    n = lattice.n_concepts
    for values in (concept_stability(lattice), delta_measure(lattice), concept_levels(lattice),
                   *stability_bounds(lattice)):
        assert len(values) == n and np.isfinite(values).all()
    assert 1 <= lattice_height(lattice) <= n
    assert 1 <= lattice_width(lattice) <= n
    assert 0 <= incomparable_pairs(lattice) <= n * (n - 1) // 2
    assert 0.0 <= linearity_index(lattice) <= 1.0
    if lattice.is_iceberg:
        with pytest.raises(ValueError, match="full lattice"):
            distributivity_index(lattice)
    else:
        assert 0.0 <= distributivity_index(lattice) <= 1.0


def test_implication_bases(lattice):
    context = lattice.context
    for basis in (compute_canonical_basis(context), compute_proper_premise_basis(context)):
        for implication in basis:
            assert len(implication.premise_names(context)) == len(implication.premise)
            assert len(implication.conclusion_names(context)) == len(implication.conclusion)
    rules = compute_luxenburger_basis(context, concepts=lattice.concepts)
    for rule in rules:
        assert len(rule.exception_names(context)) == len(rule.exceptions)
    generators = compute_minimal_generators(context)
    assert set(lattice.concepts) <= set(generators)


# ---------------------------------------------------------------------------
# Nested diagrams of the context's attributes split in two
# ---------------------------------------------------------------------------


def split_in_two(context):
    half = context.n_attributes // 2
    outer = context.subcontext(attributes=list(range(half)))
    inner = context.subcontext(attributes=list(range(half, context.n_attributes)))
    return outer, inner


@pytest.mark.filterwarnings("ignore:Glyph 9")
@pytest.mark.parametrize("backend", ["d3", "matplotlib", "tikz"])
def test_nested_diagrams(lattice, backend):
    outer, inner = split_in_two(lattice.context)
    if backend == "matplotlib":
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    figure = plot_nested(outer, inner=inner, backend=backend, timeout_ms=200)
    if backend == "d3":
        html = figure.html
        assert html.count("</script>") == html.count("<script")
        assert "<script>alert(1)</script>" not in html
    elif backend == "matplotlib":
        buffer = io.BytesIO()
        figure.savefig(buffer, format="svg")
        plt.close(figure)
    else:
        body = re.sub(r"\\[{}]", "", figure.source)
        assert body.count("{") == body.count("}")
    table = nested_combinations(nested_view(outer, inner))
    assert sorted(g for gs in table.introduced_objects for g in gs) == sorted(outer.objects)


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------


def test_files(lattice, tmp_path):
    context = lattice.context
    write_json(context, tmp_path / "c.json")
    assert read_json(tmp_path / "c.json") == context
    write_csv(context, tmp_path / "c.csv")
    assert read_csv(tmp_path / "c.csv") == context
    if any("\n" in name for name in context.objects + context.attributes):
        with pytest.raises(ValueError, match="line break"):
            write_cxt(context, tmp_path / "c.cxt")
    else:
        write_cxt(context, tmp_path / "c.cxt")
        assert read_cxt(tmp_path / "c.cxt") == context


# ---------------------------------------------------------------------------
# scikit-learn estimators on every input representation
# ---------------------------------------------------------------------------


def representations(context):
    """The same data as the types users pass to fit."""
    array = np.asarray(context.incidence)
    frame = context.to_dataframe()
    return {
        "bool array": array,
        "int array": array.astype(int),
        "float array": array.astype(float),
        "DataFrame": frame,
        "int DataFrame": frame.astype(int),
        "nullable DataFrame": frame.astype("boolean"),
        "sparse": scipy.sparse.csr_matrix(array),
    }


ESTIMATORS = {
    "lattice": lambda: ConceptLatticeEstimator(),
    "lattice/dataframe": lambda: ConceptLatticeEstimator(output="dataframe"),
    "lattice/sparse": lambda: ConceptLatticeEstimator(output="sparse"),
    "lattice/iceberg": lambda: ConceptLatticeEstimator(min_support=2),
    "lattice/stability": lambda: ConceptLatticeEstimator(select="stability", n_concepts=3),
    "encoder": lambda: ConceptMembershipEncoder(),
    "implications": lambda: ImplicationBasisEstimator(),
    "grecond": lambda: GreConD(),
    "asso": lambda: ASSO(),
}


def check_output(output, n_samples, names):
    shape = output.shape
    assert shape[0] == n_samples and shape[1] == len(names)
    if isinstance(output, pd.DataFrame):
        assert list(output.columns) == list(names)


@pytest.mark.parametrize("estimator_name", ESTIMATORS)
@pytest.mark.parametrize(
    "input_name", ["one concept", "no crosses", "duplicates", "numeric names",
                   "special names", "wide", "medium"]
)
def test_estimators(input_name, estimator_name):
    context = make_lattice(input_name).context
    for kind, X in representations(context).items():
        estimator = ESTIMATORS[estimator_name]()
        output = estimator.fit_transform(X)
        names = estimator.get_feature_names_out()
        check_output(output, context.n_objects, names)
        # Fitted estimators pickle and clone.
        copied = pickle.loads(pickle.dumps(estimator))
        again = copied.transform(X)
        if scipy.sparse.issparse(output):
            output, again = output.toarray(), again.toarray()
        assert np.array_equal(np.asarray(output), np.asarray(again)), kind
        assert clone(estimator).get_params() == estimator.get_params()
        # In a Pipeline.
        pipeline = Pipeline([("step", ESTIMATORS[estimator_name]())])
        check_output(pipeline.fit_transform(X), context.n_objects,
                     pipeline.get_feature_names_out())


@pytest.mark.parametrize("estimator_name", ESTIMATORS)
def test_estimators_in_a_column_transformer(estimator_name):
    context = make_lattice("medium").context
    frame = context.to_dataframe()
    frame["other"] = np.arange(len(frame))
    columns = list(context.attributes)
    transformer = ColumnTransformer(
        [("fca", ESTIMATORS[estimator_name](), columns)], remainder="passthrough"
    )
    output = transformer.fit_transform(frame)
    names = transformer.get_feature_names_out()
    assert output.shape == (len(frame), len(names))
    assert names[-1] == "remainder__other"


@pytest.mark.parametrize("estimator_name", ESTIMATORS)
@pytest.mark.parametrize("input_name", ["empty", "no objects"])
def test_estimators_refuse_data_without_samples(input_name, estimator_name):
    context = make_lattice(input_name).context
    with pytest.raises(ValueError):
        ESTIMATORS[estimator_name]().fit(np.asarray(context.incidence))


@pytest.mark.parametrize("estimator_name", ESTIMATORS)
def test_estimators_on_data_without_features(estimator_name):
    X = np.zeros((3, 0), dtype=bool)
    try:
        estimator = ESTIMATORS[estimator_name]().fit(X)
    except ValueError as error:  # scikit-learn's rule: at least one feature
        assert "feature" in str(error)
        return
    output = estimator.transform(X)
    assert output.shape[0] == 3 and output.shape[1] == len(estimator.get_feature_names_out())


def test_outlier_detector_on_every_input():
    for name in ["one concept", "no crosses", "duplicates", "special names", "wide", "medium"]:
        context = make_lattice(name).context
        for kind, X in representations(context).items():
            detector = ImplicationOutlierDetector().fit(X)
            scores = detector.score_samples(X)
            assert scores.shape == (context.n_objects,) and np.isfinite(scores).all(), (name, kind)
            assert set(np.unique(detector.predict(X))) <= {-1, 1}
            # One row per broken rule: as many as there are violations.
            explanation = detector.explain(X)
            assert set(explanation["object"]) == set(np.flatnonzero(scores < 0).tolist())
