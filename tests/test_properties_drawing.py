"""
Property-based tests of the drawing quality scores.

- Every score is a number in [0, 1] for any drawing, also with nodes on a
  coarse grid (coincident nodes, nodes on edges, collinear edges);
  ``vertical_axis_balance`` is NaN exactly when there is no unique top and
  bottom.
- Scores describe the shape of a drawing, not where it is or how large:
  moving it and scaling it, by amounts that are exact in floating point, do
  not change them, and neither does mirroring (except
  ``edge_length_uniformity``, whose greedy grouping of directions sweeps the
  other way; see test_drawing_quality.py).
- A drawing stored at an angle is scored as it is shown, turned upright, so
  the angle does not matter (except ``vertical_axis_balance``: see
  test_vertical_axis_balance_of_drawings_at_an_angle).
"""

import math
from dataclasses import replace

import pytest

pytest.importorskip("hypothesis")

from hypothesis import given  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

from conceptflow import ConceptLattice, plot_lattice  # noqa: E402
from conceptflow.metrics import drawing_quality_scores, vertical_axis_balance_score  # noqa: E402
from conceptflow.visualization import GraphData  # noqa: E402
from tests.property_helpers import context_of, incidences  # noqa: E402


def moved(graph_data, transform, layout=None):
    metadata = dict(graph_data.metadata)
    if layout is not None:
        metadata["layout"] = layout
    nodes = []
    for node in graph_data.nodes:
        x, y = transform(node.x, node.y)
        nodes.append(replace(node, x=float(x), y=float(y)))
    return GraphData(nodes=nodes, edges=graph_data.edges, metadata=metadata)


@st.composite
def drawings(draw, grid=True):
    """Layered drawings of random lattices, sometimes with x on a coarse grid."""
    incidence = draw(incidences(min_objects=1, max_objects=6, max_attributes=5))
    lattice = ConceptLattice.from_context(
        context_of(incidence), min_support=draw(st.sampled_from([None, None, 2]))
    )
    graph_data = plot_lattice(lattice, backend="graph_data", layout="layered")
    if grid and draw(st.booleans(), label="grid"):
        xs = draw(st.lists(st.integers(-3, 3), min_size=len(graph_data.nodes),
                           max_size=len(graph_data.nodes)), label="grid x")
        positions = iter(xs)
        graph_data = moved(graph_data, lambda x, y: (10.0 * next(positions), y))
    return lattice, graph_data


def assert_same_scores(first, second, tolerance, skip=()):
    assert first.keys() == second.keys()
    for name in first:
        if name in skip:
            continue
        a, b = first[name], second[name]
        if math.isnan(a) or math.isnan(b):
            assert math.isnan(a) and math.isnan(b), name
        else:
            assert a == pytest.approx(b, abs=tolerance), name


@given(drawings())
def test_scores_are_numbers_between_zero_and_one(data):
    _, graph_data = data
    scores = drawing_quality_scores(graph_data, worst_case=True)
    sources = {edge.source for edge in graph_data.edges}
    targets = {edge.target for edge in graph_data.edges}
    ids = graph_data.node_ids()
    tops = [n for n in ids if n not in sources]
    bottoms = [n for n in ids if n not in targets]
    for name, value in scores.items():
        if name == "vertical_axis_balance" and (len(tops) != 1 or len(bottoms) != 1):
            assert math.isnan(value)
        else:
            assert 0.0 <= value <= 1.0, (name, value)


@given(drawings(), st.integers(-64, 64), st.integers(-64, 64), st.sampled_from([0.25, 2, 8]),
       st.booleans())
def test_scores_do_not_depend_on_position_size_or_side(data, dx, dy, factor, mirror):
    _, graph_data = data
    original = drawing_quality_scores(graph_data, worst_case=True)
    sign = -1 if mirror else 1
    changed = moved(graph_data, lambda x, y: (sign * factor * x + dx, factor * y + dy))
    skip = ("edge_length_uniformity",) if mirror else ()
    assert_same_scores(original, drawing_quality_scores(changed, worst_case=True), 1e-9, skip)


@given(drawings(grid=False), st.floats(0, 2 * math.pi))
def test_drawings_at_an_angle_are_scored_upright(data, angle):
    lattice, graph_data = data
    if lattice.is_iceberg or lattice.n_concepts < 2:
        return  # shown as stored, not turned
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    # Both are turned upright before scoring ("custom": not a layered layout).
    upright = drawing_quality_scores(moved(graph_data, lambda x, y: (x, y), "custom"))
    turned = drawing_quality_scores(
        moved(graph_data, lambda x, y: (cos_a * x - sin_a * y, sin_a * x + cos_a * y), "custom")
    )
    assert_same_scores(upright, turned, 1e-6, skip=("vertical_axis_balance",))


# Known problem, left to the metrics' authors: turning a drawing upright
# leaves nodes on the axis a rounding error (~1e-15) to one side, and
# vertical_axis_balance_score counts them as left or right nodes. The M3
# lattice (np.eye(3)) drawn symmetrically scores 1.0 upright but 0.5 when
# stored at 10 or 30 degrees, as DimFlux layouts are.
@pytest.mark.xfail(reason="nodes on the axis are counted as left or right after rotation")
@given(drawings(grid=False), st.floats(0, 2 * math.pi))
def test_vertical_axis_balance_of_drawings_at_an_angle(data, angle):
    lattice, graph_data = data
    if lattice.is_iceberg or lattice.n_concepts < 2:
        return
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    upright = vertical_axis_balance_score(moved(graph_data, lambda x, y: (x, y), "custom"))
    turned = vertical_axis_balance_score(
        moved(graph_data, lambda x, y: (cos_a * x - sin_a * y, sin_a * x + cos_a * y), "custom")
    )
    assert turned == pytest.approx(upright, abs=1e-6)
