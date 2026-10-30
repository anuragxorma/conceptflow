import math

import networkx as nx
import numpy as np
import pytest

from conceptflow import ConceptLattice, FormalContext
from conceptflow.metrics import (
    bottleneck_clearance_radius,
    crossing_angle_score,
    drawing_quality_scores,
    edge_crossing_score,
    edge_edge_conflict_score,
    edge_length_uniformity_score,
    layer_consistency_score,
    nested_suitability_score,
    node_edge_conflict_score,
    node_node_conflict_score,
    slope_economy_score,
    slope_parallelism_score,
    slope_standard_score,
    slope_verticality_score,
    vertical_axis_balance_score,
    visual_chain_linearity_score,
)
from conceptflow.visualization import lattice_to_graph_data, layered_layout


def drawing(edges, positions):
    """A networkx drawing; an edge (u, v) means u < v."""
    graph = nx.DiGraph()
    for node, (x, y) in positions.items():
        graph.add_node(node, x=float(x), y=float(y))
    graph.add_edges_from(edges)
    return graph


CHAIN = [("a", "b"), ("b", "c"), ("c", "d")]
DIAMOND = [("0", "a"), ("0", "b"), ("a", "1"), ("b", "1")]
SQUARE = {"0": (0, 0), "a": (-1, 1), "b": (1, 1), "1": (0, 2)}

# Two drawings of the reference implementation's data set (dataset m4,
# lattices 66 and 64) with their published scores.
REDRAW_66 = drawing(
    [
        ("1", "0"),
        ("2", "0"),
        ("3", "0"),
        ("4", "1"),
        ("5", "2"),
        ("5", "3"),
        ("6", "3"),
        ("6", "1"),
        ("7", "6"),
        ("7", "4"),
        ("8", "2"),
        ("8", "4"),
        ("9", "5"),
        ("9", "7"),
        ("9", "8"),
    ],
    {
        "0": (0.21529754027218387, -8.408589305792294),
        "1": (0.20341170645739762, -6.742531786295462),
        "2": (-1.4101196213003973, -6.274853716830179),
        "3": (0.7332709301845642, -5.621321449485114),
        "4": (2.1524221623861473, -5.177018254152361),
        "5": (0.0379838305657976, -2.6442534603662557),
        "6": (-1.9516295755556843, -4.375915753392155),
        "7": (1.312708899851348, -2.6443373624231943),
        "8": (-1.0693943248305633, -2.69355504688711),
        "9": (-0.22395154803079387, -0.698219766566126),
    },
)
REDRAW_66_SCORES = {
    "visual_chain_linearity": 0.7741483852740543,
    "node_edge_conflict": 0.8131928679416219,
    "crossing_angle": 0.667660773810441,
    "edge_edge_conflict": 0.992660403292987,
    "edge_crossing": 0.4444444444444444,
    "edge_length_uniformity": 0.9565441359162923,
    "layer_consistency": 0.9459288325011164,
    "slope_economy": 0.21428571428571427,
    "slope_parallelism": 0.05263157894736842,
    "slope_standard": 0.8073236635776224,
    "slope_verticality": 0.7634463350115429,
    "nested_suitability": 0.38886370317563645,
    "node_node_conflict": 0.9834206493639852,
    "vertical_axis_balance": 0.43404555671700135,
    "node_node_conflict_min": 0.9449473033292813,
    "node_edge_conflict_min": 0.38886370317563645,
    "edge_edge_conflict_min": 0.9266040329298707,
    "crossing_angle_min": 0.43587031632195317,
}

# Nodes 1 and 3 coincide.
SUGIYAMA_64 = drawing(
    [
        ("1", "0"),
        ("2", "0"),
        ("3", "2"),
        ("4", "1"),
        ("4", "3"),
        ("5", "0"),
        ("6", "1"),
        ("6", "5"),
        ("7", "2"),
        ("7", "5"),
        ("8", "4"),
        ("8", "6"),
        ("8", "7"),
    ],
    {
        "0": (1, 4),
        "1": (0, 2),
        "2": (1, 3),
        "3": (0, 2),
        "4": (0, 1),
        "5": (1, 2),
        "6": (1, 1),
        "7": (2, 1),
        "8": (1, 0),
    },
)
SUGIYAMA_64_SCORES = {
    "visual_chain_linearity": 0.8296070210069572,
    "node_edge_conflict": 0.6511959798862832,
    "crossing_angle": 0.47073817647155236,
    "edge_edge_conflict": 0.7777777777777778,
    "edge_crossing": 0.8333333333333334,
    "edge_length_uniformity": 0.8525669465384754,
    "layer_consistency": 0.9519230769230769,
    "slope_economy": 0.6,
    "slope_parallelism": 0.3333333333333333,
    "slope_standard": 0.971646099376255,
    "slope_verticality": 1.0,
    "nested_suitability": 0.0,
    "node_node_conflict": 0.7777777777777778,
    "vertical_axis_balance": 0.25,
    "node_node_conflict_min": 0.0,
    "node_edge_conflict_min": 0.0,
    "edge_edge_conflict_min": 0.0,
    "crossing_angle_min": 0.20483276469913358,
}


def laid_out_lattice(seed):
    rng = np.random.default_rng(seed)
    ctx = FormalContext.from_array(rng.random((7, 5)) < 0.5)
    return layered_layout(lattice_to_graph_data(ConceptLattice.from_context(ctx)))


@pytest.mark.parametrize(
    "graph, expected",
    [(REDRAW_66, REDRAW_66_SCORES), (SUGIYAMA_64, SUGIYAMA_64_SCORES)],
)
def test_scores_match_reference_implementation(graph, expected):
    scores = drawing_quality_scores(graph, worst_case=True)
    assert scores.keys() == expected.keys()
    for name, value in expected.items():
        assert scores[name] == pytest.approx(value, abs=1e-9), name


def test_report_matches_individual_metrics():
    graph_data = laid_out_lattice(0)
    scores = drawing_quality_scores(graph_data)
    assert len(scores) == 14
    assert scores["slope_economy"] == slope_economy_score(graph_data)
    assert scores["vertical_axis_balance"] == vertical_axis_balance_score(graph_data)
    assert "crossing_angle_min" not in scores


@pytest.mark.parametrize("seed", range(10))
def test_graph_data_and_networkx_inputs_agree_and_scores_are_in_range(seed):
    graph_data = laid_out_lattice(seed)
    graph = drawing(
        [(e.source, e.target) for e in graph_data.edges],
        {n.node_id: (n.x, n.y) for n in graph_data.nodes},
    )
    scores = drawing_quality_scores(graph_data, worst_case=True)
    assert scores == drawing_quality_scores(graph, worst_case=True)
    assert all(0.0 <= value <= 1.0 for value in scores.values())


def test_scores_do_not_depend_on_scale_translation_or_mirroring():
    positions = {n: (d["x"], d["y"]) for n, d in REDRAW_66.nodes(data=True)}
    original = drawing_quality_scores(REDRAW_66, worst_case=True)

    moved = drawing(REDRAW_66.edges, {n: (3 * x + 7, 3 * y + 1) for n, (x, y) in positions.items()})
    assert drawing_quality_scores(moved, worst_case=True) == pytest.approx(original, abs=1e-9)

    # Mirrored: the greedy grouping of directions sweeps the other way.
    mirrored = drawing(REDRAW_66.edges, {n: (x, -y) for n, (x, y) in positions.items()})
    scores = drawing_quality_scores(mirrored, worst_case=True)
    assert scores.pop("edge_length_uniformity") == pytest.approx(0.9448520224984779)
    assert scores == pytest.approx(
        {k: v for k, v in original.items() if k != "edge_length_uniformity"}, abs=1e-9
    )


def test_vertical_chain_is_ideal():
    graph = drawing(CHAIN, {"a": (0, 0), "b": (0, 1), "c": (0, 2), "d": (0, 3)})
    assert all(value == 1.0 for value in drawing_quality_scores(graph).values())


def test_bent_chain_loses_linearity():
    graph = drawing(CHAIN, {"a": (0, 0), "b": (1, 1), "c": (0, 2), "d": (1, 3)})
    assert visual_chain_linearity_score(graph) == pytest.approx(0.5)


def test_square_diamond():
    graph = drawing(DIAMOND, SQUARE)
    # Opposite sides of the square are parallel, the order needs two directions.
    assert slope_parallelism_score(graph) == 1.0
    assert slope_economy_score(graph) == 1.0
    assert edge_length_uniformity_score(graph) == 1.0
    assert vertical_axis_balance_score(graph) == 1.0
    assert layer_consistency_score(graph) == pytest.approx(1.0)
    assert slope_standard_score(graph) == 1.0  # 45° edges


def test_skewed_diamond_is_not_parallel_and_unbalanced():
    graph = drawing(DIAMOND, {"0": (0, 0), "a": (-1, 1), "b": (2, 1), "1": (0, 2)})
    assert slope_parallelism_score(graph) == 0.0
    assert slope_economy_score(graph) == pytest.approx(0.5)
    assert vertical_axis_balance_score(graph) < 1.0


def test_horizontal_edge_has_zero_verticality():
    graph = drawing([("a", "b")], {"a": (0, 0), "b": (1, 0)})
    assert slope_verticality_score(graph) == 0.0
    graph = drawing([("a", "b")], {"a": (0, 0), "b": (2, math.tan(math.radians(15)) * 2)})
    assert slope_verticality_score(graph) == pytest.approx(0.5)


def test_crossings():
    # Two independent 2-chains drawn as an X crossing at a right angle.
    graph = drawing([("a", "b"), ("c", "d")], {"a": (0, 0), "b": (2, 2), "c": (2, 0), "d": (0, 2)})
    assert edge_crossing_score(graph) == 0.0
    assert crossing_angle_score(graph) == pytest.approx(1.0)

    # Collinear edges overlapping along a segment: the worst crossing.
    graph = drawing([("a", "b"), ("c", "d")], {"a": (0, 0), "b": (0, 2), "c": (0, 1), "d": (0, 3)})
    assert edge_crossing_score(graph) == 0.0
    assert crossing_angle_score(graph) == 0.0

    # Touching in one point (a T-junction) is a crossing too.
    graph = drawing([("a", "b"), ("c", "d")], {"a": (0, 0), "b": (0, 2), "c": (0, 1), "d": (2, 3)})
    assert edge_crossing_score(graph) == 0.0
    assert crossing_angle_score(graph) == pytest.approx(0.5)

    # Nearly, but not exactly, touching.
    graph = drawing(
        [("a", "b"), ("c", "d")], {"a": (0, 0), "b": (0, 2), "c": (1e-9, 1), "d": (2, 3)}
    )
    assert edge_crossing_score(graph) == 1.0


def test_coincident_nodes_and_node_on_edge():
    graph = drawing(CHAIN, {"a": (0, 0), "b": (0, 1), "c": (0, 1), "d": (0, 3)})
    assert node_node_conflict_score(graph, aggregate="min") == 0.0
    assert bottleneck_clearance_radius(graph) == 0.0
    assert nested_suitability_score(graph) == 0.0

    # The node b of the diamond lies on the edge 0 -> a.
    graph = drawing(DIAMOND, {"0": (0, 0), "a": (0, 2), "b": (0, 1), "1": (0, 3)})
    assert node_edge_conflict_score(graph, aggregate="min") == 0.0
    assert edge_edge_conflict_score(graph, aggregate="min") == 0.0


def test_parameters_change_scores():
    graph = drawing(DIAMOND, {"0": (0, 0), "a": (-1, 1), "b": (1.2, 1), "1": (0, 2)})
    assert slope_parallelism_score(graph) < slope_parallelism_score(graph, angle_tolerance=20)
    assert node_node_conflict_score(graph, threshold_factor=2.0) < 1.0
    assert slope_standard_score(graph, canonical_angles=[45]) < slope_standard_score(graph)


def test_invalid_input():
    with pytest.raises(TypeError, match="GraphData"):
        slope_economy_score([[0, 1]])
    with pytest.raises(ValueError, match="coordinates"):
        lattice = ConceptLattice.from_context(FormalContext.from_array([[1, 0]]))
        slope_economy_score(lattice_to_graph_data(lattice))
    with pytest.raises(ValueError, match="'x' and 'y'"):
        slope_economy_score(nx.DiGraph([("a", "b")]))
    with pytest.raises(ValueError, match="acyclic"):
        slope_economy_score(drawing([("a", "b"), ("b", "a")], {"a": (0, 0), "b": (0, 1)}))
    with pytest.raises(ValueError, match="NaN"):
        slope_economy_score(drawing([], {"a": (0, np.nan)}))
    with pytest.raises(ValueError, match="no nodes"):
        slope_economy_score(nx.DiGraph())

    graph = drawing(DIAMOND, SQUARE)
    with pytest.raises(ValueError, match="threshold_factor"):
        node_node_conflict_score(graph, threshold_factor=0)
    with pytest.raises(ValueError, match="angle_tolerance"):
        slope_parallelism_score(graph, angle_tolerance=-1)
    with pytest.raises(ValueError, match="aggregate"):
        crossing_angle_score(graph, aggregate="max")
    with pytest.raises(ValueError, match="canonical_angles"):
        slope_standard_score(graph, canonical_angles=[100])


def test_balance_needs_top_and_bottom():
    graph = drawing([("a", "c"), ("b", "c")], {"a": (0, 0), "b": (1, 0), "c": (0, 1)})
    with pytest.raises(ValueError, match="unique top and bottom"):
        vertical_axis_balance_score(graph)
    # The other metrics do not need them.
    assert slope_economy_score(graph) == 1.0
