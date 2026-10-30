import shutil

import numpy as np
import pytest

from conceptflow import ConceptLattice, FormalContext, lattice_to_graph_data, plot_lattice
from conceptflow.visualization import (
    GraphvizFigure,
    count_crossings,
    lattice_to_dot,
    layered_layout,
)


def random_lattice(seed, g=15, m=6):
    rng = np.random.default_rng(seed)
    return ConceptLattice.from_context(FormalContext.from_array(rng.random((g, m)) < 0.45))


@pytest.mark.parametrize("seed", range(10))
def test_layered_layout_places_upper_concepts_higher(seed):
    graph = layered_layout(lattice_to_graph_data(random_lattice(seed)))
    nodes = graph.node_map()

    assert all(node.x is not None and node.y is not None for node in graph.nodes)
    for edge in graph.edges:
        # Screen coordinates: the upper concept (target) has a smaller y.
        assert nodes[edge.target].y < nodes[edge.source].y
    assert len({(n.x, n.y) for n in graph.nodes}) == len(graph.nodes)


def test_crossing_reduction_helps_on_average():
    no_sweeps = with_sweeps = 0
    for seed in range(10):
        graph = lattice_to_graph_data(random_lattice(seed, g=18, m=7))
        no_sweeps += count_crossings(layered_layout(graph, sweeps=0))
        with_sweeps += count_crossings(layered_layout(graph))
    assert with_sweeps < no_sweeps


def test_plot_lattice_layered_option():
    graph = plot_lattice(random_lattice(0), backend="graph_data", layout="layered")
    assert graph.metadata["layout"] == "layered"

    with pytest.raises(ValueError, match="layered"):
        plot_lattice(random_lattice(0), backend="graph_data", layout="spring")


def test_dot_source_structure_and_escaping():
    ctx = FormalContext.from_array(
        [[1, 0], [1, 1]], objects=["a<b", "c&d"], attributes=["x>y", 'q"r']
    )
    lattice = ConceptLattice.from_context(ctx)
    dot = lattice_to_dot(lattice, title="Fish & chips")

    assert dot.startswith("digraph lattice {") and dot.rstrip().endswith("}")
    assert "rankdir=BT" in dot
    assert dot.count(" -> ") == len(lattice.edges)
    assert "a&lt;b" in dot and "c&amp;d" in dot and "x&gt;y" in dot and "q&quot;r" in dot


@pytest.mark.skipif(shutil.which("dot") is None, reason="Graphviz not installed")
@pytest.mark.parametrize("suffix", ["svg", "pdf", "png"])
def test_graphviz_figure_renders(tmp_path, suffix):
    fig = plot_lattice(random_lattice(1), backend="graphviz")

    assert isinstance(fig, GraphvizFigure)
    path = fig.write(tmp_path / f"lattice.{suffix}")
    assert path.stat().st_size > 0
    assert fig._repr_svg_().lstrip().startswith(("<?xml", "<svg", "<!--"))


def test_graphviz_figure_writes_dot_text(tmp_path):
    fig = plot_lattice(random_lattice(2), backend="graphviz")
    path = fig.write(tmp_path / "lattice.dot")
    assert path.read_text(encoding="utf-8") == fig.source


def test_iceberg_lattices_export_without_missing_labels():
    lattice = ConceptLattice.from_context(
        FormalContext.from_array(np.random.default_rng(3).random((20, 5)) < 0.5),
        min_support=8,
    )
    dot = lattice_to_dot(lattice)
    assert dot.count(" -> ") == len(lattice.edges)


def _edges_through_nodes(graph_data):
    """(edge, node) pairs where a straight edge passes through a node it does not connect."""
    pos = {n.node_id: np.array([n.x, n.y]) for n in graph_data.nodes}
    hits = []
    for e in graph_data.edges:
        p, q = pos[e.source], pos[e.target]
        v = q - p
        for node_id, r in pos.items():
            if node_id in (e.source, e.target):
                continue
            t = np.dot(r - p, v) / np.dot(v, v)
            if 0 < t < 1 and abs(v[0] * (r - p)[1] - v[1] * (r - p)[0]) / np.hypot(*v) < 1e-6:
                hits.append((e.source, e.target, node_id))
    return hits


@pytest.mark.parametrize(
    "name, layout",
    [("animals_en", "layered"), ("music_en", "layered"), ("missmarple_de", "dimflux")],
)
def test_no_edge_passes_through_a_node(name, layout):
    # These drew an edge straight through a third node before 0.5.0.
    from conceptflow.datasets import load_context

    lattice = ConceptLattice.from_context(load_context(name))
    graph_data = plot_lattice(lattice, backend="graph_data", layout=layout)
    assert _edges_through_nodes(graph_data) == []


def test_keeping_edges_off_nodes_moves_nodes_only_sideways():
    from conceptflow.datasets import load_context
    from conceptflow.visualization import _clearance

    graph_data = layered_layout(lattice_to_graph_data(
        ConceptLattice.from_context(load_context("animals_en"))
    ))
    limit = _clearance.MAX_NODES
    try:
        _clearance.MAX_NODES = 0
        plain = layered_layout(lattice_to_graph_data(
            ConceptLattice.from_context(load_context("animals_en"))
        ))
    finally:
        _clearance.MAX_NODES = limit
    assert _edges_through_nodes(plain)  # the case the step exists for
    assert [n.y for n in graph_data.nodes] == [n.y for n in plain.nodes]
    moved = [n for n, m in zip(graph_data.nodes, plain.nodes, strict=True) if n.x != m.x]
    assert 0 < len(moved) < len(plain.nodes) // 3


@pytest.mark.parametrize("min_support", [None, 2])
@pytest.mark.parametrize("seed", range(30))
def test_label_fallback_never_contradicts_the_lattice_labels(seed, min_support):
    # GraphData without stored labels (built by hand) derives them from the
    # drawn concepts; on iceberg lattices it must not invent object labels.
    from dataclasses import replace

    from conceptflow.visualization.static import _reduced_labels

    rng = np.random.default_rng(seed)
    context = FormalContext.from_array(
        (rng.random((int(rng.integers(1, 8)), int(rng.integers(1, 6)))) < 0.5).astype(int)
    )
    lattice = ConceptLattice.from_context(context, min_support=min_support)
    graph_data = lattice_to_graph_data(lattice)
    stripped = replace(graph_data, nodes=[
        replace(node, metadata={k: v for k, v in node.metadata.items()
                                if k not in ("attribute_labels", "object_labels")})
        for node in graph_data.nodes
    ])
    stored_attributes, stored_objects = _reduced_labels(graph_data)
    derived_attributes, derived_objects = _reduced_labels(stripped)
    for node in graph_data.nodes:
        assert sorted(derived_attributes[node.node_id]) == sorted(stored_attributes[node.node_id])
        assert set(derived_objects[node.node_id]) <= set(stored_objects[node.node_id])
        if min_support is None:
            assert sorted(derived_objects[node.node_id]) == sorted(stored_objects[node.node_id])


@pytest.mark.parametrize("layout", ["layered", "dimflux"])
def test_layout_positions_are_keyed_by_extent_and_upright(layout):
    from conceptflow.core import scale_measure
    from conceptflow.datasets import load_context
    from conceptflow.visualization import layout_positions

    context = load_context("livingbeings_en")
    lattice = ConceptLattice.from_context(context)
    positions = layout_positions(plot_lattice(lattice, backend="graph_data", layout=layout))

    assert set(positions) == {
        frozenset(context.objects[i] for i in c.extent) for c in lattice.concepts
    }
    top, bottom = max(positions, key=len), min(positions, key=len)
    assert positions[top][1] > positions[bottom][1]  # as drawn: top above bottom
    for concept in lattice.concepts:  # every upper neighbour is drawn higher
        extent = frozenset(context.objects[i] for i in concept.extent)
        for parent in lattice.parents(concept):
            above = frozenset(context.objects[i] for i in parent.extent)
            assert positions[above][1] > positions[extent][1]

    # The concepts of a scale-measure are found in its context's layout.
    scale = scale_measure(context, context.attributes[:3])
    coarse = layout_positions(
        plot_lattice(ConceptLattice.from_context(scale), backend="graph_data", layout=layout)
    )
    assert set(coarse) <= set(positions)
