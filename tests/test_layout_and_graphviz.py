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
    assert path.read_text() == fig.source


def test_iceberg_lattices_export_without_missing_labels():
    lattice = ConceptLattice.from_context(
        FormalContext.from_array(np.random.default_rng(3).random((20, 5)) < 0.5),
        min_support=8,
    )
    dot = lattice_to_dot(lattice)
    assert dot.count(" -> ") == len(lattice.edges)
