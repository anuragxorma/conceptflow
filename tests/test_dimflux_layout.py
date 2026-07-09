import numpy as np
import pytest

from conceptflow import ConceptLattice, FormalContext
from conceptflow.visualization import lattice_to_graph_data_dimflux, plot_lattice


def make_lattice():
    ctx = FormalContext.from_array(
        np.array([
            [1, 0],
            [1, 1],
        ]),
        objects=["g1", "g2"],
        attributes=["m1", "m2"],
    )

    return ConceptLattice.from_context(ctx)


def test_lattice_to_graph_data_dimflux_assigns_coordinates():
    lattice = make_lattice()

    graph_data = lattice_to_graph_data_dimflux(lattice)

    assert len(graph_data.nodes) == lattice.n_concepts
    assert len(graph_data.edges) == len(lattice.edges)
    assert graph_data.metadata["layout"] == "dimflux"

    for node in graph_data.nodes:
        assert node.x is not None
        assert node.y is not None
        assert node.metadata["layout"] == "dimflux"


def test_lattice_to_graph_data_dimflux_preserves_node_order():
    lattice = make_lattice()

    graph_data = lattice_to_graph_data_dimflux(lattice)

    assert graph_data.node_ids() == [
        f"c{i}"
        for i in range(lattice.n_concepts)
    ]


def test_lattice_to_graph_data_dimflux_supports_stable_ids():
    lattice = make_lattice()

    graph_data = lattice_to_graph_data_dimflux(
        lattice,
        stable_ids=True,
    )

    assert graph_data.node_ids() == [
        concept.stable_id()
        for concept in lattice.concepts
    ]


def test_plot_lattice_dimflux_layout_returns_graph_data_with_coordinates():
    lattice = make_lattice()

    graph_data = plot_lattice(
        lattice,
        backend="graph_data",
        layout="dimflux",
    )

    assert graph_data.metadata["layout"] == "dimflux"
    assert all(node.x is not None for node in graph_data.nodes)
    assert all(node.y is not None for node in graph_data.nodes)