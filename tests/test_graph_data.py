import numpy as np

from conceptflow import ConceptLattice, FormalContext
from conceptflow.visualization import GraphData, lattice_to_graph_data, plot_lattice


def make_lattice():
    ctx = FormalContext.from_array(
        np.array([
            [1, 1, 0],
            [1, 0, 1],
            [0, 1, 1],
        ]),
        objects=["g1", "g2", "g3"],
        attributes=["m1", "m2", "m3"],
    )

    return ConceptLattice.from_context(ctx)


def test_lattice_to_graph_data_creates_one_node_per_concept():
    lattice = make_lattice()

    graph_data = lattice_to_graph_data(lattice)

    assert isinstance(graph_data, GraphData)
    assert len(graph_data.nodes) == lattice.n_concepts


def test_lattice_to_graph_data_creates_one_edge_per_hasse_edge():
    lattice = make_lattice()

    graph_data = lattice_to_graph_data(lattice)

    assert len(graph_data.edges) == len(lattice.edges)


def test_lattice_to_graph_data_uses_local_ids_by_default():
    lattice = make_lattice()

    graph_data = lattice_to_graph_data(lattice)

    assert graph_data.node_ids() == [
        f"c{i}"
        for i in range(lattice.n_concepts)
    ]


def test_lattice_to_graph_data_can_use_stable_ids():
    lattice = make_lattice()

    graph_data = lattice_to_graph_data(
        lattice,
        stable_ids=True,
    )

    expected_ids = [
        concept.stable_id()
        for concept in lattice.concepts
    ]

    assert graph_data.node_ids() == expected_ids


def test_lattice_to_graph_data_metadata_contains_names_and_indices():
    lattice = make_lattice()

    graph_data = lattice_to_graph_data(lattice)

    node = graph_data.nodes[0]

    assert "extent" in node.metadata
    assert "intent" in node.metadata
    assert "extent_indices" in node.metadata
    assert "intent_indices" in node.metadata


def test_lattice_to_graph_data_hover_text_uses_object_and_attribute_names():
    lattice = make_lattice()

    graph_data = lattice_to_graph_data(lattice)

    hover_texts = "\n".join(node.hover_text for node in graph_data.nodes)

    assert "g1" in hover_texts
    assert "m1" in hover_texts


def test_plot_lattice_graph_data_backend():
    lattice = make_lattice()

    graph_data = plot_lattice(
        lattice,
        backend="graph_data",
        layout="none",
    )

    assert isinstance(graph_data, GraphData)


def test_plot_lattice_rejects_unknown_layout():
    lattice = make_lattice()

    try:
        plot_lattice(lattice, layout="invalid_layout")
    except ValueError as exc:
        assert 'Unknown layout "invalid_layout"' in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid layout")


def test_plot_lattice_rejects_unknown_backend():
    lattice = make_lattice()

    try:
        plot_lattice(lattice, backend="unknown")
    except ValueError as exc:
        assert "Unknown visualization backend" in str(exc)
    else:
        raise AssertionError("Expected ValueError")