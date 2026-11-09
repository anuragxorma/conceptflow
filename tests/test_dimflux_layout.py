import numpy as np
import pytest

from conceptflow import ConceptLattice, FormalContext
from conceptflow.visualization import (
    lattice_to_graph_data,
    lattice_to_graph_data_dimflux,
    plot_lattice,
)


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


def test_lattice_to_graph_data_dimflux_handles_multi_word_names():
    # Object/attribute names with spaces (and words that collide with
    # Python keywords, e.g. "in") broke the previously vendored DimFlux.
    # Names are replaced by synthetic ones before calling DimFlux; this
    # guards that path.
    ctx = FormalContext.from_array(
        np.array([
            [1, 1, 0],
            [1, 0, 1],
            [0, 1, 1],
        ]),
        objects=["water weed", "field mouse", "garden snail"],
        attributes=["lives in water", "needs chlorophyll", "can move"],
    )
    lattice = ConceptLattice.from_context(ctx)

    graph_data = lattice_to_graph_data_dimflux(lattice)

    assert len(graph_data.nodes) == lattice.n_concepts
    for node in graph_data.nodes:
        assert node.x is not None
        assert node.y is not None

    # The real (space-containing) names must still show up in the output --
    # sanitizing names for DimFlux internally must not leak into labels.
    all_labels = " ".join(node.label for node in graph_data.nodes)
    assert any(
        name in all_labels for name in ("lives in water", "needs chlorophyll", "can move")
    )


def test_dimflux_handles_object_and_attribute_sharing_a_name():
    # DimFlux rejects duplicate names; ConceptFlow allows an object and an
    # attribute to share one.
    ctx = FormalContext.from_array(
        np.array([[1, 0], [1, 1]]),
        objects=["x", "y"],
        attributes=["x", "z"],
    )

    graph_data = lattice_to_graph_data_dimflux(ConceptLattice.from_context(ctx))

    assert all(node.x is not None for node in graph_data.nodes)


@pytest.mark.parametrize(
    "data",
    [
        np.array([[1, 1]]),  # single concept
        np.zeros((3, 0)),  # no attributes
        np.zeros((0, 2)),  # no objects
    ],
)
def test_dimflux_handles_degenerate_lattices(data):
    lattice = ConceptLattice.from_context(FormalContext.from_array(data))

    graph_data = lattice_to_graph_data_dimflux(lattice)

    assert len(graph_data.nodes) == lattice.n_concepts == 1
    assert (graph_data.nodes[0].x, graph_data.nodes[0].y) == (0.0, 0.0)


@pytest.mark.parametrize("seed", range(5))
def test_dimflux_coordinates_are_mapped_to_the_right_concepts(seed):
    # DimFlux drawings are additive with upward vectors, so in a correct
    # concept-to-coordinate mapping every cover edge points the same way.
    # A mix-up between concepts would almost surely break this.
    rng = np.random.default_rng(seed)
    lattice = ConceptLattice.from_context(
        FormalContext.from_array(rng.random((8, 6)) < 0.5)
    )

    graph_data = lattice_to_graph_data_dimflux(lattice)
    nodes = graph_data.node_map()

    for edge in graph_data.edges:
        # Screen coordinates: the upper concept has the smaller y.
        assert nodes[edge.target].y < nodes[edge.source].y

    positions = {(node.x, node.y) for node in graph_data.nodes}
    assert len(positions) == lattice.n_concepts


def test_plot_lattice_passes_timeout_to_dimflux():
    graph_data = plot_lattice(
        make_lattice(), backend="graph_data", layout="dimflux", timeout_ms=200
    )

    assert all(node.x is not None for node in graph_data.nodes)


def test_moving_nodes_off_edges_keeps_unrotated_iceberg_drawings_upright():
    # Iceberg lattices are drawn as DimFlux returns them (they have no single
    # bottom concept to rotate by). Moving nodes off edges used a tilted
    # axis for them, so heights changed and edges could point downwards.
    from dataclasses import replace

    from conceptflow.datasets import load_context
    from conceptflow.visualization._clearance import keep_edges_off_nodes
    from conceptflow.visualization.layered_layout import layered_layout

    lattice = ConceptLattice.from_context(load_context("tealady"), min_support=0.1)
    upright = layered_layout(lattice_to_graph_data(lattice))
    # The same upright coordinates, as an unrotated DimFlux drawing.
    drawing = replace(upright, metadata={**upright.metadata, "layout": "dimflux"})
    cleared = keep_edges_off_nodes(drawing)
    before, after = drawing.node_map(), cleared.node_map()
    assert all(after[i].y == before[i].y for i in before)  # heights are kept
    for edge in cleared.edges:
        assert after[edge.target].y < after[edge.source].y


def _stop_before_dimflux(monkeypatch):
    from conceptflow.visualization import dimflux_layout

    def stop(*args, **kwargs):
        raise RuntimeError("DimFlux would run now")

    monkeypatch.setattr(dimflux_layout, "_dimflux_coordinates", stop)


def test_dimflux_warns_when_an_iceberg_needs_a_large_full_lattice(monkeypatch):
    # DimFlux lays out the full lattice: 762 concepts here, for an iceberg
    # of 11, which took many minutes without a word.
    rng = np.random.default_rng(0)
    context = FormalContext.from_array(rng.random((60, 14)) < 0.45)
    iceberg = ConceptLattice.from_context(context, min_support=0.4)
    assert iceberg.n_concepts < 150
    _stop_before_dimflux(monkeypatch)
    with pytest.warns(UserWarning, match="full concept lattice"):
        with pytest.raises(RuntimeError, match="DimFlux would run now"):
            lattice_to_graph_data_dimflux(iceberg)


def test_dimflux_does_not_warn_for_an_iceberg_of_a_small_lattice(monkeypatch, recwarn):
    context = FormalContext.from_array(
        np.array([[0, 1, 0, 1], [1, 0, 0, 0], [1, 1, 0, 1], [1, 1, 1, 1], [0, 1, 1, 0]])
    )
    iceberg = ConceptLattice.from_context(context, min_support=2)
    assert iceberg.is_iceberg and iceberg.n_concepts > 2
    _stop_before_dimflux(monkeypatch)
    with pytest.raises(RuntimeError, match="DimFlux would run now"):
        lattice_to_graph_data_dimflux(iceberg)
    assert not [w for w in recwarn if "DimFlux" in str(w.message)]


def test_dimflux_accepts_numpy_integer_timeout():
    graph_data = lattice_to_graph_data_dimflux(make_lattice(), timeout_ms=np.int64(100))
    assert all(node.x is not None for node in graph_data.nodes)
