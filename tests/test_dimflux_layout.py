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
