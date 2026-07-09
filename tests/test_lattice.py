import numpy as np

from conceptflow import ConceptLattice, FormalContext

def make_context():
    return FormalContext(
        objects=["g1", "g2", "g3"],
        attributes=["a", "b"],
        incidence=[
            [True, False],
            [True, True],
            [False, True],
        ],
    )

def test_lattice_from_context():
    data = np.array([
        [1, 1, 0],
        [1, 0, 1],
        [0, 1, 1],
    ])

    ctx = FormalContext.from_array(data)
    lattice = ConceptLattice.from_context(ctx, algorithm="nextclosure")

    assert lattice.n_concepts == 8
    assert lattice.context == ctx


def test_lattice_edges_are_cover_relations():
    data = np.array([
        [1, 1, 0],
        [1, 0, 1],
        [0, 1, 1],
    ])

    ctx = FormalContext.from_array(data)
    lattice = ConceptLattice.from_context(ctx)

    assert len(lattice.edges) > 0

    for lower, upper in lattice.edges:
        assert lower.extent.issubset(upper.extent)
        assert lower != upper


def test_top_and_bottom_concepts():
    data = np.array([
        [1, 1, 0],
        [1, 0, 1],
        [0, 1, 1],
    ])

    ctx = FormalContext.from_array(data)
    lattice = ConceptLattice.from_context(ctx)

    top = lattice.top()
    bottom = lattice.bottom()

    assert top.extent == frozenset({0, 1, 2})
    assert bottom.extent == frozenset()

def test_concept_lattice_to_networkx():
    ctx = make_context()
    lattice = ConceptLattice.from_context(ctx)

    graph = lattice.to_networkx()

    assert graph.number_of_nodes() == lattice.n_concepts
    assert graph.number_of_edges() == len(lattice.edges)

    for edge in lattice.edges:
        assert graph.has_edge(*edge)

def test_parents_and_children():
    data = np.array([
        [1, 1, 0],
        [1, 0, 1],
        [0, 1, 1],
    ])

    ctx = FormalContext.from_array(data)
    lattice = ConceptLattice.from_context(ctx)

    bottom = lattice.bottom()
    top = lattice.top()

    assert len(lattice.parents(bottom)) > 0
    assert len(lattice.children(top)) > 0