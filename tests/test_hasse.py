import numpy as np
import pytest

from conceptflow import ConceptLattice, FormalContext
from conceptflow.algorithms import compute_hasse_edges, enumerate_concepts, is_cover


def naive_edges(concepts):
    return {
        (lower, upper)
        for lower in concepts
        for upper in concepts
        if is_cover(lower, upper, concepts)
    }


@pytest.mark.parametrize("seed", range(30))
def test_hasse_edges_match_naive_cover_relation(seed):
    rng = np.random.default_rng(seed)
    data = rng.random((int(rng.integers(1, 9)), int(rng.integers(0, 7)))) < 0.5
    ctx = FormalContext.from_array(data)
    concepts = enumerate_concepts(ctx)
    expected = naive_edges(concepts)

    assert set(compute_hasse_edges(concepts, context=ctx)) == expected
    assert set(compute_hasse_edges(concepts)) == expected


@pytest.mark.parametrize("seed", range(10))
def test_hasse_edges_of_concept_subset_fall_back_correctly(seed):
    # A subset of the lattice is not closed under meets, so the fast
    # context-based path must detect this and fall back.
    rng = np.random.default_rng(seed)
    ctx = FormalContext.from_array(rng.random((8, 6)) < 0.5)
    subset = enumerate_concepts(ctx)[::2]

    assert set(compute_hasse_edges(subset, context=ctx)) == naive_edges(subset)


def test_hasse_edges_are_deterministically_ordered():
    rng = np.random.default_rng(0)
    ctx = FormalContext.from_array(rng.random((10, 6)) < 0.5)
    concepts = enumerate_concepts(ctx)
    position = {c: i for i, c in enumerate(concepts)}

    edges = compute_hasse_edges(concepts, context=ctx)

    keys = [(position[lower], position[upper]) for lower, upper in edges]
    assert keys == sorted(keys)


def test_lattice_edges_and_neighbours_are_cached():
    rng = np.random.default_rng(1)
    lattice = ConceptLattice.from_context(
        FormalContext.from_array(rng.random((12, 7)) < 0.5)
    )

    assert lattice.edges is lattice.edges

    for concept in lattice.concepts:
        assert lattice.parents(concept) == {
            upper for lower, upper in lattice.edges if lower == concept
        }
        assert lattice.children(concept) == {
            lower for lower, upper in lattice.edges if upper == concept
        }
