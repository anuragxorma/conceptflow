import numpy as np
import pytest

from conceptflow import Concept, FormalContext
from conceptflow.algorithms.enumeration import (
    enumerate_concepts_bruteforce,
    enumerate_concepts_nextclosure,
    enumerate_concepts_closebyone,
)


def test_bruteforce_and_nextclosure_return_same_concepts():
    data = np.array([
        [1, 1, 0],
        [1, 0, 1],
        [0, 1, 1],
    ])

    ctx = FormalContext.from_array(data)

    brute_force = set(enumerate_concepts_bruteforce(ctx))
    next_closure = set(enumerate_concepts_nextclosure(ctx))

    assert brute_force == next_closure


def test_enumerated_concepts_are_valid():
    data = np.array([
        [1, 1, 0],
        [1, 0, 1],
        [0, 1, 1],
    ])

    ctx = FormalContext.from_array(data)
    concepts = enumerate_concepts_nextclosure(ctx)

    for concept in concepts:
        assert ctx.attribute_derivation(concept.intent) == concept.extent
        assert ctx.object_derivation(concept.extent) == concept.intent


def test_known_number_of_concepts():
    data = np.array([
        [1, 1, 0],
        [1, 0, 1],
        [0, 1, 1],
    ])

    ctx = FormalContext.from_array(data)
    concepts = enumerate_concepts_nextclosure(ctx)

    assert len(concepts) == 8


def test_specific_concept_exists():
    data = np.array([
        [1, 1, 0],
        [1, 0, 1],
        [0, 1, 1],
    ])

    ctx = FormalContext.from_array(data)
    concepts = set(enumerate_concepts_nextclosure(ctx))

    assert Concept(
        extent=frozenset({0, 1}),
        intent=frozenset({0}),
    ) in concepts

def test_closebyone_matches_bruteforce_and_nextclosure():
    data = np.array([
        [1, 1, 0],
        [1, 0, 1],
        [0, 1, 1],
    ])

    ctx = FormalContext.from_array(data)

    brute_force = set(enumerate_concepts_bruteforce(ctx))
    next_closure = set(enumerate_concepts_nextclosure(ctx))
    close_by_one = set(enumerate_concepts_closebyone(ctx))

    assert close_by_one == brute_force
    assert close_by_one == next_closure

@pytest.mark.parametrize("seed", range(10))
def test_closebyone_pruning_keeps_exactly_the_frequent_concepts(seed):
    from conceptflow.algorithms._bitset import BitsetContext, to_indices
    from conceptflow.algorithms.enumeration import (
        _closebyone_bitsets,
        enumerate_concepts_bruteforce,
    )

    rng = np.random.default_rng(seed)
    ctx = FormalContext.from_array(rng.random((12, 6)) < 0.5)
    everything = enumerate_concepts_bruteforce(ctx)

    top = max(everything, key=lambda c: len(c.extent))
    for threshold in range(0, 14):
        pruned = _closebyone_bitsets(BitsetContext(ctx), min_extent_size=threshold)
        # The top concept is always kept.
        expected = {
            (c.extent, c.intent)
            for c in everything
            if len(c.extent) >= threshold or c == top
        }
        assert {(to_indices(e), to_indices(i)) for e, i in pruned} == expected
