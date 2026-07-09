import numpy as np

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