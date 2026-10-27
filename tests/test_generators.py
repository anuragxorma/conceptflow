import itertools

import numpy as np
import pytest

from conceptflow import FormalContext
from conceptflow.algorithms import (
    compute_minimal_generators,
    compute_proper_premise_basis,
    enumerate_concepts,
)


def random_context(seed):
    rng = np.random.default_rng(seed)
    g, m = int(rng.integers(0, 10)), int(rng.integers(0, 7))
    return FormalContext.from_array(rng.random((g, m)) < rng.uniform(0.1, 0.9))


def all_subsets(n):
    return [frozenset(s) for r in range(n + 1) for s in itertools.combinations(range(n), r)]


@pytest.mark.parametrize("seed", range(40))
def test_minimal_generators_match_definition(seed):
    ctx = random_context(seed)
    subsets = all_subsets(ctx.n_attributes)
    extent = {s: ctx.attribute_derivation(s) for s in subsets}
    keys = {s for s in subsets if all(extent[s - {a}] != extent[s] for a in s)}

    generators = compute_minimal_generators(ctx)

    assert set(generators) == set(enumerate_concepts(ctx))
    assert {k for ks in generators.values() for k in ks} == keys
    for concept, concept_keys in generators.items():
        assert all(ctx.attribute_derivation(k) == concept.extent for k in concept_keys)
        assert [len(k) for k in concept_keys] == sorted(len(k) for k in concept_keys)


@pytest.mark.parametrize("seed", range(40))
def test_proper_premise_basis_matches_definition_and_is_direct(seed):
    ctx = random_context(seed)
    subsets = all_subsets(ctx.n_attributes)

    expected = {}
    for m in range(ctx.n_attributes):
        premises = [s for s in subsets if m not in s and m in ctx.closure(s)]
        for s in premises:
            if not any(q < s for q in premises):
                expected[s] = expected.get(s, frozenset()) | {m}

    basis = compute_proper_premise_basis(ctx)
    assert {imp.premise: imp.conclusion for imp in basis} == expected

    for s in subsets:
        once = set(s)
        for imp in basis:
            if imp.premise <= s:
                once |= imp.conclusion
        assert frozenset(once) == ctx.closure(s)


def test_empty_premise_when_every_object_has_an_attribute():
    ctx = FormalContext.from_array([[1, 1], [1, 0]], attributes=["a", "b"])

    basis = compute_proper_premise_basis(ctx)

    assert basis[0].premise == frozenset()
    assert basis[0].conclusion_names(ctx) == ("a",)
