import numpy as np
import pytest

from conceptflow import FormalContext
from conceptflow.algorithms import Implication, compute_canonical_basis
from conceptflow.algorithms.enumeration import enumerate_concepts_bruteforce


def test_boolean_lattice_context_has_no_implications():
    # Every subset of {a, b, c} is already closed (2^3 = 8 concepts), so
    # there is nothing left to force -- the canonical basis is empty.
    ctx = FormalContext.from_array(
        np.array([
            [1, 1, 0],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 1],
        ]),
        objects=["g1", "g2", "g3", "g4"],
        attributes=["a", "b", "c"],
    )

    assert compute_canonical_basis(ctx) == []


def test_two_of_three_implies_the_third():
    # Only single attributes or all three ever co-occur, so any two
    # attributes together force the third.
    ctx = FormalContext.from_array(
        np.array([
            [1, 1, 1],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ]),
        objects=["g1", "g2", "g3", "g4"],
        attributes=["a", "b", "c"],
    )

    basis = compute_canonical_basis(ctx)

    rules = {
        (imp.premise_names(ctx), imp.conclusion_names(ctx))
        for imp in basis
    }

    assert rules == {
        (("a", "b"), ("c",)),
        (("a", "c"), ("b",)),
        (("b", "c"), ("a",)),
    }


def test_basis_is_sound_and_complete_against_bruteforce_concepts():
    """
    A basis is sound and complete iff the set of attribute sets closed under
    it (context closure plus repeatedly applying every implication) is
    exactly the set of real concept intents -- no more, no less. Verify this
    directly against independently-enumerated concepts (bruteforce) on a
    context with a nontrivial basis, rather than trusting the basis
    computation's own closure operator.
    """
    rng = np.random.default_rng(0)
    data = rng.integers(0, 2, size=(8, 5))
    ctx = FormalContext.from_array(data)

    concepts = enumerate_concepts_bruteforce(ctx)
    real_intents = {c.intent for c in concepts}

    basis = compute_canonical_basis(ctx)

    def close_under_basis(attrs: frozenset[int]) -> frozenset[int]:
        changed = True
        while changed:
            changed = False
            for imp in basis:
                if imp.premise <= attrs and not (imp.conclusion <= attrs):
                    attrs = attrs | imp.conclusion
                    changed = True
        return attrs

    # Every real intent must be a fixpoint of the basis closure (soundness:
    # the basis must not forbid anything that is actually possible).
    for intent in real_intents:
        assert close_under_basis(intent) == intent

    # Every subset, once closed under the basis, must land on a real
    # intent (completeness: the basis must forbid everything that both the
    # bruteforce enumeration and the basis agree is impossible).
    all_attribute_sets = [
        frozenset(s) for r in range(ctx.n_attributes + 1)
        for s in __import__("itertools").combinations(range(ctx.n_attributes), r)
    ]
    for attrs in all_attribute_sets:
        assert close_under_basis(attrs) in real_intents


def test_implication_premise_and_conclusion_are_disjoint():
    ctx = FormalContext.from_array(
        np.array([
            [1, 1, 1],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ]),
        attributes=["a", "b", "c"],
    )

    for imp in compute_canonical_basis(ctx):
        assert imp.premise.isdisjoint(imp.conclusion)


def test_implication_is_frozen_and_deduplicates_input():
    imp = Implication(premise=[0, 0, 1], conclusion=[2])
    assert imp.premise == frozenset({0, 1})
    with pytest.raises(Exception):
        imp.premise = frozenset()  # frozen dataclass


# ---------------------------------------------------------------------------
# Fast (LinCbO, NextClosure) bases vs. the definition-based baseline
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("algorithm", ["lincbo", "nextclosure"])
@pytest.mark.parametrize("seed", range(40))
def test_fast_basis_matches_bruteforce_on_random_contexts(seed, algorithm):
    from conceptflow.algorithms import compute_canonical_basis_bruteforce

    rng = np.random.default_rng(seed)
    n_objects = int(rng.integers(1, 12))
    n_attributes = int(rng.integers(0, 9))
    data = rng.random((n_objects, n_attributes)) < rng.uniform(0.1, 0.9)
    ctx = FormalContext.from_array(data)

    assert compute_canonical_basis(ctx, algorithm=algorithm) == (
        compute_canonical_basis_bruteforce(ctx)
    )


@pytest.mark.parametrize("seed", range(10))
def test_lincbo_matches_nextclosure_on_wide_contexts(seed):
    rng = np.random.default_rng(500 + seed)
    ctx = FormalContext.from_array(rng.random((int(rng.integers(5, 30)), 75)) < 0.12)

    assert compute_canonical_basis(ctx) == compute_canonical_basis(ctx, algorithm="nextclosure")


def test_lincbo_with_empty_premise_and_duplicate_columns():
    ctx = FormalContext.from_array([[1, 1, 1, 0], [1, 1, 0, 1], [1, 1, 0, 0]])

    assert compute_canonical_basis(ctx) == compute_canonical_basis(ctx, algorithm="nextclosure")


def test_unknown_basis_algorithm():
    with pytest.raises(ValueError, match="Unknown canonical basis algorithm"):
        compute_canonical_basis(FormalContext.from_array([[1]]), algorithm="linclosure")


def test_basis_of_empty_premise():
    # Every object has "a", so {} -> {a} is the first pseudo-intent.
    ctx = FormalContext.from_array(
        np.array([[1, 1, 0], [1, 0, 1]]),
        attributes=["a", "b", "c"],
    )

    basis = compute_canonical_basis(ctx)

    assert basis[0].premise == frozenset()
    assert basis[0].conclusion_names(ctx) == ("a",)


def test_basis_scales_beyond_bruteforce():
    # 2^28 attribute subsets would never finish with the bruteforce baseline;
    # NextClosure only visits intents and pseudo-intents.
    rng = np.random.default_rng(7)
    ctx = FormalContext.from_array(rng.random((150, 28)) < 0.2)

    basis = compute_canonical_basis(ctx)
    assert basis

    for imp in basis:
        # Each implication is P -> P'' \ P for a non-closed premise P.
        assert imp.conclusion
        assert imp.premise | imp.conclusion == ctx.closure(imp.premise)

    def close_under_basis(attrs):
        attrs = frozenset(attrs)
        changed = True
        while changed:
            changed = False
            for imp in basis:
                if imp.premise <= attrs and not imp.conclusion <= attrs:
                    attrs |= imp.conclusion
                    changed = True
        return attrs

    # Sound and complete: closing under the basis is the context closure.
    for _ in range(300):
        attrs = np.flatnonzero(rng.random(28) < 0.15).tolist()
        assert close_under_basis(attrs) == ctx.closure(attrs)


# ---------------------------------------------------------------------------
# Luxenburger basis (near-implications)
# ---------------------------------------------------------------------------


def make_rule_context():
    # a: g1..g5, b: g1..g4, c: g1..g2
    return FormalContext.from_array(
        np.array([
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 0],
            [1, 1, 0],
            [1, 0, 0],
        ]),
        objects=["g1", "g2", "g3", "g4", "g5"],
        attributes=["a", "b", "c"],
    )


def test_luxenburger_basis_rules_and_exceptions():
    from conceptflow.algorithms import compute_luxenburger_basis

    ctx = make_rule_context()
    rules = compute_luxenburger_basis(ctx)

    as_names = {
        (r.premise_names(ctx), r.conclusion_names(ctx)): r for r in rules
    }

    assert set(as_names) == {(("a",), ("b",)), (("a", "b"), ("c",))}

    ab = as_names[(("a",), ("b",))]
    assert (ab.support, ab.premise_support) == (4, 5)
    assert ab.confidence == pytest.approx(0.8)
    assert ab.exception_names(ctx) == ("g5",)

    abc = as_names[(("a", "b"), ("c",))]
    assert abc.confidence == pytest.approx(0.5)
    assert abc.exception_names(ctx) == ("g3", "g4")

    # Sorted by decreasing confidence.
    assert rules[0] is ab


def test_luxenburger_basis_thresholds():
    from conceptflow.algorithms import compute_luxenburger_basis

    ctx = make_rule_context()

    assert len(compute_luxenburger_basis(ctx, min_confidence=0.6)) == 1
    assert len(compute_luxenburger_basis(ctx, min_support=3)) == 1
    # 0.5 of 5 objects -> at least 3 (rounded up).
    assert len(compute_luxenburger_basis(ctx, min_support=0.5)) == 1
    assert len(compute_luxenburger_basis(ctx, min_support=5)) == 0

    with pytest.raises(ValueError):
        compute_luxenburger_basis(ctx, min_confidence=1.5)
    with pytest.raises(ValueError):
        compute_luxenburger_basis(ctx, min_support=1.5)


def test_luxenburger_rules_agree_with_metrics_on_random_context():
    from conceptflow.algorithms import compute_luxenburger_basis
    from conceptflow.metrics import implication_confidence, implication_exceptions

    rng = np.random.default_rng(11)
    ctx = FormalContext.from_array(rng.random((30, 8)) < 0.4)

    rules = compute_luxenburger_basis(ctx)
    assert rules

    for rule in rules:
        assert rule.premise.isdisjoint(rule.conclusion)
        assert 0 < rule.confidence < 1
        assert rule.confidence == pytest.approx(
            implication_confidence(ctx, rule.premise, rule.conclusion)
        )
        assert rule.exceptions == implication_exceptions(
            ctx, rule.premise, rule.conclusion
        )


@pytest.mark.parametrize("seed", range(25))
def test_matrix_closure_path_matches_bruteforce(seed, monkeypatch):
    # Contexts with more than 64 attributes use a different closure layout;
    # force it on small contexts so the definition-based baseline can check it.
    from conceptflow.algorithms import compute_canonical_basis_bruteforce
    from conceptflow.algorithms.implications import _ImplicationStore

    monkeypatch.setattr(_ImplicationStore, "_WORD", 0)
    rng = np.random.default_rng(1000 + seed)
    data = rng.random((int(rng.integers(1, 12)), int(rng.integers(1, 9)))) < rng.uniform(0.1, 0.9)
    ctx = FormalContext.from_array(data)

    assert compute_canonical_basis(ctx, algorithm="nextclosure") == (
        compute_canonical_basis_bruteforce(ctx)
    )


def test_basis_of_context_wider_than_64_attributes():
    rng = np.random.default_rng(5)
    ctx = FormalContext.from_array(rng.random((25, 90)) < 0.08)

    basis = compute_canonical_basis(ctx)
    assert basis

    def close_under_basis(attrs):
        attrs = frozenset(attrs)
        changed = True
        while changed:
            changed = False
            for imp in basis:
                if imp.premise <= attrs and not imp.conclusion <= attrs:
                    attrs |= imp.conclusion
                    changed = True
        return attrs

    for imp in basis:
        assert imp.premise | imp.conclusion == ctx.closure(imp.premise)
    for _ in range(200):
        attrs = np.flatnonzero(rng.random(90) < 0.05).tolist()
        assert close_under_basis(attrs) == ctx.closure(attrs)
