"""
Concept enumeration algorithms for ConceptFlow.

This module contains algorithms for enumerating formal concepts from a
formal context.

Implemented methods:

- brute-force enumeration
- NextClosure enumeration
- CloseByOne-style recursive enumeration

The brute-force algorithm is kept as a correctness baseline. NextClosure is
the first FCA-native enumeration algorithm used by ConceptFlow.
"""

from __future__ import annotations

from collections.abc import Iterable
from itertools import chain, combinations

from conceptflow.algorithms._registry import normalize_enumeration_algorithm
from conceptflow.algorithms.derivation import (
    attribute_closure,
    attribute_derivation,
)
from conceptflow.core.concept import Concept
from conceptflow.core.context import FormalContext


def powerset(items: Iterable[int]) -> Iterable[tuple[int, ...]]:
    """
    Yield all subsets of the given iterable.
    """
    items = list(items)
    return chain.from_iterable(
        combinations(items, r) for r in range(len(items) + 1)
    )


def enumerate_concepts_bruteforce(context: FormalContext) -> list[Concept]:
    """
    Enumerate all formal concepts using brute force.

    This tries every subset of attributes, closes it, and constructs the
    corresponding concept.
    """
    concepts: set[Concept] = set()

    attribute_indices = range(context.n_attributes)

    for subset in powerset(attribute_indices):
        intent = attribute_closure(context, subset)
        extent = attribute_derivation(context, intent)
        concepts.add(Concept(extent=extent, intent=intent))

    return sorted(
        concepts,
        key=lambda c: (
            len(c.extent),
            sorted(c.extent),
            sorted(c.intent),
        ),
    )


def _next_closure_candidate(
    context: FormalContext,
    current_intent: frozenset[int],
    attribute_order: list[int],
) -> frozenset[int] | None:
    """
    Compute the next closed intent after current_intent in lectic order.
    """
    index_of = {attribute: i for i, attribute in enumerate(attribute_order)}

    for i in range(len(attribute_order) - 1, -1, -1):
        attribute = attribute_order[i]

        if attribute in current_intent:
            continue

        prefix = {a for a in current_intent if index_of[a] < i}
        candidate_seed = prefix | {attribute}
        candidate_closure = attribute_closure(context, candidate_seed)

        is_valid = True

        for j in range(i):
            earlier_attribute = attribute_order[j]

            if (
                earlier_attribute in candidate_closure
            ) != (
                earlier_attribute in current_intent
            ):
                is_valid = False
                break

        if is_valid:
            return candidate_closure

    return None


def enumerate_concepts_nextclosure(context: FormalContext) -> list[Concept]:
    """
    Enumerate all formal concepts using the NextClosure algorithm.
    """
    concepts: list[Concept] = []

    attribute_order = list(range(context.n_attributes))
    current_intent: frozenset[int] | None = attribute_closure(context, [])

    while current_intent is not None:
        extent = attribute_derivation(context, current_intent)
        concepts.append(Concept(extent=extent, intent=current_intent))

        current_intent = _next_closure_candidate(
            context=context,
            current_intent=current_intent,
            attribute_order=attribute_order,
        )

    return concepts

def enumerate_concepts_closebyone(context: FormalContext) -> list[Concept]:
    """
    Enumerate formal concepts using a simple CloseByOne-style recursion.

    This implementation is intended as a clear baseline implementation of
    the CloseByOne idea: recursively generate closures and use a canonicity
    condition to avoid duplicates.
    """
    concepts: set[Concept] = set()
    n_attributes = context.n_attributes

    def is_canonical(
        current_intent: frozenset[int],
        candidate_intent: frozenset[int],
        attribute_index: int,
    ) -> bool:
        """
        Check the CloseByOne canonicity condition.

        A candidate is canonical if no earlier attribute was introduced by
        the closure unless it was already present in the current intent.
        """
        for earlier in range(attribute_index):
            if earlier in candidate_intent and earlier not in current_intent:
                return False

        return True

    def recurse(intent: frozenset[int], start_attribute: int) -> None:
        extent = attribute_derivation(context, intent)
        concepts.add(Concept(extent=extent, intent=intent))

        for attribute_index in range(start_attribute, n_attributes):
            if attribute_index in intent:
                continue

            candidate_seed = intent | {attribute_index}
            candidate_intent = attribute_closure(context, candidate_seed)

            if is_canonical(
                current_intent=intent,
                candidate_intent=candidate_intent,
                attribute_index=attribute_index,
            ):
                recurse(
                    intent=candidate_intent,
                    start_attribute=attribute_index + 1,
                )

    initial_intent = attribute_closure(context, [])
    recurse(initial_intent, 0)

    return sorted(
        concepts,
        key=lambda c: (
            len(c.extent),
            sorted(c.extent),
            sorted(c.intent),
        ),
    )

def enumerate_concepts(
    context: FormalContext,
    method: str = "nextclosure",
) -> list[Concept]:
    """
    Enumerate concepts using the selected method.
    """
    method = normalize_enumeration_algorithm(method)

    if method == "bruteforce":
        return enumerate_concepts_bruteforce(context)

    if method == "nextclosure":
        return enumerate_concepts_nextclosure(context)

    if method == "closebyone":
        return enumerate_concepts_closebyone(context)

    raise RuntimeError(
        f'Algorithm "{method}" was normalized but has no implementation.'
    )