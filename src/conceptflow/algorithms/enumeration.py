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

from conceptflow.algorithms._bitset import BitsetContext, to_indices
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


def _intent_of(extent: int, columns: list[int]) -> int:
    """A' for an object bitset A: attributes whose column contains A."""
    intent = 0
    for m, column in enumerate(columns):
        if extent & column == extent:
            intent |= 1 << m
    return intent


def _to_concept(extent: int, intent: int) -> Concept:
    return Concept(extent=to_indices(extent), intent=to_indices(intent))


def enumerate_concepts_nextclosure(context: FormalContext) -> list[Concept]:
    """
    Enumerate all formal concepts using the NextClosure algorithm.

    Concepts are returned in lectic order of their intents (attribute 0 is
    the most significant). Sets are integer bitsets internally, so a closure
    costs one AND per attribute instead of NumPy calls.
    """
    bits = BitsetContext(context)
    columns = bits.columns
    n_attributes = bits.n_attributes

    def close(attributes: int) -> tuple[int, int]:
        extent = bits.extent(attributes)
        return extent, _intent_of(extent, columns)

    concepts: list[Concept] = []
    current: tuple[int, int] | None = close(0)

    while current is not None:
        extent, intent = current
        concepts.append(_to_concept(extent, intent))

        current = None
        prefix = intent

        for i in range(n_attributes - 1, -1, -1):
            bit = 1 << i

            if prefix & bit:
                prefix &= ~bit
                continue

            candidate_extent, candidate_intent = close(prefix | bit)

            # Canonical iff the closure adds no attribute below i.
            if (candidate_intent & ~prefix) & (bit - 1) == 0:
                current = (candidate_extent, candidate_intent)
                break

    return concepts


def _closebyone_bitsets(
    bits: BitsetContext,
    min_extent_size: int = 0,
) -> list[tuple[int, int]]:
    """
    Close-by-One on integer bitsets; returns ``(extent, intent)`` pairs.

    Concepts with fewer than ``min_extent_size`` objects are pruned with
    their whole subtree: extending a concept only shrinks its extent, and a
    concept's canonical generation path runs through larger extents only,
    so no concept above the threshold is lost.
    """
    columns = bits.columns
    n_attributes = bits.n_attributes

    if bits.n_objects < min_extent_size:
        return []

    found: list[tuple[int, int]] = []
    stack = [(bits.all_objects, _intent_of(bits.all_objects, columns), 0)]

    while stack:
        extent, intent, start = stack.pop()
        found.append((extent, intent))

        for j in range(start, n_attributes):
            if intent >> j & 1:
                continue

            new_extent = extent & columns[j]

            if new_extent.bit_count() < min_extent_size:
                continue

            if any(
                not intent >> m & 1 and new_extent & columns[m] == new_extent
                for m in range(j)
            ):
                continue

            new_intent = intent | (1 << j)
            for m in range(j + 1, n_attributes):
                if not new_intent >> m & 1 and new_extent & columns[m] == new_extent:
                    new_intent |= 1 << m

            stack.append((new_extent, new_intent, j + 1))

    return found


def enumerate_concepts_closebyone(context: FormalContext) -> list[Concept]:
    """
    Enumerate formal concepts using Close-by-One.

    Starting from the top concept, each concept (A, B) is extended by every
    attribute j > (the attribute that generated it) not in B: the new extent
    is A ∩ j'. The extension is canonical, and therefore new, iff no
    attribute below j that is missing from B holds for the whole new extent;
    that test runs before the rest of the new intent is computed, so most
    duplicates are rejected early. Sets are integer bitsets throughout.
    """
    found = _closebyone_bitsets(BitsetContext(context))

    concepts = [_to_concept(extent, intent) for extent, intent in found]

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
