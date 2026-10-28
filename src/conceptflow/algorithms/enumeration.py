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

import numbers
import os
import warnings
from collections.abc import Iterable
from itertools import chain, combinations

from conceptflow.algorithms._bitset import BitsetContext, iter_bits, to_indices
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


# Warnings are attributed to the first stack frame outside ConceptFlow (and
# scikit-learn, so warnings raised inside a Pipeline point at user code).
_PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + os.sep


def _internal_prefixes() -> tuple[str, ...]:
    prefixes = [_PACKAGE_DIR]
    try:
        import sklearn

        prefixes.append(os.path.dirname(os.path.abspath(sklearn.__file__)) + os.sep)
    except ImportError:  # pragma: no cover - sklearn is a dependency
        pass
    return tuple(prefixes)


_INTERNAL_PREFIXES = _internal_prefixes()

# Enumerations warn once when they pass this many concepts (see _SizeGuard).
LARGE_LATTICE_WARNING_SIZE = 50_000


class LargeLatticeWarning(UserWarning):
    """Emitted while enumerating a lattice that is becoming very large."""


class ConceptLimitExceeded(ValueError):
    """Raised when an enumeration exceeds an explicit ``max_concepts`` limit."""


class _SizeGuard:
    """
    Watches the number of concepts produced by an enumeration.

    Nothing changes by default: past ``LARGE_LATTICE_WARNING_SIZE`` concepts a
    single warning names the opt-in options, and the enumeration continues.
    Only an explicit ``max_concepts`` stops it.
    """

    def __init__(self, max_concepts: int | None = None) -> None:
        if max_concepts is not None and (
            isinstance(max_concepts, bool)
            or not isinstance(max_concepts, numbers.Integral)
            or max_concepts < 1
        ):
            raise ValueError("max_concepts must be None or a positive integer.")
        self.max_concepts = None if max_concepts is None else int(max_concepts)
        self.count = 0
        self.warned = False

    def tick(self) -> None:
        self.count += 1

        if self.max_concepts is not None and self.count > self.max_concepts:
            raise ConceptLimitExceeded(
                f"The lattice has more than max_concepts={self.max_concepts} "
                "concepts. Use min_support=... to compute only frequent "
                "concepts, or raise max_concepts."
            )

        # With an explicit max_concepts the user has already decided.
        if (
            not self.warned
            and self.max_concepts is None
            and self.count > LARGE_LATTICE_WARNING_SIZE
        ):
            self.warned = True
            warnings.warn(
                f"The concept lattice has more than {LARGE_LATTICE_WARNING_SIZE:,} "
                "concepts and is still growing. Computation continues. To limit "
                "it, use min_support=... (only concepts with at least that many "
                "objects) or max_concepts=... (stop with an error past a limit).",
                LargeLatticeWarning,
                skip_file_prefixes=_INTERNAL_PREFIXES,
            )


def _intent_of(extent: int, columns: list[int]) -> int:
    """A' for an object bitset A: attributes whose column contains A."""
    intent = 0
    for m, column in enumerate(columns):
        if extent & column == extent:
            intent |= 1 << m
    return intent


def _to_concept(extent: int, intent: int) -> Concept:
    return Concept._from_indices(to_indices(extent), to_indices(intent))


def enumerate_concepts_nextclosure(
    context: FormalContext,
    *,
    _guard: _SizeGuard | None = None,
) -> list[Concept]:
    """
    Enumerate all formal concepts using the NextClosure algorithm.

    Concepts are returned in lectic order of their intents (attribute 0 is
    the most significant). Sets are integer bitsets internally, so a closure
    costs one AND per attribute instead of NumPy calls.
    """
    bits = BitsetContext(context)
    columns = bits.columns
    n_attributes = bits.n_attributes
    guard = _guard or _SizeGuard()

    def close(attributes: int) -> tuple[int, int]:
        extent = bits.extent(attributes)
        return extent, _intent_of(extent, columns)

    concepts: list[Concept] = []
    current: tuple[int, int] | None = close(0)

    while current is not None:
        extent, intent = current
        guard.tick()
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
    guard: _SizeGuard | None = None,
) -> list[tuple[int, int]]:
    """
    Close-by-One on integer bitsets; returns ``(extent, intent)`` pairs.

    Uses FCbO's pruning (Krajča, Outrata and Vychodil, "Advances in
    algorithms based on CbO", CLA 2010): when extending a concept by ``j``
    fails the canonicity test, the failed intent is remembered and passed
    to the children; a child skips ``j`` without computing anything if the
    remembered intent would fail again (it has an attribute below ``j`` that
    the child's intent lacks).

    Concepts with fewer than ``min_extent_size`` objects are pruned with
    their whole subtree: extending a concept only shrinks its extent, and a
    concept's canonical generation path runs through larger extents only,
    so no concept above the threshold is lost.
    """
    columns = bits.columns
    rows = bits.rows
    n_attributes = bits.n_attributes
    all_attributes = bits.all_attributes

    def intent_of(extent: int, known: int) -> int:
        # Intersect the rows of the extent's objects when there are few of
        # them; otherwise test each remaining attribute's column.
        rest = all_attributes & ~known
        if extent.bit_count() < 2 * rest.bit_count():
            result = all_attributes
            for g in iter_bits(extent):
                result &= rows[g]
            return result | known
        result = known
        for m in iter_bits(rest):
            if not extent & ~columns[m]:
                result |= 1 << m
        return result

    if bits.n_objects < min_extent_size:
        # Not even the top concept is frequent; it is still returned, so an
        # iceberg lattice always has its top.
        if guard is not None:
            guard.tick()
        return [(bits.all_objects, _intent_of(bits.all_objects, columns))]

    found: list[tuple[int, int]] = []
    # Frames: extent, intent, first attribute to try, failed intents per attribute.
    stack = [(bits.all_objects, _intent_of(bits.all_objects, columns), 0, [0] * n_attributes)]

    while stack:
        extent, intent, start, failed = stack.pop()
        if guard is not None:
            guard.tick()
        found.append((extent, intent))
        if intent == all_attributes:
            continue

        failed = list(failed)
        children = []
        for j in range(start, n_attributes):
            bit = 1 << j
            if intent & bit:
                continue
            below = bit - 1
            if failed[j] & below & ~intent:
                continue  # fails canonicity again (FCbO)

            new_extent = extent & columns[j]
            if new_extent.bit_count() < min_extent_size:
                continue

            new_intent = intent_of(new_extent, intent | bit)
            if new_intent & below & ~intent:
                failed[j] = new_intent  # not canonical: generated elsewhere
            else:
                children.append((new_extent, new_intent, j + 1))

        for new_extent, new_intent, next_start in reversed(children):
            stack.append((new_extent, new_intent, next_start, failed))

    return found


def enumerate_concepts_closebyone(
    context: FormalContext,
    *,
    _guard: _SizeGuard | None = None,
) -> list[Concept]:
    """
    Enumerate formal concepts using Close-by-One.

    Starting from the top concept, each concept (A, B) is extended by every
    attribute j > (the attribute that generated it) not in B: the new extent
    is A ∩ j'. The extension is canonical, and therefore new, iff no
    attribute below j that is missing from B holds for the whole new extent;
    that test runs before the rest of the new intent is computed, so most
    duplicates are rejected early. Sets are integer bitsets throughout.
    """
    found = _closebyone_bitsets(BitsetContext(context), guard=_guard or _SizeGuard())

    concepts = [_to_concept(extent, intent) for extent, intent in found]

    return sorted(concepts, key=_size_order)


def _size_order(concept: Concept):
    """Order used by bruteforce and Close-by-One: by extent size, then content."""
    return (len(concept.extent), sorted(concept.extent), sorted(concept.intent))


def _lectic_order(n_attributes: int):
    """NextClosure's order: intents compared as bit strings, attribute 0 first."""
    def key(concept: Concept) -> int:
        return sum(1 << (n_attributes - 1 - m) for m in concept.intent)

    return key

def _resolve_min_support(min_support: int | float | None, n_objects: int) -> int:
    """Turn an absolute (int) or relative (float) support into an object count."""
    if min_support is None:
        return 0

    from conceptflow.algorithms.implications import _resolve_min_support as resolve

    return resolve(min_support, n_objects)


def enumerate_concepts(
    context: FormalContext,
    method: str = "nextclosure",
    min_support: int | float | None = None,
    max_concepts: int | None = None,
) -> list[Concept]:
    """
    Enumerate concepts using the selected method.

    Parameters
    ----------
    context:
        The formal context.

    method:
        "nextclosure" (lectic order), "closebyone" or "bruteforce" (both
        ordered by extent size).

    min_support:
        Only enumerate concepts whose extent has at least this many objects
        (an int) or this fraction of all objects (a float): the "iceberg"
        of the lattice. These concepts form an up-set, so the top concept is
        always included.

    max_concepts:
        Stop with :class:`ConceptLimitExceeded` once more than this many
        concepts were found. By default there is no limit; enumerations past
        50,000 concepts emit a :class:`LargeLatticeWarning` and continue.
    """
    method = normalize_enumeration_algorithm(method)
    guard = _SizeGuard(max_concepts)
    threshold = _resolve_min_support(min_support, context.n_objects)

    if threshold > 0:
        # Pruned Close-by-One computes exactly the frequent concepts; they are
        # then put in the order the requested method would have produced.
        found = _closebyone_bitsets(BitsetContext(context), threshold, guard)
        concepts = [_to_concept(extent, intent) for extent, intent in found]
        if method == "nextclosure":
            return sorted(concepts, key=_lectic_order(context.n_attributes))
        return sorted(concepts, key=_size_order)

    if method == "bruteforce":
        # Bruteforce has no incremental output; count once it is done.
        concepts = enumerate_concepts_bruteforce(context)
        for _ in concepts:
            guard.tick()
        return concepts

    if method == "nextclosure":
        return enumerate_concepts_nextclosure(context, _guard=guard)

    if method == "closebyone":
        return enumerate_concepts_closebyone(context, _guard=guard)

    raise RuntimeError(
        f'Algorithm "{method}" was normalized but has no implementation.'
    )
