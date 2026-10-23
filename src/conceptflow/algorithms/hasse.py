"""
Hasse diagram / cover relation utilities for concept lattices.
"""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Iterable

from conceptflow.algorithms._bitset import BitsetContext, from_indices
from conceptflow.algorithms.order import strict_subconcept_of
from conceptflow.core.concept import Concept
from conceptflow.core.context import FormalContext


def is_cover(
    lower: Concept,
    upper: Concept,
    concepts: Iterable[Concept],
) -> bool:
    """
    Return whether ``upper`` covers ``lower``.
    """
    concepts = tuple(concepts)

    if not strict_subconcept_of(lower, upper):
        return False

    for middle in concepts:
        if middle == lower or middle == upper:
            continue

        if (
            strict_subconcept_of(lower, middle)
            and strict_subconcept_of(middle, upper)
        ):
            return False

    return True


def _lower_cover_indices_from_context(
    concepts: tuple[Concept, ...],
    extents: list[int],
    context: FormalContext,
) -> list[list[int]] | None:
    """
    Lower covers via attribute extents.

    The lower neighbours of a concept ``(A, B)`` are exactly the concepts
    whose extents are maximal among ``{A ∩ m' : m not in B}``. This needs one
    bitset intersection per attribute instead of a scan over all concepts.

    Returns ``None`` if some maximal candidate extent is not among
    ``concepts`` -- i.e. ``concepts`` is not the full lattice of
    ``context`` -- so the caller can fall back to the generic algorithm.
    """
    bits = BitsetContext(context)
    index_of_extent = {extent: i for i, extent in enumerate(extents)}

    if len(index_of_extent) != len(extents):
        return None

    lower_covers: list[list[int]] = []

    for i, concept in enumerate(concepts):
        extent = extents[i]
        candidates = {
            extent & bits.columns[m]
            for m in range(bits.n_attributes)
            if m not in concept.intent
        }
        candidates.discard(extent)

        # Scanning by decreasing size, a candidate is maximal iff it is not
        # contained in an already accepted (hence maximal) candidate.
        maximal: list[int] = []
        for candidate in sorted(candidates, key=int.bit_count, reverse=True):
            if not any(candidate & accepted == candidate for accepted in maximal):
                maximal.append(candidate)

        covers = []
        for candidate in maximal:
            j = index_of_extent.get(candidate)
            if j is None:
                return None
            covers.append(j)

        lower_covers.append(covers)

    return lower_covers


def _upper_cover_indices_generic(extents: list[int]) -> list[list[int]]:
    """
    Upper covers of an arbitrary family of concepts, by extent inclusion.

    For each concept ``i``, the concepts strictly above it are scanned in
    order of increasing extent size. A candidate ``j`` is a cover iff no
    already accepted cover lies below it: any element strictly between ``i``
    and ``j`` has a smaller extent than ``j``, and above it sits a cover of
    ``i`` that was therefore accepted earlier.
    """
    sizes = [extent.bit_count() for extent in extents]
    order = sorted(range(len(extents)), key=sizes.__getitem__)
    sorted_sizes = [sizes[j] for j in order]

    upper_covers: list[list[int]] = []

    for i, extent in enumerate(extents):
        covers: list[int] = []
        start = bisect_right(sorted_sizes, sizes[i])

        for j in order[start:]:
            candidate = extents[j]
            if extent & candidate != extent:
                continue
            if any(extents[k] & candidate == extents[k] for k in covers):
                continue
            covers.append(j)

        upper_covers.append(covers)

    return upper_covers


def compute_hasse_edges(
    concepts: Iterable[Concept],
    context: FormalContext | None = None,
) -> tuple[tuple[Concept, Concept], ...]:
    """
    Compute cover edges of a concept lattice.

    Parameters
    ----------
    concepts:
        Concepts to connect. Any family of concepts of one context works;
        the result is the cover relation of the induced order. A concept
        listed twice is linked through only one of its copies.

    context:
        The context the concepts belong to. When given and ``concepts`` is
        its full concept lattice, covers are computed from attribute
        extents in ``O(|concepts| * |M|)`` set operations. Otherwise a
        generic, roughly quadratic algorithm is used.

    Returns
    -------
    tuple of tuple
        Edges as ``(lower, upper)`` pairs, ordered by the position of
        ``lower`` and then ``upper`` in ``concepts``.
    """
    concepts = tuple(concepts)
    extents = [from_indices(concept.extent) for concept in concepts]

    lower_covers = None
    if context is not None:
        lower_covers = _lower_cover_indices_from_context(concepts, extents, context)

    pairs: list[tuple[int, int]] = []

    if lower_covers is not None:
        for upper, lowers in enumerate(lower_covers):
            pairs.extend((lower, upper) for lower in lowers)
    else:
        for lower, uppers in enumerate(_upper_cover_indices_generic(extents)):
            pairs.extend((lower, upper) for upper in uppers)

    pairs.sort()

    return tuple((concepts[lower], concepts[upper]) for lower, upper in pairs)
