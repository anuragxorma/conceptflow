"""
Hasse diagram / cover relation utilities for concept lattices.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable

from conceptflow.algorithms.order import strict_subconcept_of
from conceptflow.core.concept import Concept

_HASSE_LARGE_LATTICE_THRESHOLD = 500


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


def compute_hasse_edges(
    concepts: Iterable[Concept],
) -> tuple[tuple[Concept, Concept], ...]:
    """
    Compute cover edges of a concept lattice.

    Returns
    -------
    tuple of tuple
        Edges as ``(lower, upper)`` pairs.
    """
    concepts = tuple(concepts)
    n = len(concepts)

    if n > _HASSE_LARGE_LATTICE_THRESHOLD:
        warnings.warn(
            f"Computing Hasse edges for {n} concepts. "
            "This may be slow for large lattices.",
            stacklevel=2,
        )

    # Precompute strict order as index sets: above[i] holds all j where
    # concepts[i] < concepts[j]. This avoids O(n) tuple conversion on every
    # is_cover() call and enables set-difference to find direct covers.
    above: list[frozenset[int]] = [
        frozenset(
            j for j in range(n)
            if i != j and strict_subconcept_of(concepts[i], concepts[j])
        )
        for i in range(n)
    ]

    edges: list[tuple[Concept, Concept]] = []

    for i in range(n):
        # Elements above i that are reachable via an intermediate step are
        # not direct covers. Their union is: ⋃_{j ∈ above[i]} above[j].
        reachable_via_intermediate = frozenset(k for j in above[i] for k in above[j])
        for j in above[i] - reachable_via_intermediate:
            edges.append((concepts[i], concepts[j]))

    return tuple(edges)