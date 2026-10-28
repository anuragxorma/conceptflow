"""
Minimal generators (keys) of concept intents.

A set of attributes ``K`` is a *minimal generator* (or *key*) of the concept
``(K', K'')`` if no proper subset of ``K`` selects the same objects. Every
intent has at least one key; keys are the shortest ways to describe a
concept by attributes.

Keys are closed under taking subsets, so they are found level by level:
candidates of size ``k`` are built from keys of size ``k - 1`` and kept iff
removing any single attribute selects strictly more objects.
"""

from __future__ import annotations

from conceptflow.algorithms._bitset import BitsetContext, iter_bits, to_indices
from conceptflow.core.concept import Concept
from conceptflow.core.context import FormalContext


def _keys_with_extents(bits: BitsetContext) -> dict[int, int]:
    """All keys as bitsets, mapped to their extents, found level by level."""
    columns = bits.columns
    key_extent: dict[int, int] = {0: bits.all_objects}
    level = [0]

    while level:
        next_level = []

        for key in level:
            extent = key_extent[key]

            # Attributes after the key's largest one: each set is built once.
            for m in range(key.bit_length(), bits.n_attributes):
                candidate = key | (1 << m)
                candidate_extent = extent & columns[m]

                # Every (k-1)-subset must be a key with a strictly larger extent.
                is_key = True
                for a in iter_bits(candidate):
                    subset_extent = key_extent.get(candidate & ~(1 << a))
                    if subset_extent is None or subset_extent == candidate_extent:
                        is_key = False
                        break

                if is_key:
                    key_extent[candidate] = candidate_extent
                    next_level.append(candidate)

        level = next_level

    return key_extent


def compute_minimal_generators(
    context: FormalContext,
) -> dict[Concept, tuple[frozenset[int], ...]]:
    """
    Compute the minimal generators (keys) of every concept intent.

    Returns
    -------
    dict
        Concept -> its keys as attribute-index sets, shortest first (ties in
        lexicographic order). Every concept of the lattice appears.
    """
    bits = BitsetContext(context)
    generators: dict[Concept, list[frozenset[int]]] = {}

    for key, extent in _keys_with_extents(bits).items():
        concept = Concept._from_indices(to_indices(extent), to_indices(bits.intent(extent)))
        generators.setdefault(concept, []).append(to_indices(key))

    return {
        concept: tuple(sorted(keys, key=lambda k: (len(k), sorted(k))))
        for concept, keys in generators.items()
    }
