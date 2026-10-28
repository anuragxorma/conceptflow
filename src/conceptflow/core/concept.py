"""
Formal concepts for ConceptFlow.

A formal concept is a pair (A, B), where A is an extent and B is an
intent such that A' = B and B' = A.
"""

from __future__ import annotations

import hashlib
import operator
from dataclasses import dataclass


@dataclass(frozen=True, order=True)
class Concept:
    """
    Formal concept represented by extent and intent index sets.

    Parameters
    ----------
    extent:
        Indices of objects belonging to the concept.

    intent:
        Indices of attributes belonging to the concept.
    """

    extent: frozenset[int]
    intent: frozenset[int]

    def __post_init__(self) -> None:
        # operator.index turns numpy integers into Python ints (and rejects
        # floats), so concepts built from np.flatnonzero etc. hash, compare
        # and encode as bitsets exactly like the ones ConceptFlow computes.
        object.__setattr__(
            self, "extent", frozenset(operator.index(i) for i in self.extent)
        )
        object.__setattr__(
            self, "intent", frozenset(operator.index(i) for i in self.intent)
        )

    @classmethod
    def _from_indices(cls, extent: frozenset[int], intent: frozenset[int]) -> "Concept":
        """
        Internal: a concept from frozensets of Python ints, without checking
        and copying them again (the enumeration algorithms produce millions).
        """
        concept = object.__new__(cls)
        object.__setattr__(concept, "extent", extent)
        object.__setattr__(concept, "intent", intent)
        return concept

    def signature(self) -> str:
        """
        Return a deterministic mathematical signature for this concept.

        The signature is based only on extent and intent indices, not on
        Python object identity. This makes it suitable for stable references
        in visualization, serialization, and nested exploration.
        """
        extent_part = ",".join(map(str, sorted(self.extent)))
        intent_part = ",".join(map(str, sorted(self.intent)))

        return f"extent:{extent_part}|intent:{intent_part}"

    def stable_id(self, prefix: str = "C", length: int = 12) -> str:
        """
        Return a compact identifier derived from the concept's intent.

        Within one lattice the intent determines the concept, so ids are
        unique there. The id does not depend on the objects' order or on
        which objects the data has: the same concept gets the same id after
        rows are reordered or resampled (e.g. across cross-validation folds),
        as long as the attributes (columns) stay the same.
        """
        intent_part = ",".join(map(str, sorted(self.intent)))
        digest = hashlib.sha1(f"intent:{intent_part}".encode("utf-8")).hexdigest()[:length]

        return f"{prefix}_{digest}"
