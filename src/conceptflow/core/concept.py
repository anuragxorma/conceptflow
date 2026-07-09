"""
Formal concepts for ConceptFlow.

A formal concept is a pair (A, B), where A is an extent and B is an
intent such that A' = B and B' = A.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib


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
        object.__setattr__(self, "extent", frozenset(self.extent))
        object.__setattr__(self, "intent", frozenset(self.intent))

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
        Return a compact stable identifier derived from the concept signature.
        """
        digest = hashlib.sha1(
            self.signature().encode("utf-8")
        ).hexdigest()[:length]

        return f"{prefix}_{digest}"