"""
Order relations for formal concepts.
"""

from __future__ import annotations

from conceptflow.core.concept import Concept


def subconcept_of(lower: Concept, upper: Concept) -> bool:
    """
    Return whether ``lower <= upper`` in the concept lattice.

    In FCA:

        lower <= upper iff extent(lower) subseteq extent(upper)

    Equivalently, this is reverse inclusion on intents.
    """
    return lower.extent.issubset(upper.extent)


def strict_subconcept_of(lower: Concept, upper: Concept) -> bool:
    """
    Return whether ``lower < upper`` in the concept lattice.
    """
    return lower != upper and subconcept_of(lower, upper)