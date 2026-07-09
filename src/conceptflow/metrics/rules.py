"""
Basic support and confidence metrics for FCA.

These metrics are standard in frequent itemset mining, association rule
mining, and FCA-style implication analysis.
"""

from __future__ import annotations

from collections.abc import Iterable

from conceptflow.core import Concept, FormalContext


def concept_support(
    concept: Concept,
    context: FormalContext,
    normalize: bool = True,
) -> float | int:
    """
    Compute the support of a formal concept.

    Support is the size of the extent.

    If normalize=True:

        support = |extent| / |G|

    If normalize=False:

        support = |extent|
    """
    count = len(concept.extent)

    if not normalize:
        return count

    if context.n_objects == 0:
        raise ValueError("Cannot compute normalized support for empty context.")

    return count / context.n_objects


def attribute_set_support(
    context: FormalContext,
    attributes: Iterable[int],
    normalize: bool = True,
) -> float | int:
    """
    Compute support of an attribute set.

    This is the number of objects having all given attributes.
    """
    extent = context.attribute_derivation(attributes)
    count = len(extent)

    if not normalize:
        return count

    if context.n_objects == 0:
        raise ValueError("Cannot compute normalized support for empty context.")

    return count / context.n_objects


def implication_confidence(
    context: FormalContext,
    premise: Iterable[int],
    conclusion: Iterable[int],
) -> float:
    """
    Compute confidence of an implication premise -> conclusion.

    confidence(X -> Y) = support(X union Y) / support(X)
    """
    premise = frozenset(premise)
    conclusion = frozenset(conclusion)

    premise_support = attribute_set_support(
        context,
        premise,
        normalize=False,
    )

    if premise_support == 0:
        raise ValueError(
            "Cannot compute confidence for implication with zero premise support."
        )

    combined_support = attribute_set_support(
        context,
        premise | conclusion,
        normalize=False,
    )

    return combined_support / premise_support