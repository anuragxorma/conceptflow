from __future__ import annotations

import itertools
import pandas as pd

from typing import Set, Tuple

from conceptflow.core.context import FormalContext
from conceptflow.core.lattice import ConceptLattice


def object_concept(
    context: FormalContext,
    g: str,
) -> Tuple[Set[str], Set[str]]:
    """
    Compute the object concept of a single object g.

    Returns
    -------
    tuple[set[str], set[str]]
        (extent, intent) of the object concept
    """
    intent = context.object_derivation({g})
    extent = context.attribute_derivation(intent)
    return set(extent), set(intent)


def attribute_concept(
    context: FormalContext,
    m: str,
) -> Tuple[Set[str], Set[str]]:
    """
    Compute the attribute concept of a single attribute m.

    Returns
    -------
    tuple[set[str], set[str]]
        (extent, intent) of the attribute concept
    """
    extent = context.attribute_derivation({m})
    intent = context.object_derivation(extent)
    return set(extent), set(intent)


def object_closure(
    context: FormalContext,
    objects: Set[str],
) -> Set[str]:
    """
    Compute the double-primed closure A'' of a set of objects A.
    """
    return set(context.attribute_derivation(context.object_derivation(objects)))


def attribute_closure(
    context: FormalContext,
    attributes: Set[str],
) -> Set[str]:
    """
    Compute the double-primed closure B'' of a set of attributes B.
    """
    return set(context.object_derivation(context.attribute_derivation(attributes)))


def reduce_context(
    context: FormalContext,
) -> FormalContext:
    """
    Reduce the formal context by keeping only join-irreducible objects and
    meet-irreducible attributes.

    Notes
    -----
    This is a ConceptFlow-based port of the old helper. It assumes the lattice
    has already been computed correctly by ConceptFlow.

    Join-irreducible objects are taken from concepts with exactly one lower
    cover. Meet-irreducible attributes are taken from concepts with exactly one
    upper cover.

    This helper is kept mainly for compatibility with the dim-flux port.
    """
    lattice = ConceptLattice.from_context(context, method="nextclosure")
    edges = lattice.edges()

    # Build parent/child adjacency on actual Concept objects.
    parents: dict = {c: set() for c in lattice.concepts}
    children: dict = {c: set() for c in lattice.concepts}

    for lower, upper in edges:
        parents[lower].add(upper)
        children[upper].add(lower)

    join_irreducibles: list[str] = []
    meet_irreducibles: list[str] = []

    for concept in lattice.concepts:
        # join-irreducible: exactly one lower cover and one "new" object
        if len(children[concept]) == 1:
            lower = next(iter(children[concept]))
            new_extent = set(concept.extent) - set(lower.extent)
            if len(new_extent) == 1:
                join_irreducibles.append(next(iter(new_extent)))

        # meet-irreducible: exactly one upper cover and one "new" attribute
        if len(parents[concept]) == 1:
            upper = next(iter(parents[concept]))
            new_intent = set(concept.intent) - set(upper.intent)
            if len(new_intent) == 1:
                meet_irreducibles.append(next(iter(new_intent)))

    # Remove duplicates while preserving order.
    join_irreducibles = list(dict.fromkeys(join_irreducibles))
    meet_irreducibles = list(dict.fromkeys(meet_irreducibles))

    # Already reduced
    if (
        len(join_irreducibles) == context.n_objects
        and len(meet_irreducibles) == context.n_attributes
    ):
        return context

    # Create reduced context
    df = pd.DataFrame(0, index=join_irreducibles, columns=meet_irreducibles)
    for g, m in itertools.product(join_irreducibles, meet_irreducibles):
        if context.has_incidence(g, m):
            df.loc[g, m] = 1

    return FormalContext.from_dataframe(df)