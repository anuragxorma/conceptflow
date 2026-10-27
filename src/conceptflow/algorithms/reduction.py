"""
Context clarification utilities.

Clarification removes duplicate object rows and duplicate attribute columns
from a formal context.

This is useful before visualization because duplicate rows/columns do not
change the conceptual structure but can make diagrams unnecessarily cluttered.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from conceptflow.core import FormalContext


@dataclass(frozen=True)
class ClarificationResult:
    """
    Result of context clarification.

    Parameters
    ----------
    context:
        The clarified formal context.

    object_groups:
        Mapping from kept object name to all original object names represented
        by it.

    attribute_groups:
        Mapping from kept attribute name to all original attribute names
        represented by it.
    """

    context: FormalContext
    object_groups: dict[str, tuple[str, ...]]
    attribute_groups: dict[str, tuple[str, ...]]


def clarify_context(context: FormalContext) -> ClarificationResult:
    """
    Remove duplicate object rows and duplicate attribute columns.

    Returns
    -------
    ClarificationResult
        Clarified context plus metadata describing merged objects and
        attributes.
    """
    object_groups_by_row: dict[tuple[bool, ...], list[str]] = {}

    for object_name, row in zip(context.objects, context.incidence):
        key = tuple(bool(value) for value in row)
        object_groups_by_row.setdefault(key, []).append(object_name)

    kept_object_rows = list(object_groups_by_row.keys())
    kept_objects = tuple(
        names[0]
        for names in object_groups_by_row.values()
    )
    object_groups = {
        names[0]: tuple(names)
        for names in object_groups_by_row.values()
    }

    if kept_object_rows:
        object_clarified_incidence = np.array(kept_object_rows, dtype=bool)
    else:
        object_clarified_incidence = np.empty((0, context.n_attributes), dtype=bool)

    attribute_groups_by_column: dict[tuple[bool, ...], list[str]] = {}

    for attribute_index, attribute_name in enumerate(context.attributes):
        column = tuple(
            bool(value)
            for value in object_clarified_incidence[:, attribute_index]
        )
        attribute_groups_by_column.setdefault(column, []).append(attribute_name)

    kept_attribute_columns = list(attribute_groups_by_column.keys())
    kept_attributes = tuple(
        names[0]
        for names in attribute_groups_by_column.values()
    )
    attribute_groups = {
        names[0]: tuple(names)
        for names in attribute_groups_by_column.values()
    }

    n_kept_objects = object_clarified_incidence.shape[0]
    if kept_attribute_columns:
        clarified_incidence = np.array(kept_attribute_columns, dtype=bool).T
    else:
        clarified_incidence = np.empty((n_kept_objects, 0), dtype=bool)

    clarified_context = FormalContext(
        objects=kept_objects,
        attributes=kept_attributes,
        incidence=clarified_incidence,
    )

    return ClarificationResult(
        context=clarified_context,
        object_groups=object_groups,
        attribute_groups=attribute_groups,
    )


def clarified_context(context: FormalContext) -> FormalContext:
    """
    Return only the clarified context.

    This is a convenience wrapper around clarify_context.
    """
    return clarify_context(context).context

@dataclass(frozen=True)
class ReductionResult:
    """
    Result of context reduction.

    Parameters
    ----------
    context:
        The reduced context. Its concept lattice is isomorphic to the
        original one.

    object_groups, attribute_groups:
        From clarification: kept name -> all original names it stands for.

    removed_objects, removed_attributes:
        Reducible objects/attributes of the clarified context, which were
        dropped.
    """

    context: FormalContext
    object_groups: dict[str, tuple[str, ...]]
    attribute_groups: dict[str, tuple[str, ...]]
    removed_objects: tuple[str, ...]
    removed_attributes: tuple[str, ...]


def _reducible(rows: np.ndarray) -> list[int]:
    """
    Indices of reducible rows of a clarified incidence matrix.

    A row ``r`` is reducible iff it equals the intersection of all rows that
    strictly contain it (the intersection of no rows being the full row).
    """
    reducible = []
    rows = np.asarray(rows, dtype=bool)
    sizes = rows.sum(axis=1)

    for i, row in enumerate(rows):
        # Strict supersets: contain the row (no attribute of it missing) and
        # are larger (the context is clarified, so no other row equals it).
        superset = ~(rows[:, row] == False).any(axis=1) & (sizes > sizes[i])  # noqa: E712
        meet = rows[superset].all(axis=0) if superset.any() else np.ones_like(row)
        if np.array_equal(meet, row):
            reducible.append(i)

    return reducible


def reduce_context(context: FormalContext) -> ReductionResult:
    """
    Clarify, then remove reducible objects and attributes.

    An object is reducible if its object concept is the join of other object
    concepts (its intent is the intersection of strictly larger object
    intents); an attribute is reducible if its attribute concept is a meet
    of other attribute concepts. Removing them keeps the concept lattice up
    to isomorphism, giving the smallest context with that lattice.
    """
    clarified = clarify_context(context)
    ctx = clarified.context

    drop_objects = set(_reducible(ctx.incidence))
    drop_attributes = set(_reducible(ctx.incidence.T))

    reduced = ctx.subcontext(
        objects=[i for i in range(ctx.n_objects) if i not in drop_objects],
        attributes=[j for j in range(ctx.n_attributes) if j not in drop_attributes],
    )

    return ReductionResult(
        context=reduced,
        object_groups=clarified.object_groups,
        attribute_groups=clarified.attribute_groups,
        removed_objects=tuple(ctx.objects[i] for i in sorted(drop_objects)),
        removed_attributes=tuple(ctx.attributes[j] for j in sorted(drop_attributes)),
    )


def reduced_context(context: FormalContext) -> FormalContext:
    """Return only the reduced context; see :func:`reduce_context`."""
    return reduce_context(context).context
