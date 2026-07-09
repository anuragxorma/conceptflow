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

    object_clarified_incidence = np.array(
        kept_object_rows,
        dtype=bool,
    )

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

    clarified_incidence = np.array(
        kept_attribute_columns,
        dtype=bool,
    ).T

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