"""
Derivation and closure operators for Formal Concept Analysis.

This module contains the basic FCA derivation operations.

For a formal context (G, M, I):

- A' is the set of attributes common to all objects in A.
- B' is the set of objects having all attributes in B.
- B'' is the closure of B.

The functions in this module operate on FormalContext objects but are kept
outside the FormalContext class so that algorithms remain reusable and easier
to optimize later.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from conceptflow.core.context import FormalContext


def object_derivation(
    context: FormalContext,
    object_indices: Iterable[int],
) -> frozenset[int]:
    """
    Compute A' for a set of object indices A.
    """
    object_indices = frozenset(object_indices)

    if not object_indices:
        return frozenset(range(context.n_attributes))

    mask = context.incidence[list(object_indices), :].all(axis=0)
    return frozenset(np.flatnonzero(mask).tolist())


def attribute_derivation(
    context: FormalContext,
    attribute_indices: Iterable[int],
) -> frozenset[int]:
    """
    Compute B' for a set of attribute indices B.
    """
    attribute_indices = frozenset(attribute_indices)

    if not attribute_indices:
        return frozenset(range(context.n_objects))

    mask = context.incidence[:, list(attribute_indices)].all(axis=1)
    return frozenset(np.flatnonzero(mask).tolist())


def attribute_closure(
    context: FormalContext,
    attribute_indices: Iterable[int],
) -> frozenset[int]:
    """
    Compute B'' for a set of attribute indices B.
    """
    extent = attribute_derivation(context, attribute_indices)
    return object_derivation(context, extent)