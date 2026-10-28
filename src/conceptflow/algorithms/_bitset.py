"""
Integer-bitset view of a formal context.

Attribute and object sets are encoded as Python ints (bit ``j`` set iff
index ``j`` is in the set). Intersections, unions and subset tests then
become single machine-level operations on arbitrary-precision ints, which
is far faster than ``frozenset`` arithmetic in the inner loops of
NextClosure-style algorithms.

This module is internal: public functions keep returning ``frozenset[int]``
and convert at the boundary with ``to_indices`` / ``from_indices``.
"""

from __future__ import annotations

import operator
from collections.abc import Iterable

import numpy as np

from conceptflow.core.context import FormalContext


def from_indices(indices: Iterable[int]) -> int:
    """Encode an iterable of indices as a bitset."""
    if isinstance(indices, (frozenset, set)) and len(indices) > 32:
        # Large sets: build the bytes with NumPy instead of one shift per index.
        positions = np.fromiter(indices, dtype=np.int64, count=len(indices))
        if positions.min() < 0:
            raise ValueError("Bitset indices must be non-negative.")
        bools = np.zeros(int(positions.max()) + 1, dtype=bool)
        bools[positions] = True
        return int.from_bytes(np.packbits(bools, bitorder="little").tobytes(), "little")
    mask = 0
    for index in indices:
        # operator.index: a numpy integer here would turn the mask into a
        # fixed-width numpy integer and silently overflow past 63 bits.
        mask |= 1 << operator.index(index)
    return mask


def to_indices(mask: int) -> frozenset[int]:
    """Decode a bitset into a frozenset of indices."""
    if mask.bit_count() > 32:
        # Large sets (e.g. extents of thousands of objects): unpack the bytes
        # with NumPy instead of looping over the bits in Python.
        raw = np.frombuffer(mask.to_bytes((mask.bit_length() + 7) // 8, "little"), np.uint8)
        return frozenset(np.flatnonzero(np.unpackbits(raw, bitorder="little")).tolist())
    indices = []
    while mask:
        low = mask & -mask
        indices.append(low.bit_length() - 1)
        mask ^= low
    return frozenset(indices)


def iter_bits(mask: int):
    """Yield the indices of the set bits of ``mask``, lowest first."""
    while mask:
        low = mask & -mask
        yield low.bit_length() - 1
        mask ^= low


class BitsetContext:
    """
    Row- and column-bitset encoding of a formal context.

    Attributes
    ----------
    rows:
        ``rows[g]`` is the attribute set ``g'`` of object ``g``.

    columns:
        ``columns[m]`` is the object set ``m'`` of attribute ``m``.
    """

    def __init__(self, context: FormalContext) -> None:
        incidence = np.asarray(context.incidence, dtype=bool)

        self.n_objects = context.n_objects
        self.n_attributes = context.n_attributes
        self.all_objects = (1 << self.n_objects) - 1
        self.all_attributes = (1 << self.n_attributes) - 1
        self.rows = [from_indices(np.flatnonzero(row).tolist()) for row in incidence]
        self.columns = [
            from_indices(np.flatnonzero(column).tolist()) for column in incidence.T
        ]

    def extent(self, attributes: int) -> int:
        """B' for an attribute bitset B."""
        objects = self.all_objects
        for m in iter_bits(attributes):
            objects &= self.columns[m]
        return objects

    def intent(self, objects: int) -> int:
        """A' for an object bitset A."""
        attributes = self.all_attributes
        for g in iter_bits(objects):
            attributes &= self.rows[g]
        return attributes

    def closure(self, attributes: int) -> int:
        """B'' for an attribute bitset B."""
        return self.intent(self.extent(attributes))
