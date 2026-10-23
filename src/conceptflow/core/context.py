"""
Formal contexts for ConceptFlow.

This module defines the core binary formal context object used throughout
ConceptFlow.

A formal context is a triple (G, M, I), where:

- G is a set of objects,
- M is a set of attributes,
- I is a binary incidence relation between objects and attributes.

This class is intentionally not a scikit-learn estimator. It is a core
mathematical data structure. Sklearn-compatible wrappers should live in
modules such as conceptflow.preprocessing or conceptflow.cluster.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


def _is_index(value) -> bool:
    """Whether ``value`` is an integer index (Python or NumPy), not a name."""
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _cross_table(data) -> np.ndarray:
    """
    Convert incidence data to a boolean cross table.

    Empty cells (None, NaN, ``pd.NA``) mean "no cross", as in a printed cross
    table. They must not go through ``bool()`` directly: ``bool(NaN)`` is
    True. The marker ``"?"`` means "unknown", which a formal context cannot
    express, so it is rejected in favour of IncompleteContext.
    """
    raw = np.asarray(data)

    if raw.dtype == bool:
        # Copy, so later changes to the caller's array cannot alter a frozen
        # context.
        return raw.copy()

    if raw.dtype.kind in "US":
        has_unknown = bool((raw == ("?" if raw.dtype.kind == "U" else b"?")).any())
    elif raw.dtype == object:
        has_unknown = any(isinstance(v, str) and v == "?" for v in raw.flat)
    else:
        has_unknown = False

    if has_unknown:
        raise ValueError(
            'Incidence contains the unknown marker "?". A FormalContext has '
            "only crosses and non-crosses; use IncompleteContext for unknown "
            "entries."
        )

    empty = pd.isna(raw)
    if empty.any():
        raw = np.where(empty, False, raw)

    return raw.astype(bool)


@dataclass(frozen=True)
class FormalContext:
    """
    Binary formal context.

    Parameters
    ----------
    objects:
        Object names.

    attributes:
        Attribute names.

    incidence:
        Binary incidence matrix of shape
        ``(n_objects, n_attributes)``.

    Notes
    -----
    The incidence matrix is stored internally as a boolean NumPy array.
    """

    objects: tuple[str, ...]
    attributes: tuple[str, ...]
    incidence: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "objects", tuple(map(str, self.objects)))
        object.__setattr__(self, "attributes", tuple(map(str, self.attributes)))

        if len(set(self.objects)) != len(self.objects):
            raise ValueError("Object names must be unique.")

        if len(set(self.attributes)) != len(self.attributes):
            raise ValueError("Attribute names must be unique.")

        incidence = _cross_table(self.incidence)

        expected_shape = (len(self.objects), len(self.attributes))
        if incidence.shape != expected_shape:
            raise ValueError(
                "Incidence matrix has shape "
                f"{incidence.shape}, but expected {expected_shape}."
            )

        object.__setattr__(self, "incidence", incidence)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FormalContext):
            return NotImplemented
        return (
            self.objects == other.objects
            and self.attributes == other.attributes
            and np.array_equal(self.incidence, other.incidence)
        )

    def __hash__(self) -> int:
        return hash((self.objects, self.attributes, self.incidence.tobytes()))

    @property
    def n_objects(self) -> int:
        """Number of objects."""
        return len(self.objects)

    @property
    def n_attributes(self) -> int:
        """Number of attributes."""
        return len(self.attributes)

    @classmethod
    def from_dataframe(cls, data: pd.DataFrame) -> "FormalContext":
        """
        Create a formal context from a pandas DataFrame.

        The DataFrame must represent a binary context. Values are converted
        to boolean; empty cells (None, NaN, ``pd.NA``) mean "no cross".
        """
        return cls(
            objects=tuple(map(str, data.index)),
            attributes=tuple(map(str, data.columns)),
            incidence=data.to_numpy(),
        )

    @classmethod
    def from_array(
        cls,
        data: np.ndarray,
        objects: Iterable[str] | None = None,
        attributes: Iterable[str] | None = None,
    ) -> "FormalContext":
        """
        Create a formal context from an array-like binary matrix.
        """
        array = np.asarray(data)

        if array.ndim != 2:
            raise ValueError("Formal context data must be two-dimensional.")

        n_objects, n_attributes = array.shape

        if objects is None:
            objects = [f"g{i}" for i in range(n_objects)]

        if attributes is None:
            attributes = [f"m{j}" for j in range(n_attributes)]

        return cls(
            objects=tuple(objects),
            attributes=tuple(attributes),
            incidence=array,
        )

    @classmethod
    def from_cxt(cls, path):
        """
        Load a formal context from a Burmeister .cxt file.

        Compatibility wrapper for older ConceptFlow/DimFlux integration code.
        """
        from conceptflow.io import read_cxt

        return read_cxt(path)

    def _resolve_object_indices(self, objects: Iterable[int | str]) -> list[int]:
        """
        Resolve object indices or object names to integer indices.

        This keeps ConceptFlow internally index-based while allowing older
        code, such as the DimFlux bridge, to pass object names.
        """
        object_to_index = {
            name: index
            for index, name in enumerate(self.objects)
        }

        resolved: list[int] = []

        for obj in objects:
            if _is_index(obj):
                resolved.append(int(obj))
            else:
                resolved.append(object_to_index[obj])

        return resolved

    def _resolve_attribute_indices(
        self,
        attributes: Iterable[int | str],
    ) -> list[int]:
        """
        Resolve attribute indices or attribute names to integer indices.

        This keeps ConceptFlow internally index-based while allowing older
        code, such as the DimFlux bridge, to pass attribute names.
        """
        attribute_to_index = {
            name: index
            for index, name in enumerate(self.attributes)
        }

        resolved: list[int] = []

        for attr in attributes:
            if _is_index(attr):
                resolved.append(int(attr))
            else:
                resolved.append(attribute_to_index[attr])

        return resolved

    def object_derivation(self, objects: Iterable[int | str]) -> frozenset[int]:
        """
        Compute A' for a set of object indices or object names.
        """
        from conceptflow.algorithms.derivation import object_derivation

        resolved = self._resolve_object_indices(objects)
        return object_derivation(self, resolved)

    def attribute_derivation(
        self,
        attributes: Iterable[int | str],
    ) -> frozenset[int]:
        """
        Compute B' for a set of attribute indices or attribute names.
        """
        from conceptflow.algorithms.derivation import attribute_derivation

        resolved = self._resolve_attribute_indices(attributes)
        return attribute_derivation(self, resolved)

    def closure(self, attributes: Iterable[int | str]) -> frozenset[int]:
        """
        Compute B'' for a set of attribute indices or attribute names.
        """
        from conceptflow.algorithms.derivation import attribute_closure

        resolved = self._resolve_attribute_indices(attributes)
        return attribute_closure(self, resolved)

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert the formal context to a pandas DataFrame.
        """
        return pd.DataFrame(
            self.incidence,
            index=self.objects,
            columns=self.attributes,
        )

    def __repr__(self) -> str:
        return (
            f"FormalContext(n_objects={self.n_objects}, "
            f"n_attributes={self.n_attributes})"
        )
