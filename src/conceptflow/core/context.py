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
from functools import cached_property
from typing import Iterable

import numpy as np
import pandas as pd


def _is_index(value) -> bool:
    """Whether ``value`` is an integer index (Python or NumPy), not a name."""
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


TRUE_STRINGS = frozenset({"1", "1.0", "true", "True", "TRUE", "x", "X", "×"})
FALSE_STRINGS = frozenset({"0", "0.0", "false", "False", "FALSE", ".", ""})


def _cell_is_cross(value) -> bool:
    """One cell, by one rule independent of the other cells."""
    if value is None:
        return False
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (str, bytes)):
        text = value.decode() if isinstance(value, bytes) else value
        text = text.strip()
        if text in TRUE_STRINGS:
            return True
        if text in FALSE_STRINGS:
            return False
        if text == "?":
            raise ValueError(
                'Incidence contains the unknown marker "?". A FormalContext has '
                "only crosses and non-crosses; use IncompleteContext for unknown "
                "entries."
            )
        raise ValueError(
            f"Cannot read {value!r} as a cross or non-cross. Use booleans, "
            "numbers (> 0 is a cross) or one of "
            f"{sorted(TRUE_STRINGS)} / {sorted(FALSE_STRINGS - {''})} or empty."
        )
    if pd.isna(value):
        return False
    try:
        return float(value) > 0
    except (TypeError, ValueError):
        raise ValueError(f"Cannot read {value!r} as a cross or non-cross.") from None


def _cross_table(data) -> np.ndarray:
    """
    Convert incidence data to a boolean cross table, cell by cell:

    - booleans are taken as they are;
    - numbers: values greater than 0 are crosses (0/1 data and counts work
      as expected; ±1-encoded data reads -1 as absent);
    - empty cells (None, NaN, ``pd.NA``) mean "no cross", as in a printed
      cross table (``bool(NaN)`` would be True, so they never go through
      ``bool()``);
    - strings must be a known spelling: "1", "true", "x", "×" ... for a
      cross, "0", "false", ".", "" for none. Anything else (e.g. "yes",
      "no") raises instead of being guessed; "?" (unknown) points to
      IncompleteContext.
    """
    raw = np.asarray(data)

    if raw.dtype == bool:
        # Copy, so later changes to the caller's array cannot alter a frozen
        # context.
        return raw.copy()

    if raw.dtype.kind in "iuf":
        return np.nan_to_num(raw, nan=0.0) > 0

    cells = [_cell_is_cross(value) for value in raw.ravel().tolist()]
    return np.array(cells, dtype=bool).reshape(raw.shape)


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
        if incidence.size == 0 and 0 in expected_shape:
            # e.g. [] for a context without objects
            incidence = np.zeros(expected_shape, dtype=bool)
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

    @cached_property
    def _object_index(self) -> dict[str, int]:
        return {name: index for index, name in enumerate(self.objects)}

    @cached_property
    def _attribute_index(self) -> dict[str, int]:
        return {name: index for index, name in enumerate(self.attributes)}

    @staticmethod
    def _resolve(items, index_of: dict[str, int], size: int, kind: str) -> list[int]:
        """
        Names -> indices; integers are positions and must be in range.

        Names are always strings (FormalContext converts them), so an integer
        is never mistaken for a name.
        """
        resolved: list[int] = []
        for item in items:
            if _is_index(item):
                if not 0 <= item < size:
                    raise IndexError(
                        f"{kind} index {item} is out of range for {size} {kind}s."
                    )
                resolved.append(int(item))
            else:
                try:
                    resolved.append(index_of[item])
                except KeyError:
                    raise KeyError(f"Unknown {kind} {item!r}.") from None
        return resolved

    def _resolve_object_indices(self, objects: Iterable[int | str]) -> list[int]:
        """Resolve object names or indices to integer indices."""
        return self._resolve(objects, self._object_index, self.n_objects, "object")

    def _resolve_attribute_indices(
        self,
        attributes: Iterable[int | str],
    ) -> list[int]:
        """Resolve attribute names or indices to integer indices."""
        return self._resolve(attributes, self._attribute_index, self.n_attributes, "attribute")

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

    # ------------------------------------------------------------------
    # Building new contexts (a FormalContext is immutable)
    # ------------------------------------------------------------------

    def transpose(self) -> "FormalContext":
        """The dual context: objects and attributes swap roles."""
        return FormalContext(
            objects=self.attributes,
            attributes=self.objects,
            incidence=self.incidence.T,
        )

    def subcontext(
        self,
        objects: Iterable[int | str] | None = None,
        attributes: Iterable[int | str] | None = None,
    ) -> "FormalContext":
        """
        Restrict to the given objects and/or attributes, in the given order.

        ``None`` keeps all of them. Names and indices are both accepted.
        """
        rows = (
            list(range(self.n_objects))
            if objects is None
            else self._resolve_object_indices(objects)
        )
        columns = (
            list(range(self.n_attributes))
            if attributes is None
            else self._resolve_attribute_indices(attributes)
        )
        return FormalContext(
            objects=tuple(self.objects[i] for i in rows),
            attributes=tuple(self.attributes[j] for j in columns),
            incidence=self.incidence[
                np.ix_(np.asarray(rows, dtype=int), np.asarray(columns, dtype=int))
            ],
        )

    def remove_objects(self, objects: Iterable[int | str]) -> "FormalContext":
        """A copy without the given objects."""
        drop = set(self._resolve_object_indices(objects))
        return self.subcontext(objects=[i for i in range(self.n_objects) if i not in drop])

    def remove_attributes(self, attributes: Iterable[int | str]) -> "FormalContext":
        """A copy without the given attributes."""
        drop = set(self._resolve_attribute_indices(attributes))
        return self.subcontext(
            attributes=[j for j in range(self.n_attributes) if j not in drop]
        )

    def add_objects(self, names: Iterable[str], rows) -> "FormalContext":
        """
        A copy with extra objects.

        ``rows`` has one row per new object and one column per attribute.
        """
        names = tuple(map(str, names))
        rows = _cross_table(rows).reshape(len(names), self.n_attributes)
        return FormalContext(
            objects=self.objects + names,
            attributes=self.attributes,
            incidence=np.vstack([self.incidence, rows]),
        )

    def add_attributes(self, names: Iterable[str], columns) -> "FormalContext":
        """
        A copy with extra attributes.

        ``columns`` has one row per object and one column per new attribute.
        """
        names = tuple(map(str, names))
        columns = _cross_table(columns).reshape(self.n_objects, len(names))
        return FormalContext(
            objects=self.objects,
            attributes=self.attributes + names,
            incidence=np.hstack([self.incidence, columns]),
        )

    def rename(self, objects=None, attributes=None) -> "FormalContext":
        """
        A copy with renamed objects and/or attributes.

        Each argument is a mapping (old name -> new name; names not in it are
        kept) or a function applied to every name.
        """
        def renamed(names, how):
            if how is None:
                return names
            if callable(how):
                return tuple(str(how(name)) for name in names)
            return tuple(str(how.get(name, name)) for name in names)

        return FormalContext(
            objects=renamed(self.objects, objects),
            attributes=renamed(self.attributes, attributes),
            incidence=self.incidence,
        )

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
