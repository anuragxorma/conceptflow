"""
Many-valued contexts for ConceptFlow.

A many-valued context stores real-world tabular data before conceptual
scaling.

Conceptual scaling transforms a many-valued context into a binary formal
context.

The intended pipeline is therefore:

    many-valued data
        ↓
    ManyValuedContext
        ↓
    conceptual scaling
        ↓
    FormalContext

Design philosophy
-----------------
This class intentionally remains separate from:

- binary FCA logic,
- conceptual scales,
- scaling algorithms,
- sklearn estimators,
- visualization.

This separation is important for keeping ConceptFlow modular and
extensible.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ManyValuedContext:
    """
    Many-valued context representation.

    Parameters
    ----------
    objects:
        Object names.

    attributes:
        Attribute names.

    data:
        Tabular many-valued data.

    Notes
    -----
    Internally, data is stored as a pandas DataFrame because:

    - it preserves column metadata,
    - it interoperates naturally with the sklearn ecosystem,
    - it supports heterogeneous types,
    - it simplifies preprocessing and automatic scaling,
    - it enables vectorized operations.
    """

    objects: tuple[str, ...]
    attributes: tuple[str, ...]
    data: pd.DataFrame

    def __post_init__(self) -> None:
        """
        Validate and normalize internal structure.
        """
        objects = tuple(map(str, self.objects))
        attributes = tuple(map(str, self.attributes))

        if len(set(objects)) != len(objects):
            raise ValueError("Object names must be unique.")

        if len(set(attributes)) != len(attributes):
            raise ValueError("Attribute names must be unique.")

        if isinstance(self.data, pd.DataFrame):
            data = self.data.copy()
            data.index = objects
            data.columns = attributes
        else:
            data = pd.DataFrame(
                self.data,
                index=objects,
                columns=attributes,
            )

        expected_shape = (len(objects), len(attributes))

        if data.shape != expected_shape:
            raise ValueError(
                f"Data has shape {data.shape}, "
                f"but expected {expected_shape}."
            )

        object.__setattr__(self, "objects", objects)
        object.__setattr__(self, "attributes", attributes)
        object.__setattr__(self, "data", data)

    @property
    def n_objects(self) -> int:
        """Number of objects."""
        return len(self.objects)

    @property
    def n_attributes(self) -> int:
        """Number of attributes."""
        return len(self.attributes)

    @property
    def size(self) -> tuple[int, int]:
        """
        Context size as:

            (n_objects, n_attributes)
        """
        return self.n_objects, self.n_attributes

    @classmethod
    def from_dataframe(cls, data: pd.DataFrame) -> "ManyValuedContext":
        """
        Construct a ManyValuedContext from a DataFrame.
        """
        if data.empty:
            raise ValueError(
                "Cannot construct ManyValuedContext from an empty DataFrame."
            )

        return cls(
            objects=tuple(map(str, data.index)),
            attributes=tuple(map(str, data.columns)),
            data=data.copy(),
        )

    @classmethod
    def from_array(
        cls,
        data,
        objects: Iterable[str] | None = None,
        attributes: Iterable[str] | None = None,
    ) -> "ManyValuedContext":
        """
        Construct a ManyValuedContext from array-like data.
        """
        array = np.asarray(data)

        if array.ndim != 2:
            raise ValueError(
                "Many-valued context data must be two-dimensional."
            )

        n_objects, n_attributes = array.shape

        if objects is None:
            objects = [f"g{i}" for i in range(n_objects)]

        if attributes is None:
            attributes = [f"m{j}" for j in range(n_attributes)]

        dataframe = pd.DataFrame(
            array,
            index=objects,
            columns=attributes,
        )

        return cls(
            objects=tuple(objects),
            attributes=tuple(attributes),
            data=dataframe,
        )

    def get_value(self, obj: str, attr: str) -> Any:
        """
        Return one object-attribute value.
        """
        if obj not in self.objects:
            raise KeyError(f'Unknown object "{obj}".')

        if attr not in self.attributes:
            raise KeyError(f'Unknown attribute "{attr}".')

        return self.data.loc[obj, attr]

    def row(self, obj: str) -> dict[str, Any]:
        """
        Return one row as a dictionary.
        """
        if obj not in self.objects:
            raise KeyError(f'Unknown object "{obj}".')

        return self.data.loc[obj].to_dict()

    def column(self, attr: str) -> pd.Series:
        """
        Return one attribute column.
        """
        if attr not in self.attributes:
            raise KeyError(f'Unknown attribute "{attr}".')

        return self.data[attr]

    def unique_values(self, attr: str) -> set[Any]:
        """
        Return unique values of one attribute.
        """
        if attr not in self.attributes:
            raise KeyError(f'Unknown attribute "{attr}".')

        return set(self.data[attr].unique())

    def restrict_objects(
        self,
        objects: Iterable[str],
    ) -> "ManyValuedContext":
        """
        Restrict the context to selected objects.

        This is especially important for nested conceptual exploration,
        where child views should be built from subsets of the original
        many-valued data rather than already-scaled binary contexts.
        """
        selected = tuple(map(str, objects))
        selected_set = set(selected)

        unknown = selected_set - set(self.objects)

        if unknown:
            raise KeyError(
                f"Unknown objects: {sorted(unknown)}"
            )

        restricted_data = self.data.loc[list(selected)]

        return ManyValuedContext(
            objects=selected,
            attributes=self.attributes,
            data=restricted_data,
        )

    def to_dataframe(self) -> pd.DataFrame:
        """
        Return a copy of the internal DataFrame.
        """
        return self.data.copy()

    def __repr__(self) -> str:
        return (
            "ManyValuedContext("
            f"n_objects={self.n_objects}, "
            f"n_attributes={self.n_attributes}"
            ")"
        )