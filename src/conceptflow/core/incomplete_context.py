"""
Incomplete (three-valued) formal contexts for ConceptFlow.

In an incomplete context every object-attribute pair is in one of three
states (Burmeister & Holzer, 2000):

- known yes: the object certainly has the attribute,
- known no: the object certainly lacks the attribute,
- unknown: the entry has not been determined.

A *completion* replaces every unknown entry by yes or no. Questions about
an incomplete context are answered "certainly" (true in every completion)
or "possibly" (true in at least one completion). The two extreme
completions are the certain context (every unknown -> no) and the possible
context (every unknown -> yes); both are ordinary ``FormalContext`` objects
and can be fed to every other part of ConceptFlow.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Any, Iterable

import numpy as np
import pandas as pd

from conceptflow.core.context import FormalContext, _is_index

UNKNOWN_MARKERS = frozenset({"?"})


def is_missing(value: Any) -> bool:
    """
    Return whether a single cell value marks a missing / unknown entry.

    Missing are None, NaN, NaT, ``pd.NA`` and the string ``"?"`` (the usual
    marker in UCI-style data). ``ConceptualScaler`` and ``IncompleteContext``
    share this definition.
    """
    if value is None:
        return True
    if isinstance(value, str):
        return value in UNKNOWN_MARKERS
    return bool(pd.api.types.is_scalar(value) and pd.isna(value))


def _parse_cell(value: Any) -> bool | None:
    """
    Parse one cell into True, False, or None (unknown).

    Only booleans and the numbers 0/1 are accepted as known values. Anything
    else raises instead of being coerced with ``bool()``, which would turn
    e.g. the string ``"no"`` into True.
    """
    if is_missing(value):
        return None

    if isinstance(value, (bool, np.bool_)):
        return bool(value)

    if isinstance(value, (int, float, np.integer, np.floating)) and value in (0, 1):
        return bool(value)

    raise ValueError(
        f"Cannot interpret {value!r} as a three-valued incidence entry. "
        "Use True/False or 1/0 for known entries and None, NaN or '?' for "
        "unknown ones."
    )


def _parse_table(rows: Iterable[Iterable[Any]]) -> tuple[np.ndarray, np.ndarray]:
    """Parse a 2D table into ``(certain, unknown)`` boolean masks."""
    parsed = [[_parse_cell(value) for value in row] for row in rows]
    n_columns = len(parsed[0]) if parsed else 0

    certain = np.array(
        [[value is True for value in row] for row in parsed], dtype=bool
    ).reshape(len(parsed), n_columns)
    unknown = np.array(
        [[value is None for value in row] for row in parsed], dtype=bool
    ).reshape(len(parsed), n_columns)

    return certain, unknown


@dataclass(frozen=True)
class IncompleteContext:
    """
    Formal context with known-yes, known-no and unknown entries.

    Parameters
    ----------
    objects:
        Object names.

    attributes:
        Attribute names.

    certain:
        Boolean matrix of shape ``(n_objects, n_attributes)``; True where the
        object is known to have the attribute.

    unknown:
        Boolean matrix of the same shape; True where the entry is unknown.
        Must not overlap with ``certain``. Entries that are False in both
        matrices are known to be absent.
    """

    objects: tuple[str, ...]
    attributes: tuple[str, ...]
    certain: np.ndarray
    unknown: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "objects", tuple(map(str, self.objects)))
        object.__setattr__(self, "attributes", tuple(map(str, self.attributes)))

        if len(set(self.objects)) != len(self.objects):
            raise ValueError("Object names must be unique.")

        if len(set(self.attributes)) != len(self.attributes):
            raise ValueError("Attribute names must be unique.")

        # Copies, read-only: the context cannot change after creation.
        certain = np.array(self.certain, dtype=bool)
        unknown = np.array(self.unknown, dtype=bool)
        expected_shape = (len(self.objects), len(self.attributes))

        for name, matrix in (("certain", certain), ("unknown", unknown)):
            if matrix.shape != expected_shape:
                raise ValueError(
                    f"{name} matrix has shape {matrix.shape}, but expected "
                    f"{expected_shape}."
                )

        if (certain & unknown).any():
            raise ValueError("An entry cannot be both certain and unknown.")

        certain.flags.writeable = False
        unknown.flags.writeable = False

        object.__setattr__(self, "certain", certain)
        object.__setattr__(self, "unknown", unknown)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, IncompleteContext):
            return NotImplemented
        return (
            self.objects == other.objects
            and self.attributes == other.attributes
            and np.array_equal(self.certain, other.certain)
            and np.array_equal(self.unknown, other.unknown)
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.objects,
                self.attributes,
                self.certain.tobytes(),
                self.unknown.tobytes(),
            )
        )

    @property
    def n_objects(self) -> int:
        """Number of objects."""
        return len(self.objects)

    @property
    def n_attributes(self) -> int:
        """Number of attributes."""
        return len(self.attributes)

    @property
    def n_unknown(self) -> int:
        """Number of unknown entries."""
        return int(self.unknown.sum())

    @cached_property
    def possible(self) -> np.ndarray:
        """True where the object has, or might have, the attribute."""
        return self.certain | self.unknown

    @property
    def is_complete(self) -> bool:
        """Whether there are no unknown entries."""
        return not self.unknown.any()

    @classmethod
    def from_dataframe(cls, data: pd.DataFrame) -> "IncompleteContext":
        """
        Create an incomplete context from a DataFrame.

        Known entries must be True/False or 1/0. None, NaN, ``pd.NA`` and
        the string ``"?"`` mark unknown entries.
        """
        certain, unknown = _parse_table(data.itertuples(index=False, name=None))
        return cls(
            objects=tuple(map(str, data.index)),
            attributes=tuple(map(str, data.columns)),
            certain=certain.reshape(data.shape),
            unknown=unknown.reshape(data.shape),
        )

    @classmethod
    def from_array(
        cls,
        data,
        objects: Iterable[str] | None = None,
        attributes: Iterable[str] | None = None,
    ) -> "IncompleteContext":
        """
        Create an incomplete context from a 2D array-like.

        Known entries must be True/False or 1/0. None, NaN and ``"?"`` mark
        unknown entries.
        """
        array = np.asarray(data, dtype=object)

        if array.ndim != 2:
            raise ValueError("Incomplete context data must be two-dimensional.")

        n_objects, n_attributes = array.shape
        certain, unknown = _parse_table(array.tolist())

        if objects is None:
            objects = [f"g{i}" for i in range(n_objects)]

        if attributes is None:
            attributes = [f"m{j}" for j in range(n_attributes)]

        return cls(
            objects=tuple(objects),
            attributes=tuple(attributes),
            certain=certain.reshape(array.shape),
            unknown=unknown.reshape(array.shape),
        )

    @classmethod
    def from_formal_context(cls, context: FormalContext) -> "IncompleteContext":
        """Wrap a complete formal context (no unknown entries)."""
        return cls(
            objects=context.objects,
            attributes=context.attributes,
            certain=context.incidence,
            unknown=np.zeros_like(context.incidence, dtype=bool),
        )

    def certain_context(self) -> FormalContext:
        """
        The completion in which every unknown entry is absent.

        This is one particular completion (a closed-world reading of the
        data), not "what is known for sure": an implication can hold here
        and still fail in another completion. Use ``implication_status`` to
        ask whether an implication holds in every completion.
        """
        return FormalContext(
            objects=self.objects,
            attributes=self.attributes,
            incidence=self.certain,
        )

    def possible_context(self) -> FormalContext:
        """The completion in which every unknown entry is present."""
        return FormalContext(
            objects=self.objects,
            attributes=self.attributes,
            incidence=self.possible,
        )

    @staticmethod
    def _resolve(items: Iterable[int | str], names: tuple[str, ...], kind: str) -> list[int]:
        if isinstance(items, str):
            items = [items]
        index_of = {name: i for i, name in enumerate(names)}
        resolved = []
        for item in items:
            if _is_index(item):
                if not 0 <= item < len(names):
                    hint = (
                        f" Integers are positions; for the {kind} named '{item}', "
                        f"pass the string '{item}'."
                        if str(item) in index_of else ""
                    )
                    raise IndexError(
                        f"{kind} index {item} is out of range for {len(names)} {kind}s.{hint}"
                    )
                resolved.append(int(item))
            else:
                try:
                    resolved.append(index_of[item])
                except KeyError:
                    raise KeyError(f"Unknown {kind} {item!r}.") from None
        return resolved

    def _attribute_indices(self, attributes: Iterable[int | str]) -> list[int]:
        return self._resolve(attributes, self.attributes, "attribute")

    def _object_indices(self, objects: Iterable[int | str]) -> list[int]:
        return self._resolve(objects, self.objects, "object")

    def certain_extent(self, attributes: Iterable[int | str]) -> frozenset[int]:
        """Objects known to have every given attribute."""
        columns = self._attribute_indices(attributes)
        mask = self.certain[:, columns].all(axis=1)
        return frozenset(np.flatnonzero(mask).tolist())

    def possible_extent(self, attributes: Iterable[int | str]) -> frozenset[int]:
        """Objects not known to lack any of the given attributes."""
        columns = self._attribute_indices(attributes)
        mask = self.possible[:, columns].all(axis=1)
        return frozenset(np.flatnonzero(mask).tolist())

    def certain_intent(self, objects: Iterable[int | str]) -> frozenset[int]:
        """Attributes every given object is known to have."""
        rows = self._object_indices(objects)
        mask = self.certain[rows, :].all(axis=0)
        return frozenset(np.flatnonzero(mask).tolist())

    def possible_intent(self, objects: Iterable[int | str]) -> frozenset[int]:
        """Attributes no given object is known to lack."""
        rows = self._object_indices(objects)
        mask = self.possible[rows, :].all(axis=0)
        return frozenset(np.flatnonzero(mask).tolist())

    def implication_counterexamples(
        self,
        premise: Iterable[int | str],
        conclusion: Iterable[int | str],
    ) -> tuple[frozenset[int], frozenset[int]]:
        """
        Return ``(certain, possible)`` counterexamples of premise -> conclusion.

        - certain: objects that violate the implication in every completion
          (they certainly have the premise and certainly lack some
          conclusion attribute).
        - possible: objects that violate it in at least one completion. This
          is a superset of the certain counterexamples.
        """
        premise_idx = set(self._attribute_indices(premise))
        conclusion_idx = set(self._attribute_indices(conclusion)) - premise_idx

        certain_counter = self.certain_extent(premise_idx) - self.possible_extent(
            conclusion_idx
        )
        possible_counter = self.possible_extent(premise_idx) - self.certain_extent(
            conclusion_idx
        )

        return certain_counter, possible_counter

    def implication_status(
        self,
        premise: Iterable[int | str],
        conclusion: Iterable[int | str],
    ) -> str:
        """
        Classify an implication premise -> conclusion.

        Returns
        -------
        str
            - ``"valid"``: holds in every completion.
            - ``"possible"``: holds in some completions but not all.
            - ``"refuted"``: holds in no completion.

        Each object can be completed independently, so a completion in which
        the implication holds exists iff no object is a *certain*
        counterexample.
        """
        certain_counter, possible_counter = self.implication_counterexamples(
            premise, conclusion
        )

        if certain_counter:
            return "refuted"

        if possible_counter:
            return "possible"

        return "valid"

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert to a DataFrame of pandas' nullable ``boolean`` dtype.

        Unknown entries become ``pd.NA``.
        """
        frame = pd.DataFrame(
            self.certain,
            index=self.objects,
            columns=self.attributes,
        ).astype("boolean")

        return frame.mask(
            pd.DataFrame(self.unknown, index=self.objects, columns=self.attributes)
        )

    def __repr__(self) -> str:
        return (
            f"IncompleteContext(n_objects={self.n_objects}, "
            f"n_attributes={self.n_attributes}, n_unknown={self.n_unknown})"
        )
