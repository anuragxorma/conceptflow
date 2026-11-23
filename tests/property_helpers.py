"""
Shared pieces of the property-based tests (``test_properties_*.py``).

``Reference`` computes everything from the definitions with Python integers
as bit sets; it shares no code with ConceptFlow. The Hypothesis strategies
draw formal contexts of every shape the code has to handle: empty ones,
contexts without objects or attributes, duplicated and full or empty rows and
columns, and contexts wider than one 64-bit word.
"""

from __future__ import annotations

import itertools

import numpy as np
from hypothesis import strategies as st

from conceptflow import FormalContext

# ---------------------------------------------------------------------------
# Bit sets
# ---------------------------------------------------------------------------


def to_mask(indices) -> int:
    mask = 0
    for i in indices:
        mask |= 1 << int(i)
    return mask


def to_set(mask: int) -> frozenset[int]:
    found, i = [], 0
    while mask:
        if mask & 1:
            found.append(i)
        mask >>= 1
        i += 1
    return frozenset(found)


def submasks(mask: int):
    """Every subset of ``mask``, the empty set included."""
    sub = mask
    while True:
        yield sub
        if sub == 0:
            return
        sub = (sub - 1) & mask


# ---------------------------------------------------------------------------
# Reference implementation, from the definitions
# ---------------------------------------------------------------------------


class Reference:
    """Derivations, concepts, implications and order of a boolean matrix."""

    def __init__(self, incidence):
        incidence = np.asarray(incidence, dtype=bool)
        self.g, self.m = incidence.shape
        self.all_objects = (1 << self.g) - 1
        self.all_attributes = (1 << self.m) - 1
        self.rows = [to_mask(np.flatnonzero(incidence[i])) for i in range(self.g)]
        self.columns = [to_mask(np.flatnonzero(incidence[:, j])) for j in range(self.m)]

    # Derivation operators -------------------------------------------------

    def extent(self, attributes: int) -> int:
        """B': the objects having every attribute of B."""
        return sum(1 << i for i, row in enumerate(self.rows) if row & attributes == attributes)

    def intent(self, objects: int) -> int:
        """A': the attributes every object of A has."""
        return sum(
            1 << j for j, column in enumerate(self.columns) if column & objects == objects
        )

    def closure(self, attributes: int) -> int:
        return self.intent(self.extent(attributes))

    # Concepts --------------------------------------------------------------

    def concepts(self) -> set[tuple[int, int]]:
        """Every (A, B) with A' = B and B' = A, from the smaller side."""
        if self.g <= self.m:
            intents = {self.intent(a) for a in range(1 << self.g)}
        else:
            intents = {self.closure(b) for b in range(1 << self.m)}
        return {(self.extent(b), b) for b in intents}

    def concept_sets(self) -> set[tuple[frozenset[int], frozenset[int]]]:
        return {(to_set(a), to_set(b)) for a, b in self.concepts()}

    # Implications ----------------------------------------------------------

    def pseudo_intents(self) -> dict[int, int]:
        """Pseudo-intent -> its closure, by the recursive definition."""
        found: dict[int, int] = {}
        for p in sorted(range(1 << self.m), key=int.bit_count):
            closed = self.closure(p)
            if closed != p and all(
                closure & p == closure for q, closure in found.items() if q & p == q and q != p
            ):
                found[p] = closed
        return found

    def holds(self, premise: int, conclusion: int) -> bool:
        return self.extent(premise) & self.extent(conclusion) == self.extent(premise)

    # Order -----------------------------------------------------------------

    @staticmethod
    def covers(extents: list[int]) -> set[tuple[int, int]]:
        """(lower, upper) positions of the cover relation of extent inclusion."""
        def below(a, b):
            return a != b and a & b == a

        pairs = set()
        for i, a in enumerate(extents):
            for j, b in enumerate(extents):
                if below(a, b) and not any(
                    below(a, c) and below(c, b) for c in extents
                ):
                    pairs.add((i, j))
        return pairs


def implication_closure(attributes: int, implications) -> int:
    """The closure of ``attributes`` under (premise, conclusion) masks."""
    closed = attributes
    changed = True
    while changed:
        changed = False
        for premise, conclusion in implications:
            if premise & closed == premise and conclusion & closed != conclusion:
                closed |= conclusion
                changed = True
    return closed


def as_masks(implications) -> list[tuple[int, int]]:
    return [(to_mask(i.premise), to_mask(i.conclusion)) for i in implications]


def sample_masks(data, n_bits: int, count: int = 12) -> list[int]:
    """Every subset of a small set, or ``count`` drawn subsets of a large one."""
    if n_bits <= 7:
        return list(range(1 << n_bits))
    return data.draw(
        st.lists(st.integers(0, (1 << n_bits) - 1), min_size=1, max_size=count),
        label="attribute sets",
    )


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------


@st.composite
def incidences(draw, max_objects=7, max_attributes=7, min_objects=0, min_attributes=0):
    """
    A boolean matrix, sometimes with structure random cells rarely produce:
    duplicated rows or columns, or full and empty rows or columns.
    """
    g = draw(st.integers(min_objects, max_objects), label="objects")
    m = draw(st.integers(min_attributes, max_attributes), label="attributes")
    density = draw(st.sampled_from([0.5, 0.2, 0.8]), label="density")
    if density == 0.5:
        cells = draw(st.lists(st.booleans(), min_size=g * m, max_size=g * m))
        incidence = np.array(cells, dtype=bool).reshape(g, m)
    else:
        seed = draw(st.integers(0, 2**32 - 1), label="seed")
        incidence = np.random.default_rng(seed).random((g, m)) < density

    if g > 1 and draw(st.booleans(), label="duplicate rows"):
        incidence = incidence[draw(st.lists(st.integers(0, g - 1), min_size=g, max_size=g))]
    if m > 1 and draw(st.booleans(), label="duplicate columns"):
        incidence = incidence[:, draw(st.lists(st.integers(0, m - 1), min_size=m, max_size=m))]
    if g and draw(st.booleans(), label="full or empty row"):
        incidence = incidence.copy()
        incidence[draw(st.integers(0, g - 1))] = draw(st.booleans())
    if m and draw(st.booleans(), label="full or empty column"):
        incidence = incidence.copy()
        incidence[:, draw(st.integers(0, m - 1))] = draw(st.booleans())
    return incidence


@st.composite
def wide_incidences(draw):
    """More than 64 attributes (few objects), or more than 64 objects (few attributes)."""
    long_side = draw(st.sampled_from([63, 64, 65, 127, 128, 129]), label="long side")
    short_side = draw(st.integers(1, 5), label="short side")
    seed = draw(st.integers(0, 2**32 - 1), label="seed")
    density = draw(st.sampled_from([0.3, 0.6, 0.9]), label="density")
    rng = np.random.default_rng(seed)
    incidence = rng.random((short_side, long_side)) < density
    if draw(st.booleans(), label="duplicate columns"):
        incidence = incidence[:, rng.integers(0, long_side, long_side)]
    if draw(st.booleans(), label="tall"):
        incidence = incidence.T
    return incidence


def context_of(incidence) -> FormalContext:
    return FormalContext.from_array(np.asarray(incidence, dtype=bool))


@st.composite
def contexts(draw, **kwargs):
    return context_of(draw(incidences(**kwargs)))


# Names that file formats have to store: spaces, separators, quotes, markup,
# non-ASCII and control characters.
names = st.text(
    alphabet=st.characters(codec="utf-8", exclude_categories=("Cs",)),
    max_size=8,
)


@st.composite
def named_contexts(draw, max_objects=5, max_attributes=5):
    incidence = draw(incidences(max_objects=max_objects, max_attributes=max_attributes))
    g, m = incidence.shape
    objects = draw(st.lists(names, min_size=g, max_size=g, unique=True), label="object names")
    attributes = draw(
        st.lists(names, min_size=m, max_size=m, unique=True), label="attribute names"
    )
    return FormalContext(objects=tuple(objects), attributes=tuple(attributes), incidence=incidence)


def all_subsets(items):
    items = sorted(items)
    for size in range(len(items) + 1):
        yield from (frozenset(s) for s in itertools.combinations(items, size))
