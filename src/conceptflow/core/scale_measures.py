"""
Scale-measures: coarser views of a formal context on the same objects.

Following Hanika & Hirth, "On the lattice of conceptual measurements" (2022).
A scale-measure of a context K keeps K's objects and makes fewer
distinctions between them: every extent of the scale is an extent of K, so
its concept lattice is a coarser view of K's.

- ``scale_measure(context, attributes)`` builds one whose attributes are
  original attributes (attribute projection, Cor. 22) or conjunctions of
  them (Cor. 17); both are scale-measures unconditionally. Disjunctions and
  negations are not offered: they give a scale-measure only when the
  resulting extent happens to be an extent of K (Prop. 16).
- ``is_scale_measure(original, scale)`` checks the property for a context on
  the same objects, attribute by attribute (Prop. 20), and
  ``scale_measure_problems`` says which attributes fail.
- ``refines(coarse, fine)`` tells whether every extent of ``coarse`` is an
  extent of ``fine``. Such a *refinement pair* has every concept of the
  coarser view among the concepts of the finer one, so the two lattices (and
  their drawings) can be compared concept by concept.

These scales are not the conceptual scales of ``conceptflow.preprocessing``,
which turn many-valued data into a formal context.
"""

from __future__ import annotations

import numbers
from collections.abc import Iterable

import numpy as np

from conceptflow.core.context import FormalContext
from conceptflow.core.lattice import ConceptLattice

# Joins the names of the parts of a conjunction in the scale's attribute name.
CONJUNCTION = " ∧ "


def scale_measure(
    context: FormalContext,
    attributes: Iterable[str | int | Iterable[str | int]] | None,
) -> FormalContext:
    """
    The scale-measure of ``context`` with the given attributes.

    Each entry of ``attributes`` is one attribute of ``context`` (a name or
    a position), kept as it is, or a tuple of them, which becomes their
    conjunction: an object has it iff it has every part. The conjunction of
    ``"a"`` and ``"b"`` is named ``"a ∧ b"``. ``None`` returns ``context``.

    The objects stay those of ``context``, in the same order. The attributes
    are put in one canonical order: by the positions of their parts in
    ``context``, compared as sorted tuples. Two scales of one context then
    list their common attributes in the same relative order, which matters
    for layouts that depend on column order.

    Raises
    ------
    KeyError
        For a name that is not an attribute of ``context``.
    ValueError
        For an empty conjunction, or two entries that give the same name.
    """
    if attributes is None:
        return context

    entries = []
    for entry in attributes:
        parts = (entry,) if isinstance(entry, (str, numbers.Integral)) else tuple(entry)
        if not parts:
            raise ValueError("A conjunction needs at least one attribute.")
        indices = context._resolve_attribute_indices(parts)
        entries.append((tuple(sorted(indices)), indices))
    entries.sort(key=lambda item: item[0])  # stable: equal keys keep their order

    names = [CONJUNCTION.join(context.attributes[i] for i in indices) for _, indices in entries]
    if len(set(names)) != len(names):
        raise ValueError(f"Two attributes of the scale would have the same name: {names}.")
    incidence = np.ones((context.n_objects, len(entries)), dtype=bool)
    for column, (_, indices) in enumerate(entries):
        for index in indices:
            incidence[:, column] &= context.incidence[:, index]
    return FormalContext(objects=context.objects, attributes=tuple(names), incidence=incidence)


def scale_measure_problems(original: FormalContext, scale: FormalContext) -> list[str]:
    """
    Why ``scale`` is not a scale-measure of ``original`` (empty if it is).

    The scale must have the same objects, in the same order (the map between
    them is the identity). It is then a scale-measure iff the extent of each
    of its attributes is an extent of ``original``, i.e. equals its own
    closure there (Hanika & Hirth, Prop. 20). That is one closure per
    attribute; the scale's lattice is not enumerated.
    """
    if tuple(scale.objects) != tuple(original.objects):
        return ["object sets differ, so the identity is not a map between them"]
    problems = []
    for index, name in enumerate(scale.attributes):
        extent = scale.attribute_derivation([index])
        closed = original.attribute_derivation(original.object_derivation(extent))
        if closed != extent:
            problems.append(
                f"attribute {name!r}: its extent has {len(extent)} objects but closes to "
                f"{len(closed)} in the original context, so it is not an extent there"
            )
    return problems


def is_scale_measure(original: FormalContext, scale: FormalContext) -> bool:
    """Whether ``scale`` is a scale-measure of ``original`` (see ``scale_measure_problems``)."""
    return not scale_measure_problems(original, scale)


def _extents_by_name(view: FormalContext | ConceptLattice) -> tuple[tuple, set[frozenset]]:
    if isinstance(view, ConceptLattice):
        if view.is_iceberg:
            raise ValueError("refines() needs full lattices, not iceberg lattices.")
        lattice = view
    else:
        lattice = ConceptLattice.from_context(view)
    objects = lattice.context.objects
    return objects, {frozenset(objects[i] for i in concept.extent) for concept in lattice.concepts}


def refines(
    coarse: FormalContext | ConceptLattice,
    fine: FormalContext | ConceptLattice,
) -> bool:
    """
    Whether every extent of ``coarse`` is an extent of ``fine``.

    Then every concept of the coarser view is a concept of the finer one,
    matched by its extent, and the two form a *refinement pair*. Both must
    have the same objects (matched by name); contexts or full concept
    lattices can be given, a lattice saving its recomputation.
    """
    coarse_objects, coarse_extents = _extents_by_name(coarse)
    fine_objects, fine_extents = _extents_by_name(fine)
    if set(coarse_objects) != set(fine_objects):
        raise ValueError("refines() compares views of the same objects; these differ.")
    return coarse_extents <= fine_extents
