"""
Operations that build new formal contexts from existing ones.

Following Ganter & Wille, *Formal Concept Analysis*:

- ``apposition(K1, K2)``: side by side; same objects, attributes of both.
- ``subposition(K1, K2)``: stacked; same attributes, objects of both.
- ``direct_sum(K1, K2)``: disjoint union where each object has all
  attributes of the other context; its concept lattice is the product of
  the two lattices.
- ``direct_product(K1, K2)``: objects G1 x G2, attributes M1 x M2, and
  (g1, g2) has (m1, m2) iff g1 has m1 or g2 has m2.
- ``complement(K)``: crosses and non-crosses swapped.
- ``dual(K)``: objects and attributes swapped.

Names must stay unique. When both contexts use a name, pass
``prefixes=("A:", "B:")`` to prefix the names of each context; otherwise a
``ValueError`` names the clash.
"""

from __future__ import annotations

import numpy as np

from conceptflow.core.context import FormalContext


def _disjoint_names(
    first: tuple[str, ...],
    second: tuple[str, ...],
    prefixes: tuple[str, str] | None,
    what: str,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if prefixes is not None:
        a, b = prefixes
        if a == b:
            raise ValueError("The two prefixes must differ.")
        return tuple(a + n for n in first), tuple(b + n for n in second)

    clash = set(first) & set(second)
    if clash:
        raise ValueError(
            f"Both contexts have the {what} {sorted(clash)}. Pass "
            'prefixes=("A:", "B:") to keep them apart.'
        )
    return first, second


def _same_names(first, second, what: str) -> list[int]:
    """Positions in ``second`` of the names of ``first`` (same set required)."""
    if set(first) != set(second) or len(first) != len(second):
        raise ValueError(f"Both contexts must have the same {what}.")
    position = {name: i for i, name in enumerate(second)}
    return [position[name] for name in first]


def apposition(
    left: FormalContext,
    right: FormalContext,
    prefixes: tuple[str, str] | None = None,
) -> FormalContext:
    """
    Put two contexts with the same objects side by side.

    Objects are matched by name (order may differ); the result uses the
    object order of ``left``.
    """
    order = _same_names(left.objects, right.objects, "objects")
    left_attributes, right_attributes = _disjoint_names(
        left.attributes, right.attributes, prefixes, "attributes"
    )
    return FormalContext(
        objects=left.objects,
        attributes=left_attributes + right_attributes,
        incidence=np.hstack([left.incidence, right.incidence[order]]),
    )


def subposition(
    top: FormalContext,
    bottom: FormalContext,
    prefixes: tuple[str, str] | None = None,
) -> FormalContext:
    """
    Stack two contexts with the same attributes.

    Attributes are matched by name; the result uses the attribute order of
    ``top``.
    """
    order = _same_names(top.attributes, bottom.attributes, "attributes")
    top_objects, bottom_objects = _disjoint_names(
        top.objects, bottom.objects, prefixes, "objects"
    )
    return FormalContext(
        objects=top_objects + bottom_objects,
        attributes=top.attributes,
        incidence=np.vstack([top.incidence, bottom.incidence[:, order]]),
    )


def direct_sum(
    first: FormalContext,
    second: FormalContext,
    prefixes: tuple[str, str] | None = None,
) -> FormalContext:
    """
    Direct sum: its concept lattice is the product of the two lattices.

    Objects and attributes are the disjoint unions; an object of one context
    additionally has every attribute of the other.
    """
    g1, g2 = _disjoint_names(first.objects, second.objects, prefixes, "objects")
    m1, m2 = _disjoint_names(first.attributes, second.attributes, prefixes, "attributes")
    incidence = np.block(
        [
            [first.incidence, np.ones((first.n_objects, second.n_attributes), dtype=bool)],
            [np.ones((second.n_objects, first.n_attributes), dtype=bool), second.incidence],
        ]
    )
    return FormalContext(objects=g1 + g2, attributes=m1 + m2, incidence=incidence)


def direct_product(
    first: FormalContext,
    second: FormalContext,
    separator: str = "|",
) -> FormalContext:
    """
    Direct product: objects ``g1|g2``, attributes ``m1|m2``, and
    ``(g1, g2)`` has ``(m1, m2)`` iff ``g1`` has ``m1`` or ``g2`` has ``m2``.
    """
    objects = tuple(f"{a}{separator}{b}" for a in first.objects for b in second.objects)
    attributes = tuple(
        f"{a}{separator}{b}" for a in first.attributes for b in second.attributes
    )
    for kind, names in (("object", objects), ("attribute", attributes)):
        if len(set(names)) != len(names):
            raise ValueError(
                f"Joining {kind} names with separator={separator!r} gives duplicates "
                "(a name already contains it). Pass another separator."
            )
    # Rows (g1, g2), columns (m1, m2): I1[g1, m1] or I2[g2, m2].
    incidence = (
        first.incidence[:, None, :, None] | second.incidence[None, :, None, :]
    ).reshape(len(objects), len(attributes))
    return FormalContext(objects=objects, attributes=attributes, incidence=incidence)


def complement(context: FormalContext) -> FormalContext:
    """The complementary context: crosses and non-crosses swapped."""
    return FormalContext(
        objects=context.objects,
        attributes=context.attributes,
        incidence=~context.incidence,
    )


def dual(context: FormalContext) -> FormalContext:
    """The dual context (same as ``context.transpose()``)."""
    return context.transpose()
