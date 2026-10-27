"""
Realized and unrealized combinations of a nested line diagram.

A nested diagram draws the concept lattice of the apposition
``K = K_outer | K_inner`` inside the direct product of the two factor
lattices. Every pair ``(c, d)`` of an outer concept ``c`` and an inner
concept ``d`` is a point of that product, but only some of them are
*realized*, i.e. are the image of a concept of ``K``. Unrealized pairs are
the hollow inner nodes of the diagram; they show which attribute
combinations do not occur in the data.

A pair ``(c, d)`` is realized iff ``B_c ∪ B_d`` is an intent of ``K``,
equivalently iff ``E = A_c ∩ A_d`` derives back to exactly ``B_c`` in the
outer context and exactly ``B_d`` in the inner context.
"""

from __future__ import annotations

import pandas as pd

from conceptflow.core import Concept, ConceptLattice, FormalContext
from conceptflow.exploration.view import ExplorationView


def _names(names: tuple[str, ...], indices: frozenset[int]) -> tuple[str, ...]:
    return tuple(names[i] for i in sorted(indices))


def realized_pairs(
    outer_lattice: ConceptLattice,
    inner_lattice: ConceptLattice,
) -> set[tuple[Concept, Concept]]:
    """
    Return the realized ``(outer, inner)`` concept pairs.

    Both lattices must be built over the same objects in the same order and
    over disjoint attribute sets.
    """
    _check_compatible(outer_lattice, inner_lattice)

    outer_context = outer_lattice.context
    inner_context = inner_lattice.context
    realized: set[tuple[Concept, Concept]] = set()

    for outer in outer_lattice.concepts:
        for inner in inner_lattice.concepts:
            common = outer.extent & inner.extent
            if (
                outer_context.object_derivation(common) == outer.intent
                and inner_context.object_derivation(common) == inner.intent
            ):
                realized.add((outer, inner))

    return realized


def _check_compatible(outer_lattice: ConceptLattice, inner_lattice: ConceptLattice) -> None:
    if outer_lattice.context.objects != inner_lattice.context.objects:
        raise ValueError(
            "Outer and inner contexts must have the same objects in the same order."
        )

    overlap = set(outer_lattice.context.attributes) & set(inner_lattice.context.attributes)
    if overlap:
        raise ValueError(
            "Outer and inner attribute sets must be disjoint for the nested "
            "diagram (apposition K = K_outer | K_inner), but these appear in "
            f"both: {sorted(overlap)}"
        )


def nested_combinations(
    view: ExplorationView,
    inner: ExplorationView | None = None,
) -> pd.DataFrame:
    """
    Tabulate every (outer concept, inner concept) pair of a nested diagram.

    Parameters
    ----------
    view:
        The outer exploration view.

    inner:
        The inner view. Defaults to the child view of ``view``'s top concept,
        which is the inner template drawn inside every outer node by
        ``plot_nested``.

    Returns
    -------
    pandas.DataFrame
        One row per pair, with columns:

        - ``outer_id``, ``inner_id``: concept stable ids, as used by
          ``exploration_view_to_nested_data`` and the HTML diagram.
        - ``outer_intent``, ``inner_intent``: attribute names.
        - ``realized``: whether the pair is a concept of the combined
          context (a filled inner node).
        - ``n_objects``, ``objects``: objects having every attribute of both
          intents.
        - ``introduced_objects``: objects whose object concept is exactly
          this pair (the object labels the diagram would place here).
    """
    if inner is None:
        top_id = view.lattice.top().stable_id()
        if top_id not in view.children:
            raise ValueError(
                "view has no inner view: expand its top concept first, or "
                "pass `inner` explicitly."
            )
        inner = view.children[top_id]

    outer_lattice = view.lattice
    inner_lattice = inner.lattice
    realized = realized_pairs(outer_lattice, inner_lattice)

    objects = outer_lattice.context.objects
    outer_attributes = outer_lattice.context.attributes
    inner_attributes = inner_lattice.context.attributes

    outer_by_intent = {c.intent: c for c in outer_lattice.concepts}
    inner_by_intent = {c.intent: c for c in inner_lattice.concepts}

    introduced: dict[tuple[Concept, Concept], list[str]] = {}
    for g, name in enumerate(objects):
        pair = (
            outer_by_intent[outer_lattice.context.object_derivation([g])],
            inner_by_intent[inner_lattice.context.object_derivation([g])],
        )
        introduced.setdefault(pair, []).append(name)

    rows = []
    for outer in outer_lattice.concepts:
        for inner_concept in inner_lattice.concepts:
            pair = (outer, inner_concept)
            common = outer.extent & inner_concept.extent
            rows.append(
                {
                    "outer_id": outer.stable_id(),
                    "inner_id": inner_concept.stable_id(),
                    "outer_intent": _names(outer_attributes, outer.intent),
                    "inner_intent": _names(inner_attributes, inner_concept.intent),
                    "realized": pair in realized,
                    "n_objects": len(common),
                    "objects": _names(objects, common),
                    "introduced_objects": tuple(introduced.get(pair, ())),
                }
            )

    return pd.DataFrame(rows)


def nested_view(
    outer: FormalContext,
    inner: FormalContext,
    outer_name: str = "outer",
    inner_name: str = "inner",
    algorithm: str = "nextclosure",
) -> ExplorationView:
    """
    Build a two-level ExplorationView from two contexts over the same objects.

    The outer lattice is built from ``outer``; its top concept gets ``inner``
    (over all objects) as its child view. That is exactly what a two-level
    nested diagram draws -- every outer concept shows the same inner
    lattice, filled where combinations occur -- so no per-concept inner
    lattices are built. Use ExplorationBuilder for deeper nesting.

    Objects are matched by name; attribute names must not overlap.
    """
    if set(outer.objects) != set(inner.objects) or outer.n_objects != inner.n_objects:
        raise ValueError("outer and inner must have the same objects.")

    overlap = set(outer.attributes) & set(inner.attributes)
    if overlap:
        raise ValueError(
            "outer and inner attributes must be disjoint, but these appear in "
            f"both: {sorted(overlap)}. Rename them, e.g. with context.rename()."
        )

    inner = inner.subcontext(objects=list(outer.objects))
    root = ExplorationView(
        name=outer_name,
        context=outer,
        lattice=ConceptLattice.from_context(outer, algorithm=algorithm),
        depth=0,
        scale_names=(outer_name,),
    )

    top = root.lattice.top()
    root.add_child(
        top.stable_id(),
        ExplorationView(
            name=inner_name,
            context=inner,
            lattice=ConceptLattice.from_context(inner, algorithm=algorithm),
            depth=1,
            scale_names=(inner_name,),
            parent_concept_id=top.stable_id(),
        ),
    )

    return root
