"""
Data-quality tools built on formal concept analysis.

- ``check_implications``: which objects break the rules you expect to hold.
- ``redundancy_report``: duplicate, constant and derivable objects and
  attributes.
- ``explain_object`` / ``explain_rule``: readable explanations of where an
  object sits in the lattice and of what supports or breaks a rule.

Rules may be given as ``Implication`` / ``AssociationRule`` objects or as
``(premise, conclusion)`` pairs of attribute names or indices.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from conceptflow.algorithms._bitset import BitsetContext
from conceptflow.core import Concept, ConceptLattice, FormalContext


def _rule_parts(context: FormalContext, rule) -> tuple[frozenset[int], frozenset[int]]:
    if hasattr(rule, "premise") and hasattr(rule, "conclusion"):
        premise, conclusion = rule.premise, rule.conclusion
    elif isinstance(rule, (set, frozenset)):
        raise TypeError("A rule must be an ordered (premise, conclusion) pair, not a set.")
    else:
        try:
            premise, conclusion = rule
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "A rule must be an Implication/AssociationRule or a "
                "(premise, conclusion) pair."
            ) from exc
        if isinstance(premise, str) or isinstance(conclusion, str):
            raise TypeError(
                "Give premise and conclusion as collections of attributes, "
                'e.g. (["a"], ["b"]), not as single strings.'
            )
    return (
        frozenset(context._resolve_attribute_indices(premise)),
        frozenset(context._resolve_attribute_indices(conclusion)),
    )


def _names(names: tuple[str, ...], indices) -> tuple[str, ...]:
    return tuple(names[i] for i in sorted(indices))


def _format_rule(context: FormalContext, premise, conclusion) -> str:
    left = ", ".join(_names(context.attributes, premise)) or "∅"
    right = ", ".join(_names(context.attributes, conclusion - premise))
    return f"{{{left}}} -> {{{right}}}"


def check_implications(context: FormalContext, rules) -> pd.DataFrame:
    """
    List every violation of the given rules.

    An object violates ``premise -> conclusion`` if it has all premise
    attributes but lacks some conclusion attribute.

    Returns
    -------
    pandas.DataFrame
        One row per (rule, object) violation, with columns ``rule``,
        ``premise``, ``conclusion``, ``object`` and ``missing`` (the
        conclusion attributes the object lacks). Empty if every rule holds.
    """
    rows = []

    for rule in rules:
        premise, conclusion = _rule_parts(context, rule)
        has_premise = context.attribute_derivation(premise)
        has_all = context.attribute_derivation(premise | conclusion)

        for g in sorted(has_premise - has_all):
            missing = {m for m in conclusion if not context.incidence[g, m]}
            rows.append(
                {
                    "rule": _format_rule(context, premise, conclusion),
                    "premise": _names(context.attributes, premise),
                    "conclusion": _names(context.attributes, conclusion - premise),
                    "object": context.objects[g],
                    "missing": _names(context.attributes, missing),
                }
            )

    return pd.DataFrame(rows, columns=["rule", "premise", "conclusion", "object", "missing"])


@dataclass(frozen=True)
class RedundancyReport:
    """
    Redundant structure in a formal context.

    Attributes
    ----------
    duplicate_objects, duplicate_attributes:
        Groups (of size >= 2) of objects with identical rows / attributes
        with identical columns.

    universal_attributes, empty_attributes:
        Attributes every object has / no object has.

    full_objects, empty_objects:
        Objects having every attribute / no attribute.

    derivable_attributes:
        Attribute -> other attributes whose extents intersect to exactly its
        extent: the column can be computed from them (it is reducible).
        Duplicates and universal attributes are reported above instead.

    derivable_objects:
        The same for objects: the row is the intersection of other rows.
    """

    duplicate_objects: list[tuple[str, ...]] = field(default_factory=list)
    duplicate_attributes: list[tuple[str, ...]] = field(default_factory=list)
    universal_attributes: tuple[str, ...] = ()
    empty_attributes: tuple[str, ...] = ()
    full_objects: tuple[str, ...] = ()
    empty_objects: tuple[str, ...] = ()
    derivable_attributes: dict[str, tuple[str, ...]] = field(default_factory=dict)
    derivable_objects: dict[str, tuple[str, ...]] = field(default_factory=dict)

    @property
    def is_clean(self) -> bool:
        """Whether nothing was found."""
        return not any(
            (
                self.duplicate_objects,
                self.duplicate_attributes,
                self.universal_attributes,
                self.empty_attributes,
                self.full_objects,
                self.empty_objects,
                self.derivable_attributes,
                self.derivable_objects,
            )
        )

    def to_frame(self) -> pd.DataFrame:
        """All findings as a table with columns ``kind``, ``name``, ``detail``."""
        rows = []
        for group in self.duplicate_objects:
            rows.append(("duplicate objects", group[0], ", ".join(group[1:])))
        for group in self.duplicate_attributes:
            rows.append(("duplicate attributes", group[0], ", ".join(group[1:])))
        rows += [
            ("universal attribute", n, "every object has it") for n in self.universal_attributes
        ]
        rows += [("empty attribute", n, "no object has it") for n in self.empty_attributes]
        rows += [("full object", n, "has every attribute") for n in self.full_objects]
        rows += [("empty object", n, "has no attribute") for n in self.empty_objects]
        rows += [
            ("derivable attribute", n, " AND ".join(parts))
            for n, parts in self.derivable_attributes.items()
        ]
        rows += [
            ("derivable object", n, " AND ".join(parts))
            for n, parts in self.derivable_objects.items()
        ]
        return pd.DataFrame(rows, columns=["kind", "name", "detail"])

    def __str__(self) -> str:
        if self.is_clean:
            return "RedundancyReport: nothing redundant found."
        return "RedundancyReport\n" + self.to_frame().to_string(index=False)


def _duplicate_groups(vectors: list[int], names: tuple[str, ...]) -> list[tuple[str, ...]]:
    groups: dict[int, list[str]] = {}
    for vector, name in zip(vectors, names):
        groups.setdefault(vector, []).append(name)
    return [tuple(group) for group in groups.values() if len(group) > 1]


def _derivable(
    vectors: list[int], names: tuple[str, ...], full: int
) -> dict[str, tuple[str, ...]]:
    """
    Vectors (bitsets) equal to the intersection of strictly larger ones,
    with a small generating subset found greedily.
    """
    result = {}
    seen: set[int] = set()

    for i, vector in enumerate(vectors):
        if vector in (0, full) or vector in seen:
            # Empty / universal / duplicate: reported elsewhere.
            seen.add(vector)
            continue
        seen.add(vector)

        larger = [
            j for j, other in enumerate(vectors)
            if other != vector and other & vector == vector
        ]
        if not larger:
            continue

        meet = full
        for j in larger:
            meet &= vectors[j]
        if meet != vector:
            continue

        # Greedy: drop parts while the intersection stays the same, starting
        # with the largest (least informative) sets.
        parts = sorted(larger, key=lambda j: -vectors[j].bit_count())
        k = 0
        while k < len(parts):
            trial = parts[:k] + parts[k + 1 :]
            trial_meet = full
            for j in trial:
                trial_meet &= vectors[j]
            if trial and trial_meet == vector:
                parts = trial
            else:
                k += 1
        result[names[i]] = tuple(names[j] for j in sorted(parts))

    return result


def redundancy_report(context: FormalContext) -> RedundancyReport:
    """
    Find duplicate, constant and derivable objects and attributes.

    "Derivable" means reducible in FCA terms: an attribute whose extent is
    the intersection of other attributes' extents adds no information to the
    concept lattice (likewise for object rows). Useful for cleaning features
    before analysis or machine learning.
    """
    bits = BitsetContext(context)

    return RedundancyReport(
        duplicate_objects=_duplicate_groups(bits.rows, context.objects),
        duplicate_attributes=_duplicate_groups(bits.columns, context.attributes),
        universal_attributes=tuple(
            n for n, c in zip(context.attributes, bits.columns) if c == bits.all_objects
        ),
        empty_attributes=tuple(n for n, c in zip(context.attributes, bits.columns) if c == 0),
        full_objects=tuple(
            n for n, r in zip(context.objects, bits.rows) if r == bits.all_attributes
        ),
        empty_objects=tuple(n for n, r in zip(context.objects, bits.rows) if r == 0),
        derivable_attributes=_derivable(bits.columns, context.attributes, bits.all_objects),
        derivable_objects=_derivable(bits.rows, context.objects, bits.all_attributes),
    )


@dataclass(frozen=True)
class ObjectExplanation:
    """Where an object sits in the concept lattice."""

    object: str
    attributes: tuple[str, ...]
    object_concept: Concept
    n_concepts_containing: int
    indistinguishable_from: tuple[str, ...]
    beyond_more_general: list[tuple[tuple[str, ...], tuple[str, ...]]]

    def __str__(self) -> str:
        lines = [
            f"Object {self.object!r} has: {', '.join(self.attributes) or 'no attributes'}.",
            f"It belongs to {self.n_concepts_containing} concepts.",
        ]
        if self.indistinguishable_from:
            lines.append(
                "Same attributes as: " + ", ".join(self.indistinguishable_from) + "."
            )
        for general, extra in self.beyond_more_general:
            lines.append(
                f"Compared with the concept {{{', '.join(general) or '∅'}}}, it also has: "
                f"{', '.join(extra)}."
            )
        return "\n".join(lines)


def explain_object(lattice: ConceptLattice, obj: int | str) -> ObjectExplanation:
    """
    Explain an object through its object concept γ(g).

    Reports the object's attributes, how many concepts contain it, which
    objects have exactly the same attributes, and for each more general
    neighbouring concept, what the object has beyond it.
    """
    ctx = lattice.context
    index = ctx._resolve_object_indices([obj])[0]
    concept = lattice.object_concept(index)

    same = tuple(
        ctx.objects[h]
        for h in sorted(concept.extent)
        if h != index and np.array_equal(ctx.incidence[h], ctx.incidence[index])
    )
    beyond = [
        (
            _names(ctx.attributes, parent.intent),
            _names(ctx.attributes, concept.intent - parent.intent),
        )
        for parent in sorted(lattice.parents(concept), key=lambda c: sorted(c.intent))
    ]

    return ObjectExplanation(
        object=ctx.objects[index],
        attributes=_names(ctx.attributes, concept.intent),
        object_concept=concept,
        n_concepts_containing=len(lattice.upset(concept)),
        indistinguishable_from=same,
        beyond_more_general=beyond,
    )


@dataclass(frozen=True)
class RuleExplanation:
    """Evidence for and against a rule premise -> conclusion."""

    premise: tuple[str, ...]
    conclusion: tuple[str, ...]
    support: int
    premise_support: int
    supporting_objects: tuple[str, ...]
    exceptions: dict[str, tuple[str, ...]]

    @property
    def confidence(self) -> float:
        return self.support / self.premise_support if self.premise_support else float("nan")

    def __str__(self) -> str:
        rule = f"{{{', '.join(self.premise) or '∅'}}} -> {{{', '.join(self.conclusion)}}}"
        lines = [
            f"{rule}: holds for {self.support} of {self.premise_support} objects "
            f"with the premise (confidence {self.confidence:.2f})."
        ]
        for name, missing in self.exceptions.items():
            lines.append(f"  exception {name!r} lacks: {', '.join(missing)}")
        return "\n".join(lines)


def explain_rule(context: FormalContext, rule) -> RuleExplanation:
    """Which objects support a rule and which break it (and how)."""
    premise, conclusion = _rule_parts(context, rule)
    has_premise = context.attribute_derivation(premise)
    has_all = context.attribute_derivation(premise | conclusion)

    exceptions = {
        context.objects[g]: _names(
            context.attributes, {m for m in conclusion if not context.incidence[g, m]}
        )
        for g in sorted(has_premise - has_all)
    }

    return RuleExplanation(
        premise=_names(context.attributes, premise),
        conclusion=_names(context.attributes, conclusion - premise),
        support=len(has_all),
        premise_support=len(has_premise),
        supporting_objects=_names(context.objects, has_all),
        exceptions=exceptions,
    )


__all__ = [
    "check_implications",
    "redundancy_report",
    "RedundancyReport",
    "explain_object",
    "ObjectExplanation",
    "explain_rule",
    "RuleExplanation",
]
