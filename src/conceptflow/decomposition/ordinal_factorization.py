"""
Ordinal factorization tools for formal contexts.

This module contains infrastructure for ordinal two-factorizations.

Implemented
-----------
- incidence-pair representation
- Ferrers relation validation
- complement context construction
- incompatibility graph construction
- bipartite graph coloring
- an exact small-scale ordinal two-factorizer based on Algorithm 1 from
  Dürrschnabel and Stumme's Ord2Factor paper

Not yet implemented
-------------------
The full Ord2Factor algorithm for maximal ordinal two-factorizations is not
implemented yet. That algorithm additionally requires selecting large induced
bipartite subgraphs of the incompatibility graph.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import combinations

import numpy as np
from sklearn.base import BaseEstimator

from conceptflow.core import ConceptLattice, FormalContext
from conceptflow.validation import check_binary_context_input


IncidencePair = tuple[int, int]


@dataclass(frozen=True)
class OrdinalFactor:
    """
    Ordinal factor represented as a Ferrers relation.
    """

    relation: frozenset[IncidencePair]

    def __post_init__(self) -> None:
        object.__setattr__(self, "relation", frozenset(self.relation))

    def __len__(self) -> int:
        return len(self.relation)

    def covers(self, pair: IncidencePair) -> bool:
        return pair in self.relation


@dataclass(frozen=True)
class OrdinalTwoFactorization:
    """
    Pair of ordinal factors.
    """

    factor_1: OrdinalFactor
    factor_2: OrdinalFactor
    covered_incidence: frozenset[IncidencePair]

    @property
    def n_covered(self) -> int:
        return len(self.covered_incidence)


def incidence_pairs(context: FormalContext) -> frozenset[IncidencePair]:
    """
    Return all incidence pairs of a formal context.
    """
    pairs: set[IncidencePair] = set()

    for object_index in range(context.n_objects):
        for attribute_index in range(context.n_attributes):
            if context.incidence[object_index, attribute_index]:
                pairs.add((object_index, attribute_index))

    return frozenset(pairs)


def complement_context(context: FormalContext) -> FormalContext:
    """
    Return the complement context with the same objects and attributes.
    """
    return FormalContext(
        objects=context.objects,
        attributes=context.attributes,
        incidence=np.logical_not(context.incidence),
    )


def is_ferrers_relation(
    relation: Iterable[IncidencePair],
    context: FormalContext,
) -> bool:
    """
    Check whether a relation is Ferrers.

    A relation F is Ferrers if for all (g, m), (h, n) in F,
    either (g, n) is in F or (h, m) is in F.
    """
    relation = frozenset(relation)

    for g, m in relation:
        for h, n in relation:
            if (g, n) not in relation and (h, m) not in relation:
                return False

    return True


def incompatibility_graph(
    context: FormalContext,
) -> dict[IncidencePair, set[IncidencePair]]:
    """
    Build the incompatibility graph of a formal context.

    Vertices are incidence pairs. Two vertices (g, m) and (h, n) are adjacent
    if both cross-incidences (g, n) and (h, m) are missing.
    """
    pairs = sorted(incidence_pairs(context))
    graph: dict[IncidencePair, set[IncidencePair]] = {
        pair: set()
        for pair in pairs
    }

    for i, pair_1 in enumerate(pairs):
        g, m = pair_1

        for pair_2 in pairs[i + 1:]:
            h, n = pair_2

            if (
                not context.incidence[g, n]
                and not context.incidence[h, m]
            ):
                graph[pair_1].add(pair_2)
                graph[pair_2].add(pair_1)

    return graph


def bipartite_coloring(
    graph: dict[IncidencePair, set[IncidencePair]],
) -> dict[IncidencePair, int]:
    """
    Compute a bipartite coloring of a graph.

    Raises
    ------
    ValueError
        If the graph is not bipartite.
    """
    color: dict[IncidencePair, int] = {}

    for start in graph:
        if start in color:
            continue

        color[start] = 0
        queue = deque([start])

        while queue:
            node = queue.popleft()

            for neighbor in graph[node]:
                if neighbor not in color:
                    color[neighbor] = 1 - color[node]
                    queue.append(neighbor)
                elif color[neighbor] == color[node]:
                    raise ValueError(
                        "Incompatibility graph is not bipartite."
                    )

    return color


def _concept_leq(
    lattice: ConceptLattice,
    left_index: int,
    right_index: int,
) -> bool:
    """
    Concept order using extent inclusion.
    """
    left = lattice.concepts[left_index]
    right = lattice.concepts[right_index]
    return left.extent.issubset(right.extent)


def _poset_predecessors(
    lattice: ConceptLattice,
) -> dict[int, set[int]]:
    """
    Return predecessor sets for the concept poset.
    """
    n_concepts = lattice.n_concepts
    predecessors = {i: set() for i in range(n_concepts)}

    for i, j in combinations(range(n_concepts), 2):
        if _concept_leq(lattice, i, j):
            predecessors[j].add(i)
        elif _concept_leq(lattice, j, i):
            predecessors[i].add(j)

    return predecessors


def _linear_extensions(
    predecessors: dict[int, set[int]],
) -> list[tuple[int, ...]]:
    """
    Enumerate all linear extensions of a finite poset.

    This is intentionally simple and suitable only for small contexts.
    """
    elements = set(predecessors)
    result: list[tuple[int, ...]] = []

    def backtrack(
        current: list[int],
        remaining: set[int],
    ) -> None:
        if not remaining:
            result.append(tuple(current))
            return

        available = sorted(
            element
            for element in remaining
            if predecessors[element].issubset(current)
        )

        for element in available:
            current.append(element)
            remaining.remove(element)

            backtrack(current, remaining)

            remaining.add(element)
            current.pop()

    backtrack([], set(elements))
    return result


def _realizer_pair(
    lattice: ConceptLattice,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """
    Find two linear extensions whose intersection is the concept order.

    For small contexts, this brute-force search is a clear way to obtain the
    two chain orders required by Algorithm 1 when the lattice has order
    dimension at most two.
    """
    predecessors = _poset_predecessors(lattice)
    extensions = _linear_extensions(predecessors)

    if not extensions:
        raise ValueError("Could not compute linear extensions.")

    n_concepts = lattice.n_concepts

    poset_relation = {
        (i, j)
        for i in range(n_concepts)
        for j in range(n_concepts)
        if _concept_leq(lattice, i, j)
    }

    extension_positions = []
    for extension in extensions:
        extension_positions.append(
            {
                concept_index: position
                for position, concept_index in enumerate(extension)
            }
        )

    for ext_1, pos_1 in zip(extensions, extension_positions):
        for ext_2, pos_2 in zip(extensions, extension_positions):
            intersection_relation = {
                (i, j)
                for i in range(n_concepts)
                for j in range(n_concepts)
                if pos_1[i] <= pos_1[j] and pos_2[i] <= pos_2[j]
            }

            if intersection_relation == poset_relation:
                return ext_1, ext_2

    raise ValueError(
        "The complement concept lattice does not appear to have "
        "order dimension at most two."
    )


def _factor_from_chain_order(
    context: FormalContext,
    complement_lattice: ConceptLattice,
    chain_order: tuple[int, ...],
) -> OrdinalFactor:
    """
    Construct one factor from a chain order as in Algorithm 1.

    For a chain order L, we scan concepts in chain order, accumulate extents,
    and collect accumulated_extent x intent. The factor is the complement of
    this collected relation in G x M.
    """
    accumulated_extent: set[int] = set()
    excluded_pairs: set[IncidencePair] = set()

    for concept_index in chain_order:
        concept = complement_lattice.concepts[concept_index]
        accumulated_extent.update(concept.extent)

        for object_index in accumulated_extent:
            for attribute_index in concept.intent:
                excluded_pairs.add((object_index, attribute_index))

    all_pairs = {
        (object_index, attribute_index)
        for object_index in range(context.n_objects)
        for attribute_index in range(context.n_attributes)
    }

    return OrdinalFactor(frozenset(all_pairs - excluded_pairs))


class ExactOrdinalTwoFactorizer(BaseEstimator):
    """
    Exact ordinal two-factorization for already two-factorizable contexts.

    This implements a small-scale version of the construction from Algorithm 1:
    it uses the complement context, computes its concept lattice, finds two
    linear extensions whose intersection is the lattice order, and constructs
    the two Ferrers factors from those chain orders.

    This is not the full maximal Ord2Factor algorithm.
    """

    def __init__(self, lattice_algorithm: str = "nextclosure"):
        self.lattice_algorithm = lattice_algorithm

    def fit(self, X, y=None):
        context = check_binary_context_input(X)

        graph = incompatibility_graph(context)
        coloring = bipartite_coloring(graph)

        complement = complement_context(context)
        complement_lattice = ConceptLattice.from_context(
            complement,
            algorithm=self.lattice_algorithm,
        )

        chain_1, chain_2 = _realizer_pair(complement_lattice)

        factor_1 = _factor_from_chain_order(
            context=context,
            complement_lattice=complement_lattice,
            chain_order=chain_1,
        )
        factor_2 = _factor_from_chain_order(
            context=context,
            complement_lattice=complement_lattice,
            chain_order=chain_2,
        )

        all_incidence = incidence_pairs(context)
        covered = factor_1.relation | factor_2.relation

        if not factor_1.relation.issubset(all_incidence):
            raise ValueError("Computed factor_1 contains non-incidence pairs.")

        if not factor_2.relation.issubset(all_incidence):
            raise ValueError("Computed factor_2 contains non-incidence pairs.")

        if not is_ferrers_relation(factor_1.relation, context):
            raise ValueError("Computed factor_1 is not a Ferrers relation.")

        if not is_ferrers_relation(factor_2.relation, context):
            raise ValueError("Computed factor_2 is not a Ferrers relation.")

        if covered != all_incidence:
            raise ValueError(
                "Computed factors do not cover the full incidence relation."
            )

        self.context_ = context
        self.incompatibility_graph_ = graph
        self.coloring_ = coloring
        self.complement_context_ = complement
        self.complement_lattice_ = complement_lattice
        self.chain_orders_ = (chain_1, chain_2)
        self.factorization_ = OrdinalTwoFactorization(
            factor_1=factor_1,
            factor_2=factor_2,
            covered_incidence=frozenset(covered),
        )
        self.factors_ = (factor_1, factor_2)
        self.covered_incidence_ = frozenset(covered)
        self.coverage_ = len(covered) / len(all_incidence) if all_incidence else 1.0

        return self


class Ord2Factor(BaseEstimator):
    """
    Placeholder for the full maximal Ord2Factor algorithm.

    The full algorithm requires repeatedly selecting large induced bipartite
    subgraphs of the incompatibility graph before applying exact ordinal
    two-factorization. This is intentionally not implemented yet.
    """

    def __init__(self, strategy: str = "maximal_bipartite"):
        self.strategy = strategy

    def fit(self, X, y=None):
        raise NotImplementedError(
            "Full Ord2Factor is not implemented yet. "
            "Use ExactOrdinalTwoFactorizer for contexts that already admit "
            "an exact ordinal two-factorization."
        )