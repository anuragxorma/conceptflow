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
- ``ExactOrdinalTwoFactorizer``: the two Ferrers factors of a
  two-factorizable context (Algorithm 1 of Dürrschnabel and Stumme,
  "Maximal Ordinal Two-Factorizations", ICCS 2023)
- ``Ord2Factor``: large ordinal two-factorizations of any context
  (Algorithm 2 of the same paper), with the bipartite-subgraph methods of
  ``conceptflow.decomposition.bipartite``
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass

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


def _concept_order(lattice: ConceptLattice) -> np.ndarray:
    """``leq[i, j]``: concept i lies below (or is) concept j, by extent inclusion."""
    extents = np.zeros((lattice.n_concepts, lattice.context.n_objects), dtype=int)
    for i, concept in enumerate(lattice.concepts):
        extents[i, list(concept.extent)] = 1
    # i <= j iff no object is in extent i but not in extent j.
    return extents @ (1 - extents).T == 0


def _transitive_orientation(adjacent: list[set[int]]) -> set[tuple[int, int]]:
    """
    A transitive orientation of an undirected graph (Golumbic's algorithm).

    Repeatedly orients one remaining edge and everything it forces (its
    implication class: ab forces ac if bc is not an edge, and cb if ac is
    not an edge), then removes those edges. Raises ValueError if the graph
    has no transitive orientation.
    """
    remaining = [set(neighbours) for neighbours in adjacent]
    orientation: set[tuple[int, int]] = set()

    for a in range(len(remaining)):
        while remaining[a]:
            start = (a, next(iter(remaining[a])))
            implication_class = {start}
            stack = [start]
            while stack:
                x, y = stack.pop()
                forced = [(x, z) for z in remaining[x] if z != y and z not in remaining[y]]
                forced += [(z, y) for z in remaining[y] if z != x and z not in remaining[x]]
                for edge in forced:
                    if edge not in implication_class:
                        implication_class.add(edge)
                        stack.append(edge)

            if any((y, x) in implication_class for x, y in implication_class):
                raise ValueError(
                    "The complement concept lattice does not have order "
                    "dimension at most two."
                )
            orientation |= implication_class
            for x, y in implication_class:
                remaining[x].discard(y)
                remaining[y].discard(x)

    return orientation


def _realizer_pair(
    lattice: ConceptLattice,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """
    Find two linear extensions whose intersection is the concept order.

    A poset has order dimension at most two iff its incomparability graph
    has a transitive orientation T (Dushnik and Miller); the two linear
    extensions are then P + T and P + T^-1. Polynomial, unlike trying all
    pairs of linear extensions.
    """
    leq = _concept_order(lattice)
    n_concepts = lattice.n_concepts
    incomparable = ~(leq | leq.T)
    adjacent = [set(np.flatnonzero(row).tolist()) for row in incomparable]
    orientation = _transitive_orientation(adjacent)

    chains = []
    for forward in (True, False):
        before = leq.copy()
        for x, y in orientation:
            if forward:
                before[x, y] = True
            else:
                before[y, x] = True
        # In a linear order, the number of elements before x is x's position.
        positions = before.sum(axis=0) - 1
        if sorted(positions.tolist()) != list(range(n_concepts)):
            raise ValueError(
                "The complement concept lattice does not have order "
                "dimension at most two."
            )
        chains.append(tuple(int(i) for i in np.argsort(positions)))

    position_1 = np.argsort(chains[0])
    position_2 = np.argsort(chains[1])
    both = (position_1[:, None] <= position_1[None, :]) & (
        position_2[:, None] <= position_2[None, :]
    )
    if not np.array_equal(both, leq):
        raise ValueError(
            "The complement concept lattice does not have order dimension at most two."
        )

    return chains[0], chains[1]


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
        self.n_features_in_ = context.n_attributes
        self.feature_names_in_ = context.attributes
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

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.sparse = True
        return tags


class Ord2Factor(BaseEstimator):
    """
    Large ordinal two-factorizations of any context (Ord2Factor).

    Algorithm 2 of Dürrschnabel & Stumme, "Maximal Ordinal
    Two-Factorizations" (ICCS 2023): while the incompatibility graph of the
    context is not bipartite, keep only the crosses of a large induced
    bipartite subgraph of it (dropping the others) and rebuild the graph --
    dropping crosses can create new incompatibilities, so this may repeat.
    The remaining context is two-factorizable, and its two Ferrers factors
    are computed as by ``ExactOrdinalTwoFactorizer`` (Algorithm 1).

    Computing a maximal two-factorization is NP-hard, so the bipartite step
    offers the methods of Dürrschnabel, Hanika & Stumme (2022):

    Parameters
    ----------
    method:
        ``"exact"`` (default): a smallest set of crosses to drop, from the
        papers' SAT formulation solved with MiniSat 2.2 (exponential in the
        worst case; meant for small contexts). ``"greedy"``: an
        locally minimal set (Algorithm 1 of the 2022 paper), in polynomial
        time. ``"annealing"``: the
        greedy result improved by simulated annealing.

    n_iterations, temperature:
        Steps and starting temperature of ``method="annealing"``.

    random_state:
        Seed for the heuristics' random choices.

    lattice_algorithm:
        Concept enumeration algorithm for the complement lattice.

    Attributes
    ----------
    factors_:
        The two ordinal factors (Ferrers relations, sets of incidence pairs).

    removed_incidence_:
        The crosses that are not covered (dropped to make the context
        two-factorizable).

    covered_incidence_, coverage_:
        The covered crosses and their share of all crosses.

    n_rounds_:
        How often crosses were dropped (0 if the context was already
        two-factorizable).

    reduced_context_:
        The two-factorizable context that was factorized.
    """

    def __init__(
        self,
        method: str = "exact",
        n_iterations: int = 1000,
        temperature: float = 1.0,
        random_state=None,
        lattice_algorithm: str = "nextclosure",
    ):
        self.method = method
        self.n_iterations = n_iterations
        self.temperature = temperature
        self.random_state = random_state
        self.lattice_algorithm = lattice_algorithm

    def _deletion(self, graph, rng):
        from conceptflow.decomposition.bipartite import (
            annealing_bipartite_deletion,
            greedy_bipartite_deletion,
            minimum_bipartite_deletion,
        )

        if self.method == "exact":
            return minimum_bipartite_deletion(graph)
        if self.method == "greedy":
            return greedy_bipartite_deletion(graph, random_state=rng)
        return annealing_bipartite_deletion(
            graph, self.n_iterations, self.temperature, random_state=rng
        )

    def fit(self, X, y=None):
        """Find two ordinal factors covering a large part of X's crosses."""
        from sklearn.utils import check_random_state

        from conceptflow.decomposition.bipartite import is_bipartite

        if self.method not in {"exact", "greedy", "annealing"}:
            raise ValueError('method must be "exact", "greedy" or "annealing".')
        if self.method == "annealing" and (
            isinstance(self.n_iterations, bool)
            or not isinstance(self.n_iterations, (int, np.integer))
            or self.n_iterations < 0
        ):
            raise ValueError("n_iterations must be a non-negative integer.")

        context = check_binary_context_input(X)
        rng = check_random_state(self.random_state)
        incidence = np.array(context.incidence, dtype=bool)
        removed: set[IncidencePair] = set()
        rounds = 0

        while True:
            current = FormalContext(context.objects, context.attributes, incidence)
            graph = incompatibility_graph(current)
            if is_bipartite(graph):
                break
            dropped = self._deletion(graph, rng)
            for g, m in dropped:
                incidence[g, m] = False
            removed |= dropped
            rounds += 1

        exact = ExactOrdinalTwoFactorizer(lattice_algorithm=self.lattice_algorithm).fit(current)

        self.context_ = context
        self.n_features_in_ = context.n_attributes
        self.feature_names_in_ = context.attributes
        self.reduced_context_ = current
        self.removed_incidence_ = frozenset(removed)
        self.n_rounds_ = rounds
        self.factorization_ = exact.factorization_
        self.factors_ = exact.factors_
        self.covered_incidence_ = exact.covered_incidence_
        all_incidence = incidence_pairs(context)
        self.coverage_ = (
            len(self.covered_incidence_) / len(all_incidence) if all_incidence else 1.0
        )
        return self

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.sparse = True
        return tags
