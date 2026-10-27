"""
Pure concept lattice data structure.

This module defines the internal mathematical representation of a concept
lattice. It is not a scikit-learn estimator.

The sklearn-facing estimator will live in:

    conceptflow.cluster

and will use this class internally.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from functools import cached_property

from conceptflow.algorithms.hasse import compute_hasse_edges
from conceptflow.core.concept import Concept
from conceptflow.core.context import FormalContext


@dataclass(frozen=True)
class ConceptLattice:
    """
    Pure concept lattice representation.

    Parameters
    ----------
    context:
        Source formal context.

    concepts:
        Formal concepts of the context.

    min_support:
        0 for the full lattice. A positive number means ``concepts`` are the
        "iceberg": exactly the concepts with at least that many objects
        (see :meth:`from_context`).
    """

    context: FormalContext
    concepts: tuple[Concept, ...]
    min_support: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "concepts", tuple(self.concepts))

    @classmethod
    def from_context(
        cls,
        context: FormalContext,
        algorithm: str = "nextclosure",
        min_support: int | float | None = None,
        max_concepts: int | None = None,
    ) -> "ConceptLattice":
        """
        Build a concept lattice from a formal context.

        Parameters
        ----------
        context:
            The formal context.

        algorithm:
            Enumeration algorithm: "nextclosure", "closebyone" or
            "bruteforce".

        min_support:
            Build only the iceberg lattice: concepts whose extent has at
            least this many objects (int) or this fraction of all objects
            (float). The top concept is always included. Navigation that
            would leave the iceberg (e.g. a meet below it) raises an error.

        max_concepts:
            Stop with ``ConceptLimitExceeded`` past this many concepts. By
            default there is no limit; past 50,000 concepts a
            ``LargeLatticeWarning`` names these options and computation
            continues.
        """
        from conceptflow.algorithms.enumeration import (
            _resolve_min_support,
            enumerate_concepts,
        )

        concepts = enumerate_concepts(
            context,
            method=algorithm,
            min_support=min_support,
            max_concepts=max_concepts,
        )
        threshold = _resolve_min_support(min_support, context.n_objects)

        # Only an iceberg if something was cut off: every concept has at
        # least as many objects as the bottom concept (the objects having all
        # attributes), so a threshold up to that size keeps the full lattice.
        bottom_size = len(context.attribute_derivation(range(context.n_attributes)))
        only_top_is_bottom = len(concepts) == 1 and len(concepts[0].extent) == bottom_size
        if threshold <= bottom_size or only_top_is_bottom:
            threshold = 0

        return cls(context=context, concepts=tuple(concepts), min_support=threshold)

    @property
    def is_iceberg(self) -> bool:
        """Whether this holds only the frequent concepts (``min_support > 0``)."""
        return self.min_support > 0

    @property
    def n_concepts(self) -> int:
        """Number of concepts in the lattice."""
        return len(self.concepts)

    # Derived structure is computed once and cached on the instance.
    # cached_property writes to the instance __dict__ directly, so it works on
    # a frozen dataclass.

    @cached_property
    def edges(self) -> tuple[tuple[Concept, Concept], ...]:
        """
        Hasse diagram edges as ``(lower, upper)`` pairs.
        """
        return compute_hasse_edges(
            self.concepts, context=self.context, upset=self.is_iceberg
        )

    @cached_property
    def _upper_neighbours(self) -> dict[Concept, frozenset[Concept]]:
        neighbours: dict[Concept, set[Concept]] = {c: set() for c in self.concepts}
        for lower, upper in self.edges:
            neighbours[lower].add(upper)
        return {c: frozenset(ups) for c, ups in neighbours.items()}

    @cached_property
    def _lower_neighbours(self) -> dict[Concept, frozenset[Concept]]:
        neighbours: dict[Concept, set[Concept]] = {c: set() for c in self.concepts}
        for lower, upper in self.edges:
            neighbours[upper].add(lower)
        return {c: frozenset(lows) for c, lows in neighbours.items()}

    def to_networkx(self):
        """
        Convert the concept lattice to a NetworkX directed graph.

        Nodes are Concept objects.
        Edges are Hasse cover relations, directed lower -> upper.
        """
        try:
            import networkx as nx
        except ImportError as exc:
            raise ImportError(
                "NetworkX is required for ConceptLattice.to_networkx(). "
                "Install it with `pip install networkx`."
            ) from exc

        graph = nx.DiGraph()
        graph.add_nodes_from(self.concepts)
        graph.add_edges_from(self.edges)

        return graph

    def _check_member(self, concept: Concept) -> None:
        if concept not in self._upper_neighbours:
            raise ValueError("The concept is not in this lattice.")

    def parents(self, concept: Concept) -> frozenset[Concept]:
        """
        Immediate upper neighbours of a concept.
        """
        self._check_member(concept)
        return self._upper_neighbours[concept]

    def children(self, concept: Concept) -> frozenset[Concept]:
        """
        Immediate lower neighbours of a concept.
        """
        self._check_member(concept)
        return self._lower_neighbours[concept]

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    @cached_property
    def _by_extent(self) -> dict[frozenset[int], Concept]:
        return {concept.extent: concept for concept in self.concepts}

    def _lookup(self, extent: frozenset[int], what: str) -> Concept:
        try:
            return self._by_extent[frozenset(extent)]
        except KeyError:
            if self.is_iceberg:
                raise ValueError(
                    f"The {what} has {len(extent)} objects, below this iceberg "
                    f"lattice's min_support={self.min_support}, so it is not in "
                    "the lattice."
                ) from None
            raise ValueError(
                f"The {what} is not among this lattice's concepts."
            ) from None

    def concept_from_objects(self, objects: Iterable[int | str]) -> Concept:
        """
        The smallest concept containing the given objects: ``(A'', A')``.

        Objects may be given as indices or names.
        """
        intent = self.context.object_derivation(objects)
        return self._lookup(self.context.attribute_derivation(intent), "concept")

    def concept_from_attributes(self, attributes: Iterable[int | str]) -> Concept:
        """
        The largest concept whose intent contains the given attributes:
        ``(B', B'')``.

        Attributes may be given as indices or names.
        """
        return self._lookup(self.context.attribute_derivation(attributes), "concept")

    def object_concept(self, obj: int | str) -> Concept:
        """γ(g): the smallest concept whose extent contains object ``g``."""
        return self.concept_from_objects([obj])

    def attribute_concept(self, attribute: int | str) -> Concept:
        """μ(m): the largest concept whose intent contains attribute ``m``."""
        return self.concept_from_attributes([attribute])

    def join(self, *concepts: Concept) -> Concept:
        """
        Supremum: the smallest concept above all given concepts.

        Its extent is the closure of the union of their extents.
        """
        if not concepts:
            raise ValueError("join() needs at least one concept.")
        objects = frozenset().union(*(c.extent for c in concepts))
        return self.concept_from_objects(objects)

    def meet(self, *concepts: Concept) -> Concept:
        """
        Infimum: the largest concept below all given concepts.

        Its extent is the intersection of their extents.
        """
        if not concepts:
            raise ValueError("meet() needs at least one concept.")
        extent = frozenset.intersection(*(c.extent for c in concepts))
        return self._lookup(extent, "meet")

    def _reachable(self, concept: Concept, neighbours) -> frozenset[Concept]:
        self._check_member(concept)
        seen = {concept}
        stack = [concept]
        while stack:
            for other in neighbours[stack.pop()]:
                if other not in seen:
                    seen.add(other)
                    stack.append(other)
        return frozenset(seen)

    def upset(self, concept: Concept) -> frozenset[Concept]:
        """All concepts above or equal to ``concept`` (its principal filter)."""
        return self._reachable(concept, self._upper_neighbours)

    def downset(self, concept: Concept) -> frozenset[Concept]:
        """All concepts below or equal to ``concept`` (its principal ideal)."""
        return self._reachable(concept, self._lower_neighbours)

    def atoms(self) -> frozenset[Concept]:
        """The upper neighbours of the bottom concept."""
        return self.parents(self.bottom())

    def coatoms(self) -> frozenset[Concept]:
        """The lower neighbours of the top concept."""
        return self.children(self.top())

    def _require_full(self, what: str) -> None:
        if self.is_iceberg:
            raise ValueError(
                f"{what} only exist in the full lattice; this is an iceberg lattice "
                f"(min_support={self.min_support}), which has no bottom concept."
            )

    def top(self) -> Concept:
        """
        Return the top concept.

        The top concept has the largest extent.
        """
        return max(self.concepts, key=lambda c: len(c.extent))

    def bottom(self) -> Concept:
        """
        Return the bottom concept.

        The bottom concept has the smallest extent. An iceberg lattice has
        no bottom concept, so this raises ``ValueError`` there.
        """
        self._require_full("The bottom concept and atoms")
        return min(self.concepts, key=lambda c: len(c.extent))

    def __repr__(self) -> str:
        return (
            "ConceptLattice("
            f"n_concepts={self.n_concepts}, "
            f"context_size=({self.context.n_objects}, "
            f"{self.context.n_attributes})"
            ")"
        )
