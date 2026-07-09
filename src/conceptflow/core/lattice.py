"""
Pure concept lattice data structure.

This module defines the internal mathematical representation of a concept
lattice. It is not a scikit-learn estimator.

The sklearn-facing estimator will live in:

    conceptflow.cluster

and will use this class internally.
"""

from __future__ import annotations

from dataclasses import dataclass

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
    """

    context: FormalContext
    concepts: tuple[Concept, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "concepts", tuple(self.concepts))

    @classmethod
    def from_context(
        cls,
        context: FormalContext,
        algorithm: str = "nextclosure",
    ) -> "ConceptLattice":
        """
        Build a concept lattice from a formal context.
        """
        from conceptflow.algorithms.enumeration import enumerate_concepts

        concepts = enumerate_concepts(context, method=algorithm)
        return cls(context=context, concepts=tuple(concepts))

    @property
    def n_concepts(self) -> int:
        """Number of concepts in the lattice."""
        return len(self.concepts)

    @property
    def edges(self) -> tuple[tuple[Concept, Concept], ...]:
        """
        Hasse diagram edges as ``(lower, upper)`` pairs.
        """
        return compute_hasse_edges(self.concepts)
    
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

    def parents(self, concept: Concept) -> frozenset[Concept]:
        """
        Immediate upper neighbours of a concept.
        """
        return frozenset(
            upper for lower, upper in self.edges if lower == concept
        )

    def children(self, concept: Concept) -> frozenset[Concept]:
        """
        Immediate lower neighbours of a concept.
        """
        return frozenset(
            lower for lower, upper in self.edges if upper == concept
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

        The bottom concept has the smallest extent.
        """
        return min(self.concepts, key=lambda c: len(c.extent))

    def __repr__(self) -> str:
        return (
            "ConceptLattice("
            f"n_concepts={self.n_concepts}, "
            f"context_size=({self.context.n_objects}, "
            f"{self.context.n_attributes})"
            ")"
        )