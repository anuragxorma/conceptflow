"""
Backend-neutral graph representation for ConceptFlow visualization.

This module converts mathematical FCA structures into graph data without
choosing a rendering backend or layout algorithm.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from conceptflow.core import Concept, ConceptLattice


@dataclass(frozen=True)
class GraphNode:
    """
    Backend-neutral visualization node.
    """

    node_id: str
    label: str
    hover_text: str
    concept: Concept
    metadata: dict[str, Any] = field(default_factory=dict)
    x: float | None = None
    y: float | None = None


@dataclass(frozen=True)
class GraphEdge:
    """
    Backend-neutral visualization edge.
    """

    source: str
    target: str
    directed: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GraphData:
    """
    Backend-neutral graph representation.
    """

    nodes: list[GraphNode]
    edges: list[GraphEdge]
    metadata: dict[str, Any] = field(default_factory=dict)

    def node_ids(self) -> list[str]:
        """
        Return node ids in node order.
        """
        return [node.node_id for node in self.nodes]

    def node_map(self) -> dict[str, GraphNode]:
        """
        Return a mapping from node id to node.
        """
        return {node.node_id: node for node in self.nodes}


def _names_from_indices(names: tuple[str, ...], indices: frozenset[int]) -> tuple[str, ...]:
    """
    Resolve integer indices to names.
    """
    return tuple(names[index] for index in sorted(indices))


def _concept_label(
    lattice: ConceptLattice,
    concept: Concept,
) -> str:
    """
    Create a compact display label for a concept.
    """
    intent_names = _names_from_indices(
        lattice.context.attributes,
        concept.intent,
    )

    if intent_names:
        return ", ".join(intent_names)

    if len(concept.extent) == lattice.context.n_objects:
        return "⊤"

    if len(concept.extent) == 0:
        return "⊥"

    return f"|G|={len(concept.extent)}, |M|={len(concept.intent)}"


def _concept_hover_text(
    lattice: ConceptLattice,
    concept: Concept,
) -> str:
    """
    Create rich hover text for a concept.
    """
    extent_names = _names_from_indices(
        lattice.context.objects,
        concept.extent,
    )
    intent_names = _names_from_indices(
        lattice.context.attributes,
        concept.intent,
    )

    extent_text = ", ".join(extent_names) if extent_names else "∅"
    intent_text = ", ".join(intent_names) if intent_names else "∅"

    return (
        f"Extent: {{{extent_text}}}\n"
        f"Intent: {{{intent_text}}}\n"
        f"|Extent| = {len(concept.extent)}\n"
        f"|Intent| = {len(concept.intent)}"
    )


def lattice_to_graph_data(
    lattice: ConceptLattice,
    stable_ids: bool = False,
) -> GraphData:
    """
    Convert a ConceptLattice into backend-neutral GraphData.

    Parameters
    ----------
    lattice:
        Concept lattice to convert.

    stable_ids:
        If True, use concept-derived stable ids. If False, use compact local
        ids such as c0, c1, c2.

    Returns
    -------
    GraphData
        Backend-neutral graph representation.
    """
    concept_to_id: dict[Concept, str] = {}
    nodes: list[GraphNode] = []

    for index, concept in enumerate(lattice.concepts):
        node_id = concept.stable_id() if stable_ids else f"c{index}"
        concept_to_id[concept] = node_id

        extent_names = _names_from_indices(
            lattice.context.objects,
            concept.extent,
        )
        intent_names = _names_from_indices(
            lattice.context.attributes,
            concept.intent,
        )

        nodes.append(
            GraphNode(
                node_id=node_id,
                label=_concept_label(lattice, concept),
                hover_text=_concept_hover_text(lattice, concept),
                concept=concept,
                metadata={
                    "extent": list(extent_names),
                    "intent": list(intent_names),
                    "extent_indices": sorted(concept.extent),
                    "intent_indices": sorted(concept.intent),
                    "extent_size": len(concept.extent),
                    "intent_size": len(concept.intent),
                    "concept_signature": concept.signature(),
                    "stable_id": concept.stable_id(),
                    "uses_stable_node_id": stable_ids,
                },
            )
        )

    edges: list[GraphEdge] = []

    for lower, upper in lattice.edges:
        edges.append(
            GraphEdge(
                source=concept_to_id[lower],
                target=concept_to_id[upper],
                directed=True,
                metadata={
                    "relation": "cover",
                },
            )
        )

    return GraphData(
        nodes=nodes,
        edges=edges,
        metadata={
            "kind": "concept_lattice",
            "context_size": (
                lattice.context.n_objects,
                lattice.context.n_attributes,
            ),
            "n_objects": lattice.context.n_objects,
            "n_attributes": lattice.context.n_attributes,
            "n_concepts": lattice.n_concepts,
            "stable_ids": stable_ids,
            "layout": None,
        },
    )