"""
Exploration view model for Toscana-style nested conceptual exploration.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from conceptflow.core import ConceptLattice, FormalContext


@dataclass
class ExplorationView:
    """
    One level in a nested conceptual exploration.
    """

    name: str
    context: FormalContext
    lattice: ConceptLattice
    depth: int = 0
    scale_names: tuple[str, ...] = ()
    parent_concept_id: str | None = None
    children: dict[str, "ExplorationView"] = field(default_factory=dict)

    def add_child(self, concept_id: str, child: "ExplorationView") -> None:
        """
        Attach a child exploration view to a concept in this view.
        """
        self.children[concept_id] = child

    def has_child(self, concept_id: str) -> bool:
        """
        Return whether the given concept has an attached child view.
        """
        return concept_id in self.children

    def child(self, concept_id: str) -> "ExplorationView":
        """
        Return the child view attached to a concept.
        """
        return self.children[concept_id]

    def __repr__(self) -> str:
        """
        Return a compact debug representation.
        """
        context_size = (
            self.context.n_objects,
            self.context.n_attributes,
        )

        return (
            "ExplorationView("
            f"name={self.name!r}, "
            f"depth={self.depth}, "
            f"context_size={context_size}, "
            f"n_concepts={self.lattice.n_concepts}, "
            f"n_children={len(self.children)}"
            ")"
        )