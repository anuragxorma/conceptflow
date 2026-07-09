"""
Builder API for nested conceptual exploration.
"""

from __future__ import annotations

from collections.abc import Callable

from conceptflow.core import Concept, ConceptLattice, ManyValuedContext
from conceptflow.exploration.view import ExplorationView
from conceptflow.preprocessing import ConceptualScaler, Scale


class ExplorationBuilder:
    """
    Build root and child ExplorationView objects from many-valued data.
    """

    def __init__(
        self,
        mvc: ManyValuedContext,
        algorithm: str = "nextclosure",
    ) -> None:
        self.mvc = mvc
        self.algorithm = algorithm

    def root(
        self,
        name: str,
        scales: list[Scale],
    ) -> ExplorationView:
        """
        Build the root exploration view.
        """
        scaler = ConceptualScaler(
            scales=scales,
            output="context",
        )

        ctx = scaler.fit_transform(self.mvc)

        lattice = ConceptLattice.from_context(
            ctx,
            algorithm=self.algorithm,
        )

        return ExplorationView(
            name=name,
            context=ctx,
            lattice=lattice,
            depth=0,
            scale_names=tuple(scale.source_attribute for scale in scales),
        )

    def _is_expandable(self, concept: Concept) -> bool:
        """
        Return whether a concept can be expanded into a child view.
        """
        return len(concept.extent) > 0

    def _object_names_from_parent_extent(
        self,
        parent: ExplorationView,
        concept: Concept,
    ) -> list[str]:
        """
        Resolve concept extent indices to object names in the parent context.
        """
        return [
            parent.context.objects[index]
            for index in sorted(concept.extent)
        ]

    def expand(
        self,
        parent: ExplorationView,
        concept: Concept,
        name: str,
        scales: list[Scale],
    ) -> ExplorationView:
        """
        Expand one parent concept into a child exploration view.
        """
        if not self._is_expandable(concept):
            raise ValueError(
                "Cannot expand a concept with an empty extent. "
                "There are no objects from which to build a child view."
            )

        concept_id = concept.stable_id()

        object_names = self._object_names_from_parent_extent(
            parent,
            concept,
        )

        sub_mvc = self.mvc.restrict_objects(object_names)

        scaler = ConceptualScaler(
            scales=scales,
            output="context",
        )

        child_ctx = scaler.fit_transform(sub_mvc)

        child_lattice = ConceptLattice.from_context(
            child_ctx,
            algorithm=self.algorithm,
        )

        child_view = ExplorationView(
            name=name,
            context=child_ctx,
            lattice=child_lattice,
            depth=parent.depth + 1,
            scale_names=tuple(scale.source_attribute for scale in scales),
            parent_concept_id=concept_id,
        )

        parent.add_child(concept_id, child_view)

        return child_view

    def concept_label(
        self,
        parent: ExplorationView,
        concept: Concept,
    ) -> str:
        """
        Create a readable label for a concept.
        """
        if len(concept.extent) == 0:
            return "bottom"

        if concept.intent:
            intent_names = [
                parent.context.attributes[index]
                for index in sorted(concept.intent)
            ]
            return ", ".join(intent_names)

        return "top"

    def expand_all(
        self,
        parent: ExplorationView,
        scales: list[Scale],
        name_template: str = "{label}",
        min_extent_size: int = 0,
        max_extent_size: int | None = None,
        include_top: bool = True,
        include_bottom: bool = True,
        predicate: Callable[[Concept], bool] | None = None,
    ) -> list[ExplorationView]:
        """
        Expand all selected expandable concepts of a parent view.
        """
        children: list[ExplorationView] = []

        parent_object_count = parent.context.n_objects

        for concept in parent.lattice.concepts:
            extent_size = len(concept.extent)

            is_bottom = extent_size == 0
            is_top = extent_size == parent_object_count

            if is_bottom and not include_bottom:
                continue

            if is_top and not include_top:
                continue

            if extent_size < min_extent_size:
                continue

            if max_extent_size is not None and extent_size > max_extent_size:
                continue

            if predicate is not None and not predicate(concept):
                continue

            if not self._is_expandable(concept):
                continue

            label = self.concept_label(parent, concept)

            intent_text = ""
            if concept.intent:
                intent_text = ", ".join(
                    parent.context.attributes[index]
                    for index in sorted(concept.intent)
                )

            name = name_template.format(
                label=label,
                intent=intent_text,
                extent_size=extent_size,
                depth=parent.depth + 1,
            )

            child = self.expand(
                parent=parent,
                concept=concept,
                name=name,
                scales=scales,
            )

            children.append(child)

        return children