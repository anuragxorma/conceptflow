"""
Builder API for nested conceptual exploration.
"""

from __future__ import annotations

import numbers
from collections.abc import Callable

from sklearn.utils import check_scalar

from conceptflow.core import Concept, ConceptLattice, ManyValuedContext
from conceptflow.exploration.view import ExplorationView
from conceptflow.preprocessing import ConceptualScaler, Scale


class ExplorationBuilder:
    """
    Build root and child ExplorationView objects from many-valued data.

    Parameters
    ----------
    mvc:
        Many-valued data to explore.

    algorithm:
        Concept enumeration algorithm.

    missing:
        Missing-value policy passed to ``ConceptualScaler``: "error"
        (default) or "false". Nested diagrams need complete contexts, so
        "unknown" is not available here.
    """

    def __init__(
        self,
        mvc: ManyValuedContext,
        algorithm: str = "nextclosure",
        missing: str = "error",
    ) -> None:
        if missing not in {"error", "false"}:
            raise ValueError(
                'ExplorationBuilder supports missing="error" or missing="false".'
            )

        self.mvc = mvc
        self.algorithm = algorithm
        self.missing = missing

    def root(
        self,
        name: str,
        scales: list[Scale],
    ) -> ExplorationView:
        """
        Build the root exploration view.
        """
        ctx = self._scale(self.mvc, scales)

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

    @staticmethod
    def _check_scales(scales) -> None:
        if isinstance(scales, (str, Scale)) or not all(
            isinstance(scale, Scale) for scale in scales
        ):
            raise TypeError(
                f"scales must be a list of Scale objects, got {scales!r}."
            )

    def _scale(self, mvc: ManyValuedContext, scales: list[Scale]):
        """
        Scale ``mvc`` into a formal context with this builder's policy.

        The scales are fit on all of the builder's data, so a view of some
        objects has the same attributes as the view of all objects, including
        values its objects lack: the "≠ v" of a contranominal scale holds for
        all of them, and dropping it would change their intents.
        """
        self._check_scales(scales)
        scaler = ConceptualScaler(
            scales=scales,
            output="context",
            missing="false",  # the vocabulary; missing values are checked below
        ).fit(self.mvc)
        scaler.set_params(missing=self.missing)

        try:
            return scaler.transform(mvc)
        except ValueError as exc:
            if "Input contains NaN" not in str(exc):
                raise
            # The scaler's own hint mentions missing="unknown", which nested
            # diagrams cannot use.
            raise ValueError(
                f"{str(exc).split('. Pass ')[0]}. Create the builder with "
                'ExplorationBuilder(..., missing="false") to treat missing values '
                "as having none of the scale's attributes."
            ) from exc

    def _is_expandable(self, concept: Concept) -> bool:
        """
        Return whether a concept can be expanded into a child view.
        """
        return len(concept.extent) > 0

    @staticmethod
    def _is_concept_of(view: ExplorationView, concept: Concept) -> bool:
        """Whether ``concept`` is a concept of ``view``'s context."""
        context = view.context
        if not all(0 <= g < context.n_objects for g in concept.extent) or not all(
            0 <= m < context.n_attributes for m in concept.intent
        ):
            return False
        return (
            context.object_derivation(concept.extent) == concept.intent
            and context.attribute_derivation(concept.intent) == concept.extent
        )

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

        The child view scales the objects of ``concept`` with ``scales``,
        fit on all of the builder's data: it has the same attributes as a view
        of all objects with these scales.
        ``concept`` must be a concept of ``parent.lattice``. Expanding a
        concept that already has a child view replaces that view.
        """
        if not self._is_concept_of(parent, concept):
            raise ValueError(
                f"The concept is not a concept of the parent view {parent.name!r}. "
                "Pass a concept from parent.lattice.concepts; concepts of other "
                "views refer to other objects."
            )
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

        child_ctx = self._scale(sub_mvc, scales)

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

        Concepts that already have a child view get a new one (see
        ``expand``).

        Parameters
        ----------
        name_template:
            Name of each child view. May use ``{label}``, ``{intent}``,
            ``{extent_size}`` and ``{depth}``.

        min_extent_size, max_extent_size:
            Expand only concepts with at least / at most this many objects.

        include_top:
            If False, skip the top concept (the one with every object in its
            extent).

        include_bottom:
            Has no effect: a concept with an empty extent has no objects to
            build a child view from, so it is always unexpandable (``expand()``
            raises ValueError for it) regardless of this flag. Kept for
            backward-compatible signatures; do not rely on it to include or
            exclude anything.

        predicate:
            If given, expand only concepts for which ``predicate(concept)``
            is true.
        """
        self._check_scales(scales)
        if not isinstance(name_template, str):
            raise TypeError(
                f"name_template must be a string, got {name_template!r}."
            )
        try:
            name_template.format(label="", intent="", extent_size=0, depth=0)
        except (KeyError, IndexError, ValueError) as exc:
            raise ValueError(
                f"name_template {name_template!r} is not valid: it may use {{label}}, "
                "{intent}, {extent_size} and {depth}."
            ) from exc
        check_scalar(min_extent_size, "min_extent_size", numbers.Integral, min_val=0)
        if max_extent_size is not None:
            check_scalar(max_extent_size, "max_extent_size", numbers.Integral, min_val=0)
        if predicate is not None and not callable(predicate):
            raise TypeError(
                f"predicate must be a callable taking a Concept, got {predicate!r}."
            )
        children: list[ExplorationView] = []

        parent_object_count = parent.context.n_objects

        for concept in parent.lattice.concepts:
            extent_size = len(concept.extent)

            is_top = extent_size == parent_object_count

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
