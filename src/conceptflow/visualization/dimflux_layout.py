"""
DimFlux layout bridge.

This module uses DimFlux as a layout engine and attaches computed coordinates
to ConceptFlow GraphData.
"""

from __future__ import annotations

import tempfile
from dataclasses import replace
from pathlib import Path

from conceptflow.core import Concept, ConceptLattice
from conceptflow.io import write_cxt
from conceptflow.visualization.graph_data import GraphData, lattice_to_graph_data
from conceptflow.visualization.dimflux.src.dim_flux.realizer import Realizer
from conceptflow.visualization.dimflux.src.fdp.forces import ForceDirectedPlacement
from conceptflow.visualization.dimflux.src.utils.variables import Variables


def _concept_signature(concept: Concept) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """
    Return a hashable concept signature based on extent and intent indices.
    """
    return (
        tuple(sorted(concept.extent)),
        tuple(sorted(concept.intent)),
    )

def _concept_indices_to_names(concept: Concept, context):
    """
    Convert a ConceptFlow concept with integer extents/intents to name sets.

    DimFlux internals expect object/attribute labels when computing
    force-directed positions.
    """
    extent_names = {
        context.objects[index]
        for index in concept.extent
    }

    intent_names = {
        context.attributes[index]
        for index in concept.intent
    }

    return extent_names, intent_names

def lattice_to_graph_data_dimflux(
    lattice: ConceptLattice,
    stable_ids: bool = False,
    x_scale: float = 25.0,
    y_scale: float = 25.0,
) -> GraphData:
    """
    Convert a ConceptLattice into GraphData with DimFlux coordinates.
    """
    graph_data = lattice_to_graph_data(
        lattice,
        stable_ids=stable_ids,
    )

    with tempfile.NamedTemporaryFile(suffix=".cxt", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        write_cxt(lattice.context, tmp_path)

        vars_ = Variables(str(tmp_path), {})

        realizer = Realizer(vars_)
        vars_.base_vectors = realizer.base_vectors
        vars_.coordinates = realizer.coordinates

        # --------------------------------------------------------------
        # Compatibility with the new ConceptFlow core:
        #
        # ConceptFlow concepts now store extent/intent as integer indices.
        # DimFlux force placement expects object/attribute labels because
        # its element_map is keyed by labels.
        #
        # Therefore we convert concept extents/intents to label sets before
        # running ForceDirectedPlacement.
        # --------------------------------------------------------------
        vars_.extents = {}
        vars_.intents = {}

        for concept in vars_.coordinates:
            extent_names, intent_names = _concept_indices_to_names(
                concept,
                vars_.context,
            )
            vars_.extents[concept] = extent_names
            vars_.intents[concept] = intent_names

        fdp = ForceDirectedPlacement(vars_)

        concept_to_node = {
            _concept_signature(node.concept): node
            for node in graph_data.nodes
        }

        new_nodes = []

        for dimflux_concept, coords in fdp.coordinates.items():
            signature = _concept_signature(dimflux_concept)
            try:
                node = concept_to_node[signature]
            except KeyError as exc:
                raise RuntimeError(
                    f"DimFlux returned a concept with no matching ConceptFlow node "
                    f"(signature={signature!r}). "
                    "This likely indicates a mismatch between the .cxt file written "
                    "to disk and the in-memory lattice."
                ) from exc

            x = float(coords[0]) * x_scale
            y = -float(coords[1]) * y_scale

            new_nodes.append(
                replace(
                    node,
                    x=x,
                    y=y,
                    metadata={
                        **node.metadata,
                        "layout": "dimflux",
                    },
                )
            )

        node_map = {
            node.node_id: node
            for node in new_nodes
        }

        ordered_nodes = [
            node_map[node.node_id]
            for node in graph_data.nodes
        ]

        return GraphData(
            nodes=ordered_nodes,
            edges=graph_data.edges,
            metadata={
                **graph_data.metadata,
                "layout": "dimflux",
            },
        )

    finally:
        tmp_path.unlink(missing_ok=True)