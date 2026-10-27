"""
DimFlux layout bridge.

This module uses DimFlux (the ``dim-flux`` package on PyPI) as a layout
engine and attaches computed coordinates to ConceptFlow GraphData.

DimFlux: Nöhre, Dürrschnabel, Ganter & Stumme, "DimFlux: Force-Directed
Additive Line Diagrams" (2026), arXiv:2603.16366.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import replace

import numpy as np
import pandas as pd

from conceptflow.core import ConceptLattice, FormalContext
from conceptflow.visualization.graph_data import GraphData, lattice_to_graph_data

# Above this many concepts DimFlux typically needs minutes, regardless of
# timeout_ms (see lattice_to_graph_data_dimflux).
DIMFLUX_SLOW_LATTICE_SIZE = 150


def _sanitized_dataframe(context: FormalContext) -> pd.DataFrame:
    """
    Return the incidence with synthetic object/attribute names.

    DimFlux rejects duplicate names, and an object may share its name with
    an attribute in ConceptFlow. Synthetic names ``g<i>`` / ``m<j>`` keep the
    original order, so DimFlux extents translate straight back to object
    indices.
    """
    return pd.DataFrame(
        np.asarray(context.incidence, dtype=bool),
        index=[f"g{i}" for i in range(context.n_objects)],
        columns=[f"m{j}" for j in range(context.n_attributes)],
    )


def _dimflux_coordinates(
    context: FormalContext,
    timeout_ms: int | None,
) -> dict[frozenset[int], np.ndarray]:
    """
    Run the DimFlux pipeline and return coordinates keyed by concept extent.

    These are the steps ``dim_flux.plot`` runs: DimDraw-based initial
    realizer, then force-directed placement. ``plot`` only returns positions
    in its own internal concept order, so the steps are run here directly to
    read each concept's extent alongside its coordinates.
    """
    # Imported lazily: dim_flux pulls in matplotlib.pyplot, sympy, odis and
    # fcapy, which `import conceptflow` should not pay for.
    from dim_flux.core.realizer import Realizer
    from dim_flux.fdp.forces import ForceDirectedPlacement
    from dim_flux.utils.variables import Variables

    variables = Variables(_sanitized_dataframe(context), {}, timeout_ms=timeout_ms)

    realizer = Realizer(variables)
    if not hasattr(realizer, "base_vectors"):
        raise RuntimeError(
            "DimFlux could not compute an initial layout for this lattice. "
            'Use layout="none" instead.'
        )
    variables.base_vectors = realizer.base_vectors
    variables.coordinates = realizer.coordinates
    variables.lectic_order = realizer.lectic_order

    placement = ForceDirectedPlacement(variables)

    return {
        frozenset(int(name[1:]) for name in variables.full_extents[concept_id]): coords
        for concept_id, coords in placement.coordinates.items()
    }


def lattice_to_graph_data_dimflux(
    lattice: ConceptLattice,
    stable_ids: bool = False,
    x_scale: float = 25.0,
    y_scale: float = 25.0,
    timeout_ms: int | None = 1000,
) -> GraphData:
    """
    Convert a ConceptLattice into GraphData with DimFlux coordinates.

    Parameters
    ----------
    lattice:
        Concept lattice to lay out.

    stable_ids:
        Whether to use stable concept ids as node ids.

    x_scale, y_scale:
        Factors applied to DimFlux coordinates.

    timeout_ms:
        Time limit, in milliseconds, for DimFlux's layout search and for its
        force-directed optimization; the best layout found so far is kept.
        It is checked between optimization steps, so it bounds each phase
        only loosely and is *not* a limit on total run time: lattices with
        a few hundred concepts can take minutes even with the default.
        ``None`` runs both to a proven/converged optimum, which can take
        much longer.
    """
    if timeout_ms is not None and (
        isinstance(timeout_ms, bool) or not isinstance(timeout_ms, int) or timeout_ms < 0
    ):
        raise ValueError("timeout_ms must be None or a non-negative integer.")

    if lattice.n_concepts > DIMFLUX_SLOW_LATTICE_SIZE:
        warnings.warn(
            f"DimFlux layout of {lattice.n_concepts} concepts may take several "
            'minutes. For large lattices, layout="none" is much faster.',
            skip_file_prefixes=(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),),
        )

    graph_data = lattice_to_graph_data(
        lattice,
        stable_ids=stable_ids,
    )

    # DimFlux needs at least two concepts, and for exactly two it returns
    # (almost) the same point twice: draw those by hand, top above bottom
    # (screen coordinates, y grows downwards).
    if lattice.n_concepts < 2:
        coordinates = {concept.extent: np.zeros(2) for concept in lattice.concepts}
    elif lattice.n_concepts == 2:
        coordinates = {
            concept.extent: np.array([0.0, 0.0 if concept == lattice.top() else 1.0])
            for concept in lattice.concepts
        }
    else:
        coordinates = _dimflux_coordinates(lattice.context, timeout_ms)

    new_nodes = []

    for node in graph_data.nodes:
        try:
            coords = coordinates[node.concept.extent]
        except KeyError as exc:
            raise RuntimeError(
                "DimFlux returned no coordinates for the concept with extent "
                f"{sorted(node.concept.extent)}; the lattice does not match its "
                "context."
            ) from exc

        new_nodes.append(
            replace(
                node,
                x=float(coords[0]) * x_scale,
                y=-float(coords[1]) * y_scale,
                metadata={
                    **node.metadata,
                    "layout": "dimflux",
                },
            )
        )

    return GraphData(
        nodes=new_nodes,
        edges=graph_data.edges,
        metadata={
            **graph_data.metadata,
            "layout": "dimflux",
        },
    )
