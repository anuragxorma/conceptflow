"""
Public plotting entry points for ConceptFlow lattice visualization.
"""

from __future__ import annotations

from conceptflow.core import ConceptLattice
from conceptflow.visualization.d3_backend import render_with_d3
from conceptflow.visualization.dimflux_layout import lattice_to_graph_data_dimflux
from conceptflow.visualization.graph_data import lattice_to_graph_data


SUPPORTED_BACKENDS = {"graph_data", "d3"}
SUPPORTED_LAYOUTS = {"none", "dimflux"}


def plot_lattice(
    lattice: ConceptLattice,
    backend: str = "d3",
    layout: str = "dimflux",
    stable_ids: bool = False,
    width: int = 900,
    height: int = 700,
    title: str = "Concept lattice",
    **kwargs,
):
    """
    Plot or export a concept lattice.

    Parameters
    ----------
    lattice:
        Concept lattice to visualize.

    backend:
        Rendering backend.

        - "d3": return an interactive HTMLFigure.
        - "graph_data": return backend-neutral GraphData.

    layout:
        Layout strategy.

        - "dimflux": compute DimFlux coordinates.
        - "none": do not assign coordinates.

    stable_ids:
        Whether to use stable concept ids instead of local node ids.

    width, height:
        Figure size in pixels.

    title:
        Figure title shown by interactive renderers.

    Returns
    -------
    HTMLFigure or GraphData
        Depends on the selected backend.
    """
    if backend not in SUPPORTED_BACKENDS:
        raise ValueError(
            f'Unknown visualization backend "{backend}". '
            'Use "graph_data" or "d3".'
        )

    if layout not in SUPPORTED_LAYOUTS:
        raise ValueError(
            f'Unknown layout "{layout}". Use "none" or "dimflux".'
        )

    if layout == "none":
        graph_data = lattice_to_graph_data(
            lattice,
            stable_ids=stable_ids,
        )
    else:
        graph_data = lattice_to_graph_data_dimflux(
            lattice,
            stable_ids=stable_ids,
        )

    if backend == "graph_data":
        return graph_data

    return render_with_d3(
        graph_data,
        width=width,
        height=height,
        title=title,
        **kwargs,
    )