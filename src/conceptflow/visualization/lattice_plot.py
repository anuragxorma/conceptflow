"""
Public plotting entry points for ConceptFlow lattice visualization.
"""

from __future__ import annotations

from conceptflow.core import ConceptLattice
from conceptflow.visualization.d3_backend import render_with_d3
from conceptflow.visualization.dimflux_layout import lattice_to_graph_data_dimflux
from conceptflow.visualization.graph_data import lattice_to_graph_data

SUPPORTED_BACKENDS = {"graph_data", "d3", "matplotlib"}
SUPPORTED_LAYOUTS = {"none", "dimflux"}


def plot_lattice(
    lattice: ConceptLattice,
    backend: str = "d3",
    layout: str = "dimflux",
    stable_ids: bool = False,
    width: int = 900,
    height: int = 700,
    title: str | None = None,
    timeout_ms: int | None = 1000,
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
        - "matplotlib": return a static matplotlib Figure; write it with
          ``fig.savefig("lattice.pdf")`` (or .svg/.png/.eps). Requires
          matplotlib.
        - "graph_data": return backend-neutral GraphData.

    layout:
        Layout strategy.

        - "dimflux": compute DimFlux coordinates.
        - "none": do not assign coordinates. The matplotlib backend then
          uses a quick layered layout, which is useful for lattices too
          large for DimFlux.

    stable_ids:
        Whether to use stable concept ids instead of local node ids.

    width, height:
        Figure size in pixels. For the matplotlib backend they set
        ``figsize`` at 100 dpi unless ``figsize`` is passed explicitly.

    title:
        Figure title. Defaults to "Concept lattice" for the d3 backend and
        to no title for matplotlib figures.

    timeout_ms:
        DimFlux time budget in milliseconds (see
        ``lattice_to_graph_data_dimflux``); ``None`` searches for the optimal
        layout, which can take very long on large lattices.

    **kwargs:
        Passed to the renderer: ``render_with_d3`` or
        ``render_lattice_matplotlib`` (e.g. ``ax``, ``figsize``,
        ``show_object_labels``, ``max_label_items``, ``object_label``).

    Returns
    -------
    HTMLFigure, matplotlib Figure, or GraphData
        Depends on the selected backend.
    """
    if backend not in SUPPORTED_BACKENDS:
        raise ValueError(
            f'Unknown visualization backend "{backend}". '
            'Use "graph_data", "d3" or "matplotlib".'
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
            timeout_ms=timeout_ms,
        )

    if backend == "graph_data":
        return graph_data

    if backend == "matplotlib":
        from conceptflow.visualization.static import render_lattice_matplotlib

        kwargs.setdefault("figsize", (width / 100, height / 100))
        return render_lattice_matplotlib(graph_data, title=title, **kwargs)

    return render_with_d3(
        graph_data,
        width=width,
        height=height,
        title="Concept lattice" if title is None else title,
        **kwargs,
    )
