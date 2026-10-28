"""
Public plotting entry points for ConceptFlow lattice visualization.
"""

from __future__ import annotations

from conceptflow.core import ConceptLattice
from conceptflow.visualization.d3_backend import render_with_d3
from conceptflow.visualization.dimflux_layout import lattice_to_graph_data_dimflux
from conceptflow.visualization.graph_data import lattice_to_graph_data

SUPPORTED_BACKENDS = {"graph_data", "d3", "matplotlib", "graphviz", "tikz"}
SUPPORTED_LAYOUTS = {"none", "dimflux", "layered"}


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
        - "graphviz": return a GraphvizFigure (DOT text; ``.write("x.pdf")``
          renders it if Graphviz is installed). Graphviz computes its own
          layout, so ``layout`` is ignored.
        - "tikz": return a TikZFigure (``.source`` is a ``tikzpicture``;
          ``.write("x.tex")`` saves a standalone document, ``.write("x.pdf")``
          compiles it if LaTeX is installed). Uses ``layout``.
        - "graph_data": return backend-neutral GraphData.

    layout:
        Layout strategy.

        - "dimflux": DimFlux coordinates (additive drawing; best for
          small lattices, slow above ~150 concepts).
        - "layered": a fast layered (Sugiyama-style) layout, usable for
          tens of thousands of concepts.
        - "none": no coordinates. Renderers that need them (matplotlib)
          fall back to the layered layout.

    stable_ids:
        Whether to use ``Concept.stable_id()`` (derived from the intent)
        instead of local node ids c0, c1, ...

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
            'Use "graph_data", "d3", "matplotlib", "graphviz" or "tikz".'
        )

    if layout not in SUPPORTED_LAYOUTS:
        raise ValueError(
            f'Unknown layout "{layout}". Use "dimflux", "layered" or "none".'
        )

    if backend == "graphviz":
        from conceptflow.visualization.graphviz import GraphvizFigure, lattice_to_dot

        return GraphvizFigure(lattice_to_dot(lattice, title=title, **kwargs))

    if layout == "none":
        graph_data = lattice_to_graph_data(
            lattice,
            stable_ids=stable_ids,
        )
    elif layout == "layered":
        from conceptflow.visualization.layered_layout import layered_layout

        graph_data = layered_layout(lattice_to_graph_data(lattice, stable_ids=stable_ids))
    else:
        graph_data = lattice_to_graph_data_dimflux(
            lattice,
            stable_ids=stable_ids,
            timeout_ms=timeout_ms,
        )

    if backend == "graph_data":
        return graph_data

    if backend == "tikz":
        from conceptflow.visualization.tikz import TikZFigure, graph_data_to_tikz

        return TikZFigure(graph_data_to_tikz(graph_data, title=title, **kwargs))

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
