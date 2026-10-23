"""
Static (matplotlib) rendering of lattices and nested diagrams.

The interactive renderers produce HTML. This module draws the same layouts
onto a matplotlib ``Figure``, so publication figures can be written with
``fig.savefig("lattice.pdf")`` (or ``.svg``, ``.png``, ``.eps``) without a
browser.

matplotlib is installed together with ConceptFlow (DimFlux depends on it).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

from conceptflow.visualization.graph_data import GraphData

_LINE_COLOR = "#555555"
_FILLED_COLOR = "#1f4e79"
_ATTRIBUTE_COLOR = "#1f4e79"
_OBJECT_COLOR = "#333333"
# Light backing so labels stay readable where they cross edges.
_LABEL_BOX = {"boxstyle": "round,pad=0.1", "facecolor": "white", "edgecolor": "none", "alpha": 0.8}


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "Static figures need matplotlib: `pip install matplotlib`."
        ) from exc
    return plt


def _new_axes(ax, figsize):
    plt = _require_matplotlib()
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    ax.set_axis_off()
    # The renderers frame the drawing with _set_limits, which never yields a
    # zero-width box, so a chain does not shrink the axes to nothing.
    ax.set_aspect("equal")
    return fig, ax


def _set_limits(ax, points, pad: float) -> None:
    """Frame ``points`` with padding, never collapsing to a zero-width box."""
    xs = [x for x, _ in points]
    ys = [y for _, y in points]
    span = max(max(xs) - min(xs), max(ys) - min(ys), 1e-9)
    half_x = max((max(xs) - min(xs)) / 2, 0.25 * span) + pad
    half_y = max((max(ys) - min(ys)) / 2, 0.25 * span) + pad
    cx = (max(xs) + min(xs)) / 2
    cy = (max(ys) + min(ys)) / 2
    ax.set_xlim(cx - half_x, cx + half_x)
    ax.set_ylim(cy - half_y, cy + half_y)


def _math_coordinates(x: float, y: float) -> tuple[float, float]:
    """GraphData coordinates are screen coordinates (y grows downwards)."""
    return x, -y


def _join_labels(items: list[str], max_items: int | None) -> str:
    if max_items is not None and len(items) > max_items:
        hidden = f"+{len(items) - max_items}"
        return ", ".join([*items[:max_items], hidden])
    return ", ".join(items)


def _upright(
    positions: dict[str, tuple[float, float]],
    top_id: str,
    bottom_id: str,
) -> dict[str, tuple[float, float]]:
    """
    Rotate a layout so that the bottom-to-top vector points straight up.

    Layout engines such as DimFlux return drawings in an arbitrary
    orientation; the HTML renderers apply the same normalization.
    """
    (bx, by), (tx, ty) = positions[bottom_id], positions[top_id]
    if math.hypot(tx - bx, ty - by) < 1e-9:
        return dict(positions)

    angle = math.pi / 2 - math.atan2(ty - by, tx - bx)
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    return {
        node_id: (x * cos_a - y * sin_a, x * sin_a + y * cos_a)
        for node_id, (x, y) in positions.items()
    }


def _layered_positions(graph_data: GraphData) -> dict[str, tuple[float, float]]:
    """
    Simple fallback layout for nodes without coordinates.

    Nodes are placed in rows by their longest-path distance from the top
    concept, and each row is ordered by the mean x of the nodes above it.
    """
    upper: dict[str, list[str]] = {node.node_id: [] for node in graph_data.nodes}
    for edge in graph_data.edges:
        upper[edge.source].append(edge.target)

    # Longest-path depth from the top, iteratively (Kahn's algorithm) so
    # deep lattices cannot hit the recursion limit.
    lower: dict[str, list[str]] = {node_id: [] for node_id in upper}
    for node_id, ups in upper.items():
        for u in ups:
            lower[u].append(node_id)

    remaining = {node_id: len(ups) for node_id, ups in upper.items()}
    level: dict[str, int] = {}
    frontier = [node_id for node_id, n in remaining.items() if n == 0]
    while frontier:
        node_id = frontier.pop()
        level[node_id] = 1 + max((level[u] for u in upper[node_id]), default=-1)
        for low in lower[node_id]:
            remaining[low] -= 1
            if remaining[low] == 0:
                frontier.append(low)

    rows: dict[int, list[str]] = {}
    for node in graph_data.nodes:
        rows.setdefault(level[node.node_id], []).append(node.node_id)

    x: dict[str, float] = {}
    for row_index in sorted(rows):
        row = rows[row_index]
        row.sort(
            key=lambda n: (
                sum(x[u] for u in upper[n]) / len(upper[n]) if upper[n] else 0.0
            )
        )
        for i, node_id in enumerate(row):
            x[node_id] = i - (len(row) - 1) / 2

    return {node_id: (x[node_id], -float(level[node_id])) for node_id in x}


def _reduced_labels(graph_data: GraphData) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """
    Reduced labelling: attribute m at the largest concept containing m in
    its intent, object g at the smallest concept containing g in its extent.
    """
    attributes_at: dict[str, list[str]] = {node.node_id: [] for node in graph_data.nodes}
    objects_at: dict[str, list[str]] = {node.node_id: [] for node in graph_data.nodes}

    best_attr: dict[str, Any] = {}
    best_obj: dict[str, Any] = {}

    for node in graph_data.nodes:
        size = node.metadata["extent_size"]
        for name in node.metadata["intent"]:
            if name not in best_attr or size > best_attr[name][0]:
                best_attr[name] = (size, node.node_id)
        for name in node.metadata["extent"]:
            if name not in best_obj or size < best_obj[name][0]:
                best_obj[name] = (size, node.node_id)

    for name, (_, node_id) in best_attr.items():
        attributes_at[node_id].append(name)
    for name, (_, node_id) in best_obj.items():
        objects_at[node_id].append(name)

    return attributes_at, objects_at


def render_lattice_matplotlib(
    graph_data: GraphData,
    ax=None,
    figsize: tuple[float, float] = (8.0, 8.0),
    title: str | None = None,
    show_attribute_labels: bool = True,
    show_object_labels: bool = True,
    max_label_items: int | None = 6,
    object_label: Callable[[str], str] | None = None,
    node_size: float = 60.0,
    font_size: float = 7.0,
):
    """
    Draw a lattice ``GraphData`` with matplotlib.

    Nodes keep the coordinates of the chosen layout (e.g. DimFlux); nodes
    without coordinates get a simple layered layout. Labels follow the usual
    reduced labelling: attributes above the node that introduces them,
    objects below.

    Returns
    -------
    matplotlib.figure.Figure
        Save it with ``fig.savefig("lattice.pdf")`` (or .svg/.png/.eps).
    """
    from matplotlib.collections import LineCollection

    fig, ax = _new_axes(ax, figsize)

    if any(node.x is None or node.y is None for node in graph_data.nodes):
        positions = _layered_positions(graph_data)
    else:
        positions = {
            node.node_id: _math_coordinates(node.x, node.y) for node in graph_data.nodes
        }
        by_size = sorted(graph_data.nodes, key=lambda n: n.metadata["extent_size"])
        positions = _upright(positions, by_size[-1].node_id, by_size[0].node_id)

    segments = [
        (positions[edge.source], positions[edge.target]) for edge in graph_data.edges
    ]
    ax.add_collection(LineCollection(segments, colors=_LINE_COLOR, linewidths=0.8, zorder=1))

    xs = [positions[node.node_id][0] for node in graph_data.nodes]
    ys = [positions[node.node_id][1] for node in graph_data.nodes]
    ax.scatter(
        xs, ys, s=node_size, facecolors="white", edgecolors="black", linewidths=0.9, zorder=2
    )

    attributes_at, objects_at = _reduced_labels(graph_data)
    format_object = object_label or (lambda name: name)
    offset = math.sqrt(node_size) / 2 + 2

    for node in graph_data.nodes:
        x, y = positions[node.node_id]
        if show_attribute_labels and attributes_at[node.node_id]:
            ax.annotate(
                _join_labels(sorted(attributes_at[node.node_id]), max_label_items),
                (x, y), xytext=(0, offset), textcoords="offset points",
                ha="center", va="bottom", fontsize=font_size,
                color=_ATTRIBUTE_COLOR, bbox=_LABEL_BOX,
            )
        if show_object_labels and objects_at[node.node_id]:
            names = sorted(format_object(name) for name in objects_at[node.node_id])
            ax.annotate(
                _join_labels(names, max_label_items),
                (x, y), xytext=(0, -offset), textcoords="offset points",
                ha="center", va="top", fontsize=font_size, color=_OBJECT_COLOR, bbox=_LABEL_BOX,
            )

    span = max(
        max(xs) - min(xs) if xs else 0.0,
        max(ys) - min(ys) if ys else 0.0,
    )
    if xs:
        _set_limits(ax, list(zip(xs, ys)), pad=0.12 * span if span > 0 else 1.0)

    if title:
        ax.set_title(title)

    return fig


def _normalized_template(template: dict[str, Any]) -> dict[str, tuple[float, float]]:
    """Center the inner template on the origin and scale it into the unit disc."""
    nodes = template["nodes"]
    by_size = sorted(nodes, key=lambda n: n["extent_size"])
    positions = _upright(
        {node["id"]: _math_coordinates(node["x"], node["y"]) for node in nodes},
        by_size[-1]["id"],
        by_size[0]["id"],
    )
    xs = [x for x, _ in positions.values()]
    ys = [y for _, y in positions.values()]
    cx = (min(xs) + max(xs)) / 2
    cy = (min(ys) + max(ys)) / 2
    radius = max(math.hypot(x - cx, y - cy) for x, y in zip(xs, ys)) or 1.0
    return {
        node_id: ((x - cx) / radius, (y - cy) / radius)
        for node_id, (x, y) in positions.items()
    }


def render_nested_matplotlib(
    data: dict[str, Any],
    ax=None,
    figsize: tuple[float, float] = (9.0, 9.0),
    title: str | None = None,
    show_attribute_labels: bool = True,
    show_object_labels: bool = False,
    max_label_items: int | None = 4,
    inner_node_size: float = 14.0,
    font_size: float = 7.0,
):
    """
    Draw a two-level nested line diagram with matplotlib.

    ``data`` is the output of ``exploration_view_to_nested_data``. Every outer
    concept is drawn as a circle containing the inner template; realized
    (outer, inner) pairs are filled, unrealized ones hollow. Outer attribute
    labels sit above their circle; inner attribute labels are shown once,
    inside the top outer concept.

    Only the first nesting level is drawn, as in the HTML preview.

    Returns
    -------
    matplotlib.figure.Figure
    """
    from matplotlib.collections import LineCollection
    from matplotlib.patches import Circle

    fig, ax = _new_axes(ax, figsize)

    outer_nodes = data["nodes"]
    template = data.get("inner_template")
    by_size = sorted(outer_nodes, key=lambda n: n["metadata"]["extent_size"])
    outer_positions = _upright(
        {node["id"]: _math_coordinates(node["x"], node["y"]) for node in outer_nodes},
        by_size[-1]["id"],
        by_size[0]["id"],
    )

    # Scale outer coordinates so neighbouring circles never overlap.
    distances = [
        math.dist(p, q)
        for i, p in enumerate(outer_positions.values())
        for q in list(outer_positions.values())[i + 1:]
    ]
    bubble = 1.0
    spread = (2.6 * bubble / min(distances)) if distances and min(distances) > 0 else 1.0
    centers = {nid: (x * spread, y * spread) for nid, (x, y) in outer_positions.items()}

    # Outer edges, clipped at the circle boundaries.
    outer_segments = []
    for edge in data["edges"]:
        (x1, y1), (x2, y2) = centers[edge["source"]], centers[edge["target"]]
        length = math.hypot(x2 - x1, y2 - y1) or 1.0
        ux, uy = (x2 - x1) / length, (y2 - y1) / length
        outer_segments.append(
            ((x1 + ux * bubble, y1 + uy * bubble), (x2 - ux * bubble, y2 - uy * bubble))
        )
    ax.add_collection(LineCollection(outer_segments, colors="black", linewidths=1.6, zorder=1))

    inner_unit = _normalized_template(template) if template else {}
    inner_scale = 0.78 * bubble
    top_id = by_size[-1]["id"]

    for node in outer_nodes:
        cx, cy = centers[node["id"]]
        ax.add_patch(
            Circle((cx, cy), bubble, facecolor="white", edgecolor="black", linewidth=1.1, zorder=2)
        )

        if show_attribute_labels and node["new_attrs"]:
            ax.annotate(
                _join_labels(node["new_attrs"], max_label_items), (cx, cy + bubble),
                xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
                fontsize=font_size, color=_ATTRIBUTE_COLOR, bbox=_LABEL_BOX, zorder=5,
            )

        if not template:
            if show_object_labels and node["new_objects"]:
                ax.annotate(
                    _join_labels(node["new_objects"], max_label_items), (cx, cy - bubble),
                    xytext=(0, -3), textcoords="offset points", ha="center", va="top",
                    fontsize=font_size, color=_OBJECT_COLOR, bbox=_LABEL_BOX, zorder=5,
                )
            continue

        place = {
            iid: (cx + ux * inner_scale, cy + uy * inner_scale)
            for iid, (ux, uy) in inner_unit.items()
        }
        ax.add_collection(
            LineCollection(
                [(place[e["source"]], place[e["target"]]) for e in template["edges"]],
                colors=_LINE_COLOR, linewidths=0.5, zorder=3,
            )
        )

        filled = set(node["filled_inner"])
        for iid, (x, y) in place.items():
            ax.scatter(
                [x], [y], s=inner_node_size,
                facecolors=_FILLED_COLOR if iid in filled else "white",
                edgecolors=_FILLED_COLOR if iid in filled else _LINE_COLOR,
                linewidths=0.6, zorder=4,
            )

        if show_attribute_labels and node["id"] == top_id:
            for inode in template["nodes"]:
                if inode["new_attrs"]:
                    x, y = place[inode["id"]]
                    ax.annotate(
                        _join_labels(inode["new_attrs"], max_label_items), (x, y),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
                        fontsize=font_size * 0.8, color=_ATTRIBUTE_COLOR, bbox=_LABEL_BOX, zorder=5,
                    )

        if show_object_labels and node.get("child"):
            for cnode in node["child"]["nodes"]:
                if cnode["new_objects"]:
                    x, y = place[cnode["id"]]
                    ax.annotate(
                        _join_labels(cnode["new_objects"], max_label_items), (x, y),
                        xytext=(0, -3), textcoords="offset points", ha="center", va="top",
                        fontsize=font_size * 0.8, color=_OBJECT_COLOR, bbox=_LABEL_BOX, zorder=5,
                    )

    _set_limits(ax, list(centers.values()), pad=1.6 * bubble)

    if title:
        ax.set_title(title)

    return fig
