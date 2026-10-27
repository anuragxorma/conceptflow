"""
Fast layered ("Sugiyama-style") layout for concept lattices.

Concepts are placed in rows by their longest-path distance from the top
concept, and the order within every row is improved by median sweeps over a
graph whose long edges are split by dummy nodes (Sugiyama's framework),
which reduces edge crossings. Much faster than DimFlux on large lattices,
though lattices are rarely planar, so some crossings remain.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import replace

from conceptflow.visualization.graph_data import GraphData


def _levels(
    graph_data: GraphData,
) -> tuple[dict[str, int], dict[str, list[str]], dict[str, list[str]]]:
    """Longest-path level from the top for every node, iteratively."""
    upper: dict[str, list[str]] = {node.node_id: [] for node in graph_data.nodes}
    lower: dict[str, list[str]] = {node.node_id: [] for node in graph_data.nodes}
    for edge in graph_data.edges:
        upper[edge.source].append(edge.target)
        lower[edge.target].append(edge.source)

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

    return level, upper, lower


def count_crossings(graph_data: GraphData) -> int:
    """
    Number of pairwise crossings of the straight edges of a laid-out graph.

    Quadratic in the number of edges; meant for comparing layouts.
    """
    position = {node.node_id: (node.x, node.y) for node in graph_data.nodes}
    segments = [
        (position[e.source], position[e.target], e.source, e.target)
        for e in graph_data.edges
    ]

    def orientation(p, q, r):
        value = (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
        return (value > 1e-12) - (value < -1e-12)

    crossings = 0
    for i, (a1, a2, s1, t1) in enumerate(segments):
        for b1, b2, s2, t2 in segments[i + 1 :]:
            if {s1, t1} & {s2, t2}:
                continue  # edges sharing a node do not cross
            if (
                orientation(a1, a2, b1) * orientation(a1, a2, b2) < 0
                and orientation(b1, b2, a1) * orientation(b1, b2, a2) < 0
            ):
                crossings += 1
    return crossings


def _gap_crossings(pairs: list[tuple[int, int]]) -> int:
    """
    Crossings between two adjacent rows: inversions among the
    ``(upper position, lower position)`` pairs of their edges.
    """
    pairs.sort()
    seen: list[int] = []
    crossings = 0
    for _, low in pairs:
        i = bisect_right(seen, low)
        crossings += len(seen) - i
        seen.insert(i, low)
    return crossings


def layered_layout(
    graph_data: GraphData,
    sweeps: int | None = None,
    x_spacing: float = 60.0,
    y_spacing: float | None = None,
) -> GraphData:
    """
    Return ``graph_data`` with layered-layout coordinates.

    ``y_spacing`` (distance between rows) defaults to at least 80 and grows
    with the widest row, so large lattices come out roughly 4:3 instead of
    as a flat strip.

    Steps (Sugiyama's framework):

    1. rows by longest-path distance from the top concept;
    2. edges skipping rows get invisible "dummy" nodes, so every edge links
       adjacent rows and takes part in the ordering;
    3. ``sweeps`` down-and-up passes order every row by the median position
       of its neighbours in the row before; the order with the fewest
       crossings (counted exactly between adjacent rows) is kept;
    4. nodes are spaced evenly within their row, dummies keeping their slot.

    Coordinates follow the GraphData convention used by the renderers
    (screen coordinates: the top concept has the smallest ``y``).

    ``sweeps`` defaults to 12, or 3 above 10,000 concepts, where each sweep
    costs more and the drawing is too dense to read closely anyway.
    """
    if sweeps is None:
        sweeps = 12 if len(graph_data.nodes) <= 10_000 else 3

    level, upper, lower = _levels(graph_data)
    up = {node_id: list(ns) for node_id, ns in upper.items()}
    down = {node_id: list(ns) for node_id, ns in lower.items()}

    for edge in graph_data.edges:
        source, target = edge.source, edge.target  # target is the upper concept
        span = level[source] - level[target]
        if span <= 1:
            continue
        up[source].remove(target)
        down[target].remove(source)
        previous = source
        for k in range(1, span):
            dummy = ("dummy", source, target, k)
            level[dummy] = level[source] - k
            up[previous].append(dummy)
            down[dummy] = [previous]
            up[dummy] = []
            previous = dummy
        up[previous].append(target)
        down[target].append(previous)

    rows: dict[int, list] = {}
    for node_id, row_index in level.items():
        rows.setdefault(row_index, []).append(node_id)
    depth = sorted(rows)
    position = {node_id: i for r in depth for i, node_id in enumerate(rows[r])}

    def total_crossings() -> int:
        return sum(
            _gap_crossings([(position[t], position[s]) for s in rows[r + 1] for t in up[s]])
            for r in depth[:-1]
        )

    def reorder(row_index: int, neighbours: dict) -> None:
        def median(node_id) -> float:
            placed = sorted(position[n] for n in neighbours[node_id])
            if not placed:
                return position[node_id]
            k = len(placed)
            return placed[k // 2] if k % 2 else (placed[k // 2 - 1] + placed[k // 2]) / 2

        row = rows[row_index]
        row.sort(key=lambda node_id: (median(node_id), position[node_id]))
        for i, node_id in enumerate(row):
            position[node_id] = i

    best = (total_crossings(), {r: list(rows[r]) for r in depth})
    for _ in range(sweeps):
        for row_index in depth[1:]:
            reorder(row_index, up)
        for row_index in reversed(depth[:-1]):
            reorder(row_index, down)
        crossings = total_crossings()
        if crossings < best[0]:
            best = (crossings, {r: list(rows[r]) for r in depth})

    x: dict = {}
    for row_index, row in best[1].items():
        for i, node_id in enumerate(row):
            x[node_id] = i - (len(row) - 1) / 2

    if y_spacing is None:
        width = max(len(row) for row in best[1].values()) * x_spacing
        y_spacing = max(80.0, 0.75 * width / max(1, len(depth) - 1))

    nodes = [
        replace(
            node,
            x=x[node.node_id] * x_spacing,
            y=level[node.node_id] * y_spacing,
            metadata={**node.metadata, "layout": "layered"},
        )
        for node in graph_data.nodes
    ]

    return GraphData(
        nodes=nodes,
        edges=graph_data.edges,
        metadata={**graph_data.metadata, "layout": "layered"},
    )
