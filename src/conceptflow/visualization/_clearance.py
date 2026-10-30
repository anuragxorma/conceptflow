"""
Keep edges off the nodes they do not connect.

In a line diagram an edge drawn through (or right past) a third node reads as
two cover relations through that node. Layouts produce this easily: the
layered layout draws edges spanning several rows straight across the grid,
and DimFlux can place three or more concepts on one line.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from scipy.spatial import cKDTree

from conceptflow.visualization.graph_data import GraphData

# How close an edge may come to a node it does not connect, as a fraction of
# the smallest distance between two nodes. Interactive diagrams draw nodes
# with a radius of 0.4 of that distance, so 0.45 keeps edges off the circles.
CLEARANCE = 0.45
MAX_ROUNDS = 30
MAX_NODES = 2000  # above this a line diagram is too dense to read node by node


def _up_axis(graph_data: GraphData, xy: np.ndarray) -> np.ndarray:
    """Unit vector from the bottom concept to the top concept."""
    sizes = [node.metadata.get("extent_size", 0) for node in graph_data.nodes]
    top, bottom = int(np.argmax(sizes)), int(np.argmin(sizes))
    up = xy[top] - xy[bottom]
    norm = np.hypot(*up)
    return up / norm if norm > 0 else np.array([0.0, -1.0])


def _try_move(k, target, height, across, unit) -> bool:
    """
    Move node ``k`` sideways to ``target`` unless it would come closer than
    ``unit`` to another node. (Pushing the rest of its row along to make room
    was tried: the pushed nodes drag their edges across others and it gets
    worse.)
    """
    distance = np.hypot(across - target, height - height[k])
    distance[k] = np.inf
    if distance.min() < unit * 0.9999:
        return False
    across[k] = target
    return True


def keep_edges_off_nodes(graph_data: GraphData, clearance: float = CLEARANCE) -> GraphData:
    """
    Move nodes sideways until no edge passes closer than ``clearance`` times
    the smallest node distance to a node it does not connect.

    "Sideways" is perpendicular to the line from the bottom to the top
    concept, so every concept keeps its height and edges keep pointing up. A
    node is only moved where it keeps at least the smallest node distance to
    every other node, so the drawing does not get denser. Lattices above
    ``MAX_NODES`` concepts are returned unchanged.
    """
    nodes = graph_data.nodes
    n = len(nodes)
    if n < 3 or n > MAX_NODES or any(node.x is None or node.y is None for node in nodes):
        return graph_data

    index = {node.node_id: i for i, node in enumerate(nodes)}
    xy = np.array([(node.x, node.y) for node in nodes], dtype=float)
    up = _up_axis(graph_data, xy)
    side = np.array([-up[1], up[0]])
    height = xy @ up
    across = xy @ side

    unit = cKDTree(xy).query(xy, k=2)[0][:, 1].min()
    if not np.isfinite(unit) or unit <= 0:
        return graph_data
    limit = clearance * unit
    # Heights never change, so the nodes between two heights are a slice.
    order = np.argsort(height)
    sorted_height = height[order]

    edges = [(index[e.source], index[e.target]) for e in graph_data.edges]
    moved_any = False
    for _ in range(MAX_ROUNDS):
        moved = False
        for a, b in edges:
            low, high = sorted((height[a], height[b]))
            between = order[
                np.searchsorted(sorted_height, low, "right"):
                np.searchsorted(sorted_height, high, "left")
            ]
            if len(between) == 0:
                continue
            d_across = across[b] - across[a]
            d_height = height[b] - height[a]
            length = np.hypot(d_across, d_height)
            # Sideways offset from the edge at each node's height, and the
            # perpendicular distance it amounts to.
            on_edge = across[a] + (height[between] - height[a]) / d_height * d_across
            offset = across[between] - on_edge
            too_close = np.abs(offset) * abs(d_height) / length < limit
            needed = limit * length / abs(d_height) * 1.0001
            for k, edge_at_k, off in zip(
                between[too_close], on_edge[too_close], offset[too_close], strict=True
            ):
                first = 1.0 if off >= 0 else -1.0
                for direction in (first, -first):
                    if _try_move(k, edge_at_k + direction * needed, height, across, unit):
                        moved = moved_any = True
                        break
        if not moved:
            break

    if not moved_any:
        return graph_data
    new_xy = np.outer(height, up) + np.outer(across, side)
    return replace(
        graph_data,
        nodes=[
            replace(node, x=float(new_xy[i, 0]), y=float(new_xy[i, 1]))
            for i, node in enumerate(nodes)
        ],
    )
