"""
Quality metrics for line diagrams (straight-line drawings of lattices).

Every metric takes a drawing and returns a score in [0, 1], where 1 is best,
following scikit-learn's ``*_score`` convention ("greater is better"):

- ``visual_chain_linearity_score``: chains run straight through vertices,
- ``node_node_conflict_score``: nodes keep apart from each other,
- ``node_edge_conflict_score``: nodes keep off edges they do not belong to,
- ``edge_edge_conflict_score``: edges at a vertex leave it at distinct angles,
- ``edge_crossing_score``: few of the crossings that could happen do,
- ``crossing_angle_score``: crossings are close to right angles,
- ``edge_length_uniformity_score``: edges of one direction have one length,
- ``layer_consistency_score``: the height of a node follows its rank,
- ``slope_economy_score``: no more edge directions than the order needs,
- ``slope_parallelism_score``: transposed cover edges are parallel,
- ``slope_standard_score``: edge slopes are standard ones such as 45°,
- ``slope_verticality_score``: no edge is close to horizontal,
- ``nested_suitability_score``: every node has room for a nested diagram,
- ``vertical_axis_balance_score``: left-right balance around the top.

``drawing_quality_scores`` computes all of them at once.

A drawing is a ``GraphData`` whose nodes all have coordinates (from
``layered_layout`` or ``lattice_to_graph_data_dimflux``, for example), or a
``networkx.DiGraph`` with ``x`` and ``y`` node attributes. An edge ``u -> v``
means ``u < v``, as in ``GraphData`` (source = lower concept), so it does not
matter whether ``y`` grows upwards or downwards. A ``GraphData`` is scored as
it is drawn: layouts such as DimFlux's, which the renderers turn upright,
are scored upright (the coordinates ``layout_positions`` returns); a graph
is scored with its coordinates as given.

Thresholds such as "too close" are relative to the average length of a cover
edge, so the scores do not depend on the scale or position of the drawing.
They do not change when it is mirrored either, except
``edge_length_uniformity_score``: its greedy grouping of directions can
group a mirrored drawing's edges slightly differently.
"""

from __future__ import annotations

from fractions import Fraction
from functools import cached_property
from numbers import Real

import networkx as nx
import numpy as np
from scipy.spatial import cKDTree
from sklearn.utils import check_scalar

from conceptflow.algorithms._bitset import iter_bits
from conceptflow.visualization.graph_data import GraphData, needs_rotation

# Slopes of the edges in a standard drawing: arctan(4/5), 45°, arctan(3/2), 90°.
DEFAULT_CANONICAL_ANGLES = (
    float(np.degrees(np.arctan(4 / 5))),
    45.0,
    float(np.degrees(np.arctan(3 / 2))),
    90.0,
)

# Cells per chunk of the pairwise (nodes x edges, edges x edges) computations.
_CHUNK_CELLS = 1 << 20

# Relative error above which the sign of a floating point orientation test is
# certain (Shewchuk's bound for orient2d is about 3.3e-16).
_ORIENTATION_ERROR = 1e-15


class _Drawing:
    """A validated drawing with the derived structures the metrics share."""

    def __init__(self, nodes: list, xy: np.ndarray, edges: list[tuple]):
        index = {node: i for i, node in enumerate(nodes)}
        graph = nx.DiGraph()
        graph.add_nodes_from(range(len(nodes)))
        graph.add_edges_from((index[u], index[v]) for u, v in edges)
        if not nx.is_directed_acyclic_graph(graph):
            raise ValueError(
                "The drawing must be acyclic: an edge u -> v means u < v in the order."
            )

        self.n = len(nodes)
        self.xy = xy
        self.graph = graph
        # Crossings and slopes are measured on the drawn edges, everything
        # else on the cover relation.
        self.edges = np.array(list(graph.edges), dtype=np.intp).reshape(-1, 2)
        self.cover = nx.transitive_reduction(graph)
        self.cover_edges = np.array(list(self.cover.edges), dtype=np.intp).reshape(-1, 2)

    @cached_property
    def down_sets(self) -> list[int]:
        """Bitsets of the elements below or equal to each node."""
        down = [0] * self.n
        for x in nx.topological_sort(self.graph):
            down[x] = 1 << x
            for p in self.graph.predecessors(x):
                down[x] |= down[p]
        return down

    @cached_property
    def up_sets(self) -> list[int]:
        """Bitsets of the elements above or equal to each node."""
        up = [0] * self.n
        for x in reversed(list(nx.topological_sort(self.graph))):
            up[x] = 1 << x
            for s in self.graph.successors(x):
                up[x] |= up[s]
        return up

    @cached_property
    def average_cover_edge_length(self) -> float:
        if len(self.cover_edges) == 0:
            return 1.0
        vectors = self.xy[self.cover_edges[:, 1]] - self.xy[self.cover_edges[:, 0]]
        return float(np.mean(np.hypot(vectors[:, 0], vectors[:, 1])))

    @cached_property
    def nearest_node_distances(self) -> np.ndarray:
        """Distance of every node to its nearest other node."""
        if self.n < 2:
            return np.full(self.n, np.inf)
        # k=2: the nearest point is the node itself.
        distances, _ = cKDTree(self.xy).query(self.xy, k=2)
        return distances[:, 1]

    @cached_property
    def nearest_edge_distances(self) -> np.ndarray:
        """
        Distance of every node to its nearest cover edge it is not an endpoint
        of (``inf`` if it is an endpoint of every cover edge).
        """
        distances = np.full(self.n, np.inf)
        if len(self.cover_edges) == 0:
            return distances

        lower, upper = self.cover_edges[:, 0], self.cover_edges[:, 1]
        a = self.xy[lower]
        ab = self.xy[upper] - a
        length_sq = np.einsum("ij,ij->i", ab, ab)
        degenerate = length_sq < 1e-18
        chunk = max(1, _CHUNK_CELLS // len(lower))

        for start in range(0, self.n, chunk):
            nodes = np.arange(start, min(start + chunk, self.n))
            ap = self.xy[nodes, None, :] - a[None, :, :]
            with np.errstate(divide="ignore", invalid="ignore"):
                t = np.clip(np.einsum("kij,ij->ki", ap, ab) / length_sq, 0.0, 1.0)
            t[:, degenerate] = 0.0
            offset = ap - t[:, :, None] * ab[None, :, :]
            d = np.hypot(offset[:, :, 0], offset[:, :, 1])
            incident = (lower[None, :] == nodes[:, None]) | (upper[None, :] == nodes[:, None])
            d[incident] = np.inf
            distances[nodes] = d.min(axis=1)

        return distances

    @cached_property
    def crossings(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Pairs ``(i, j)`` of drawn edges (indices into ``edges``) that intersect
        without sharing an endpoint, and whether each pair meets in a single
        point (``True``) or overlaps along a segment (``False``).
        """
        return _segment_crossings(self.xy, self.edges)


def _check_drawing(drawing) -> _Drawing:
    if isinstance(drawing, _Drawing):
        return drawing

    if isinstance(drawing, GraphData):
        missing = [node.node_id for node in drawing.nodes if node.x is None or node.y is None]
        if missing:
            raise ValueError(
                f"{len(missing)} node(s) have no coordinates, e.g. {missing[0]!r}. Lay the "
                "drawing out first, e.g. with layered_layout() or lattice_to_graph_data_dimflux()."
            )
        nodes = drawing.node_ids()
        if needs_rotation(drawing):
            # Score the drawing as it is shown: the renderers turn layouts
            # such as DimFlux's upright (top above bottom), so their stored
            # coordinates are not what a reader sees.
            from conceptflow.visualization.static import _lattice_positions

            _, positions = _lattice_positions(drawing)
            xy = np.array([positions[node] for node in nodes], dtype=float)
        else:
            xy = np.array([(node.x, node.y) for node in drawing.nodes], dtype=float)
        edges = [(edge.source, edge.target) for edge in drawing.edges]
    elif isinstance(drawing, nx.DiGraph):
        nodes = list(drawing.nodes)
        try:
            xy = np.array([(data["x"], data["y"]) for _, data in drawing.nodes(data=True)])
        except KeyError as exc:
            raise ValueError("Every node of the graph needs 'x' and 'y' attributes.") from exc
        xy = xy.astype(float).reshape(-1, 2)
        edges = list(drawing.edges)
    else:
        raise TypeError(
            "A drawing must be a GraphData with coordinates or a networkx.DiGraph with "
            f"'x' and 'y' node attributes, got {type(drawing).__name__}."
        )

    if not nodes:
        raise ValueError("The drawing has no nodes.")
    if not np.isfinite(xy).all():
        raise ValueError("Node coordinates contain NaN or infinity.")
    known = set(nodes)
    for u, v in edges:
        if u not in known or v not in known:
            raise ValueError(f"Edge ({u!r}, {v!r}) refers to an unknown node.")

    return _Drawing(nodes, xy, edges)


def _check_aggregate(aggregate: str) -> None:
    if aggregate not in ("mean", "min"):
        raise ValueError(f"aggregate must be 'mean' or 'min', got {aggregate!r}.")


def _conflict_score(distances, threshold: float, aggregate: str = "mean") -> float:
    """
    Aggregate of ``1 - penalty`` over ``distances``: the penalty is 0 at or
    above ``threshold`` and rises quadratically to 1 at distance 0, so near
    misses barely count but overlaps count fully.
    """
    distances = np.asarray(distances, dtype=float)
    if len(distances) == 0 or threshold <= 0:
        return 1.0
    factors = 1.0 - np.clip((threshold - distances) / threshold, 0.0, 1.0) ** 2
    return float(factors.min() if aggregate == "min" else factors.mean())


def _angles_deg(vectors: np.ndarray, others: np.ndarray) -> np.ndarray:
    """Unsigned angles (degrees, in [0, 180]) between all pairs of vectors."""
    norms = np.hypot(vectors[:, 0], vectors[:, 1])
    other_norms = np.hypot(others[:, 0], others[:, 1])
    cos = (vectors @ others.T) / np.outer(norms, other_norms)
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def _orientation_signs(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """
    Exact signs of the orientation of the triangles ``(a, b, c)``: +1 for
    counter-clockwise, -1 for clockwise, 0 for collinear.

    Uses floating point where the sign is certain and exact rationals
    elsewhere, so collinear points (common in drawings on a grid) are
    recognised exactly.
    """
    left = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
    right = (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    det = left - right
    signs = np.sign(det).astype(int)

    uncertain = np.abs(det) <= _ORIENTATION_ERROR * (np.abs(left) + np.abs(right))
    for k in np.flatnonzero(uncertain):
        ax, ay, bx, by, cx, cy = map(Fraction, (*a[k], *b[k], *c[k]))
        exact = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
        signs[k] = (exact > 0) - (exact < 0)
    return signs


def _segment_crossings(xy: np.ndarray, edges: np.ndarray):
    """
    All pairs of edges that intersect without sharing an endpoint, with
    exact predicates. Edges of (almost) zero length are ignored.
    """
    empty = np.empty(0, dtype=np.intp)
    start, end = xy[edges[:, 0]], xy[edges[:, 1]]
    keep = np.flatnonzero(np.hypot(*(end - start).T) > 1e-12) if len(edges) else empty
    if len(keep) < 2:
        return empty, empty, np.empty(0, dtype=bool)

    lo = np.minimum(start[keep], end[keep])
    hi = np.maximum(start[keep], end[keep])
    endpoints = edges[keep]
    chunk = max(1, _CHUNK_CELLS // len(keep))
    rows, cols = [], []

    # Candidates: bounding boxes overlap, no shared endpoint.
    for first in range(0, len(keep), chunk):
        block = slice(first, first + chunk)
        overlap = (
            (lo[block, None, 0] <= hi[None, :, 0])
            & (lo[None, :, 0] <= hi[block, None, 0])
            & (lo[block, None, 1] <= hi[None, :, 1])
            & (lo[None, :, 1] <= hi[block, None, 1])
        )
        shared = (endpoints[block, None, :, None] == endpoints[None, :, None, :]).any(axis=(2, 3))
        i, j = np.nonzero(overlap & ~shared)
        i += first
        later = j > i
        rows.append(i[later])
        cols.append(j[later])

    i, j = np.concatenate(rows), np.concatenate(cols)
    a, b = start[keep[i]], end[keep[i]]
    c, d = start[keep[j]], end[keep[j]]

    s1, s2 = _orientation_signs(a, b, c), _orientation_signs(a, b, d)
    s3, s4 = _orientation_signs(c, d, a), _orientation_signs(c, d, b)
    collinear = (s1 == 0) & (s2 == 0)
    meets_in_point = ~collinear & (s1 * s2 <= 0) & (s3 * s4 <= 0)

    # Collinear segments: compare their intervals on the line, measured along
    # x unless the line is vertical. Overlaps of positive length are not
    # proper crossings; touching in one point is.
    axis = np.where(a[:, 0] != b[:, 0], 0, 1)
    rows_ = np.arange(len(i))
    a_t, b_t, c_t, d_t = (p[rows_, axis] for p in (a, b, c, d))
    low = np.maximum(np.minimum(a_t, b_t), np.minimum(c_t, d_t))
    high = np.minimum(np.maximum(a_t, b_t), np.maximum(c_t, d_t))
    overlaps = collinear & (high > low)
    touches = collinear & (high == low)

    found = meets_in_point | overlaps | touches
    return keep[i[found]], keep[j[found]], ~overlaps[found]


def _group_within_tolerance(sorted_values: np.ndarray, tolerance: float) -> list[int]:
    """
    Sizes of the groups of a greedy left-to-right partition of sorted values
    into groups of diameter at most ``tolerance``. This uses the fewest
    groups possible, and unlike single linkage no group drifts arbitrarily
    far through a chain of small steps.
    """
    sizes = [1]
    first = sorted_values[0]
    for value in sorted_values[1:]:
        if value - first > tolerance:
            sizes.append(0)
            first = value
        sizes[-1] += 1
    return sizes


def _group_directions(directions: np.ndarray, tolerance: float, period: float) -> list:
    """
    Indices of the groups of directions (angles modulo ``period``) that lie
    within ``tolerance`` of each other. The circle is cut at the widest gap
    between consecutive directions, which no group can usefully span.
    """
    order = np.argsort(directions)
    ordered = directions[order]
    gaps = np.diff(np.append(ordered, ordered[0] + period))
    cut = int(np.argmax(gaps)) + 1
    order = np.concatenate([order[cut:], order[:cut]])
    unrolled = np.concatenate([ordered[cut:], ordered[:cut] + period])
    sizes = _group_within_tolerance(unrolled, tolerance)
    return np.split(order, np.cumsum(sizes)[:-1])


def _edge_vectors(drawing: _Drawing) -> np.ndarray:
    """Vectors of the non-degenerate drawn edges, from the upper to the lower end."""
    vectors = drawing.xy[drawing.edges[:, 0]] - drawing.xy[drawing.edges[:, 1]]
    return vectors[(vectors[:, 0] != 0) | (vectors[:, 1] != 0)]


def _unsigned_slopes(drawing: _Drawing) -> np.ndarray:
    """Angle of every drawn edge to the horizontal, in [0, 90]."""
    vectors = _edge_vectors(drawing)
    return np.degrees(np.arctan2(np.abs(vectors[:, 1]), np.abs(vectors[:, 0])))


def _signed_cover_directions(drawing: _Drawing) -> dict[tuple[int, int], float]:
    """Direction (degrees in [0, 360)) of every cover edge, from lower to upper."""
    directions = {}
    for u, v in drawing.cover_edges:
        dx, dy = drawing.xy[v] - drawing.xy[u]
        if dx != 0 or dy != 0:
            directions[(int(u), int(v))] = float(np.degrees(np.arctan2(dy, dx)) % 360.0)
    return directions


def _check_angle_tolerance(angle_tolerance) -> None:
    check_scalar(angle_tolerance, "angle_tolerance", Real, min_val=0, max_val=180)


def _check_threshold_factor(threshold_factor) -> None:
    check_scalar(
        threshold_factor, "threshold_factor", Real, min_val=0, include_boundaries="neither"
    )


def visual_chain_linearity_score(drawing) -> float:
    """
    How straight chains run through the vertices of the diagram.

    At every vertex with lower and upper covers, each neighbour on the side
    with fewer covers is matched with its straightest partner on the other
    side, scored ``angle / 180`` (1 for a straight line through the vertex).
    Only one partner is asked for: two lower covers lie in different
    directions, so they cannot both continue straight into the same upper
    cover. The score is the mean over all such neighbours, 1.0 if there is
    no vertex with both lower and upper covers.
    """
    drawing = _check_drawing(drawing)
    cover, xy = drawing.cover, drawing.xy
    straightness = []

    for v in cover.nodes:
        lower = xy[list(cover.predecessors(v))] - xy[v]
        upper = xy[list(cover.successors(v))] - xy[v]
        lower = lower[np.einsum("ij,ij->i", lower, lower) >= 1e-18]
        upper = upper[np.einsum("ij,ij->i", upper, upper) >= 1e-18]
        if len(lower) == 0 or len(upper) == 0:
            continue
        fewer, more = (lower, upper) if len(lower) <= len(upper) else (upper, lower)
        straightness.extend(_angles_deg(fewer, more).max(axis=1) / 180.0)

    if not straightness:
        return 1.0
    return float(np.mean(straightness))


def node_node_conflict_score(
    drawing,
    *,
    threshold_factor: float = 0.5,
    aggregate: str = "mean",
) -> float:
    """
    How well nodes keep apart from each other.

    Every node is scored by the distance ``d`` to its nearest other node,
    whether or not an edge joins them: ``1 - ((t - d) / t)^2`` below the
    threshold ``t = threshold_factor`` times the average cover edge length,
    1 above it. ``aggregate="mean"`` averages over the nodes;
    ``aggregate="min"`` returns the worst node. 1.0 for fewer than two nodes.
    """
    _check_threshold_factor(threshold_factor)
    _check_aggregate(aggregate)
    drawing = _check_drawing(drawing)
    if drawing.n < 2:
        return 1.0
    threshold = threshold_factor * drawing.average_cover_edge_length
    return _conflict_score(drawing.nearest_node_distances, threshold, aggregate)


def node_edge_conflict_score(
    drawing,
    *,
    threshold_factor: float = 0.5,
    aggregate: str = "mean",
) -> float:
    """
    How well nodes keep off edges they are not an endpoint of.

    Every node is scored by the distance to its nearest such cover edge, with
    the same penalty as ``node_node_conflict_score``. Taking the nearest edge
    per node, not all node-edge pairs, keeps a node lying on an edge from
    being diluted by the many edges far away from it. Nodes that are an
    endpoint of every cover edge are skipped. ``aggregate="min"`` returns
    the worst node instead of the mean.
    """
    _check_threshold_factor(threshold_factor)
    _check_aggregate(aggregate)
    drawing = _check_drawing(drawing)
    if len(drawing.cover_edges) == 0:
        return 1.0
    distances = drawing.nearest_edge_distances
    threshold = threshold_factor * drawing.average_cover_edge_length
    return _conflict_score(distances[np.isfinite(distances)], threshold, aggregate)


def edge_edge_conflict_score(
    drawing,
    *,
    angle_tolerance: float = 15.0,
    aggregate: str = "mean",
) -> float:
    """
    How distinctly the cover edges at a vertex leave it (angular resolution).

    Every vertex with at least two cover edges is scored by the smallest
    angle between two of them, with the penalty of
    ``node_node_conflict_score`` against ``angle_tolerance`` degrees. Edges
    a few degrees apart stay merged under the node marker for much of their
    length, hence the default of 15°; only vertices with more than 13 upper
    (or lower) covers are penalised however they are drawn.
    ``aggregate="min"`` returns the worst vertex instead of the mean.
    """
    _check_angle_tolerance(angle_tolerance)
    _check_aggregate(aggregate)
    drawing = _check_drawing(drawing)
    cover, xy = drawing.cover, drawing.xy
    smallest = []

    for v in cover.nodes:
        neighbours = list(cover.predecessors(v)) + list(cover.successors(v))
        vectors = xy[neighbours] - xy[v]
        vectors = vectors[np.einsum("ij,ij->i", vectors, vectors) >= 1e-18]
        if len(vectors) < 2:
            continue
        angles = _angles_deg(vectors, vectors)
        smallest.append(angles[np.triu_indices(len(vectors), k=1)].min())

    return _conflict_score(smallest, angle_tolerance, aggregate)


def edge_crossing_score(drawing) -> float:
    """
    ``1 - crossings / possible crossings``.

    Two edges can cross in some upward drawing only if all four endpoints are
    pairwise incomparable; other pairs are joined by monotone paths and never
    need to cross. Only those pairs are counted, so the score measures how
    much of the avoidable crossing budget the drawing spends. 1.0 if no pair
    of edges could cross (a chain, for example).
    """
    drawing = _check_drawing(drawing)
    edges = drawing.edges
    if len(edges) < 2:
        return 1.0

    down, up = drawing.down_sets, drawing.up_sets
    comparable = np.zeros((drawing.n, drawing.n), dtype=bool)
    for x in range(drawing.n):
        comparable[x, list(iter_bits(down[x] | up[x]))] = True

    lower, upper = edges[:, 0], edges[:, 1]
    possible = 0
    chunk = max(1, _CHUNK_CELLS // len(edges))
    for first in range(0, len(edges), chunk):
        block = slice(first, first + chunk)
        # Comparability is reflexive, so this also blocks shared endpoints.
        blocked = (
            comparable[np.ix_(lower[block], lower)]
            | comparable[np.ix_(lower[block], upper)]
            | comparable[np.ix_(upper[block], lower)]
            | comparable[np.ix_(upper[block], upper)]
        )
        # Count each unordered pair once: row k of the block is edge first + k.
        possible += int(np.triu(~blocked, k=first + 1).sum())
    if possible == 0:
        return 1.0

    # Crossings between comparable edges are not valid upward drawings at
    # all; they are left out of both counts.
    i, j, _ = drawing.crossings
    free = ~(
        comparable[lower[i], lower[j]]
        | comparable[lower[i], upper[j]]
        | comparable[upper[i], lower[j]]
        | comparable[upper[i], upper[j]]
    )
    return float(1.0 - int(free.sum()) / possible)


def _crossing_angles(drawing: _Drawing) -> np.ndarray:
    """Angle of every crossing folded to [0, 90]; 0 for overlapping edges."""
    i, j, proper = drawing.crossings
    edges, xy = drawing.edges, drawing.xy
    first = xy[edges[i, 1]] - xy[edges[i, 0]]
    second = xy[edges[j, 1]] - xy[edges[j, 0]]
    cos = np.einsum("ij,ij->i", first, second) / (
        np.hypot(first[:, 0], first[:, 1]) * np.hypot(second[:, 0], second[:, 1])
    )
    angles = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
    return np.where(proper, np.minimum(angles, 180.0 - angles), 0.0)


def crossing_angle_score(drawing, *, aggregate: str = "mean") -> float:
    """
    How close crossings are to right angles.

    Every crossing scores ``angle / 90`` for its angle folded to [0, 90], so
    1 for a right angle and 0 for (nearly) parallel edges; edges overlapping
    along a segment count as angle 0. ``aggregate="mean"`` averages over the
    crossings, ``aggregate="min"`` returns the sharpest one. 1.0 for a
    drawing without crossings.
    """
    _check_aggregate(aggregate)
    angles = _crossing_angles(_check_drawing(drawing))
    if len(angles) == 0:
        return 1.0
    terms = 1.0 - np.abs(90.0 - angles) / 90.0
    return float(terms.min() if aggregate == "min" else terms.mean())


def edge_length_uniformity_score(drawing, *, angle_tolerance: float = 5.0) -> float:
    """
    How uniform the lengths of edges with the same direction are.

    Once the heights of the nodes are fixed, an edge's length follows from
    its slope, so lengths are only compared among edges of (nearly) the same
    direction: groups of diameter ``angle_tolerance`` degrees, with left and
    right leaning edges apart. Each group scores ``1 - std / mean`` of its
    lengths (clipped to [0, 1]); the score is the average weighted by group
    size, so a single edge of an odd slope cannot drag it down. 1.0 for a
    drawing without edges.
    """
    _check_angle_tolerance(angle_tolerance)
    drawing = _check_drawing(drawing)
    vectors = _edge_vectors(drawing)
    if len(vectors) == 0:
        return 1.0

    directions = np.degrees(np.arctan2(vectors[:, 1], vectors[:, 0])) % 180.0
    lengths = np.hypot(vectors[:, 0], vectors[:, 1])
    scores, weights = [], []

    for group in _group_directions(directions, angle_tolerance, period=180.0):
        group_lengths = lengths[group]
        mean = float(np.mean(group_lengths))
        if len(group) <= 1 or mean < 1e-12:
            scores.append(1.0)
        else:
            scores.append(1.0 - np.clip(float(np.std(group_lengths)) / mean, 0.0, 1.0))
        weights.append(len(group))

    return float(np.average(scores, weights=weights))


def _freese_ranks(cover: nx.DiGraph) -> np.ndarray:
    """
    Freese's rank function, ``height(a) - depth(a) + M`` (Freese, "Automated
    Lattice Drawing", ICFCA 2004): longest chain down to a minimal element
    minus longest chain up to a maximal one, shifted so minimal elements of
    a longest chain get 0.
    """
    order = list(nx.topological_sort(cover))
    height = dict.fromkeys(order, 0)
    depth = dict.fromkeys(order, 0)
    for a in order:
        for b in cover.successors(a):
            height[b] = max(height[b], height[a] + 1)
    for a in reversed(order):
        for b in cover.predecessors(a):
            depth[b] = max(depth[b], depth[a] + 1)
    longest = max(height[a] + depth[a] for a in order)
    ranks = np.empty(len(order), dtype=float)
    for a in order:
        ranks[a] = height[a] - depth[a] + longest
    return ranks


def layer_consistency_score(drawing) -> float:
    """
    How well the height of the nodes follows their rank.

    The R² of the least-squares fit ``y ~ a * rank + b`` with Freese's rank
    function: 1 when equal ranks are at equal heights and the layers are
    spaced in proportion to their rank difference. 1.0 if all nodes have the
    same rank, 0.0 if they are drawn at one height although they have not.
    """
    drawing = _check_drawing(drawing)
    ranks = _freese_ranks(drawing.cover)
    heights = drawing.xy[:, 1]
    if np.ptp(ranks) < 1e-12:
        return 1.0
    if np.ptp(heights) < 1e-12:
        return 0.0
    return float(np.corrcoef(ranks, heights)[0, 1] ** 2)


def slope_economy_score(drawing, *, angle_tolerance: float = 1.0) -> float:
    """
    ``k_min / k``: the fewest edge directions the order allows over the
    number of directions the drawing uses.

    ``k_min`` is the largest number of upper or lower covers of one element,
    which must all point in different directions; distributive lattices
    reach it with additive drawings. Directions are signed (left and right
    leaning edges differ) and grouped with diameter ``angle_tolerance``
    degrees. Clipped at 1.0; 1.0 for a drawing without cover edges.
    """
    _check_angle_tolerance(angle_tolerance)
    drawing = _check_drawing(drawing)
    directions = _signed_cover_directions(drawing)
    if not directions:
        return 1.0

    used = len(_group_directions(np.array(list(directions.values())), angle_tolerance, 360.0))
    cover = drawing.cover
    needed = max(max(cover.in_degree(x), cover.out_degree(x)) for x in cover.nodes)
    return float(min(1.0, needed / used))


def _transposed_cover_pairs(drawing: _Drawing) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    """
    Pairs of cover edges ``(u, v), (w, t)`` where ``u -> v`` transposes up to
    ``w -> t``: ``v ∧ w = u`` and ``v ∨ w = t``, so both are the same step at
    different heights (opposite sides of a covering square, for example).
    Pairs whose meet or join does not exist are skipped.
    """
    down, up = drawing.down_sets, drawing.up_sets
    by_down = {mask: x for x, mask in enumerate(down)}
    by_up = {mask: x for x, mask in enumerate(up)}
    cover = drawing.cover
    pairs = []

    for u, v in drawing.cover_edges:
        u, v = int(u), int(v)
        for w in iter_bits(up[u] & ~(1 << u)):
            if by_down.get(down[v] & down[w]) != u:
                continue
            top = by_up.get(up[v] & up[w])
            if top is not None and cover.has_edge(w, top):
                pairs.append(((u, v), (w, top)))
    return pairs


def slope_parallelism_score(drawing, *, angle_tolerance: float = 1.0) -> float:
    """
    Share of transposed pairs of cover edges drawn parallel.

    Generalises the parallelogram rule of additive drawings: if ``u -> v``
    transposes up to ``w -> t`` (``v ∧ w = u``, ``v ∨ w = t``), the upper
    edge adds what the lower one does and should point the same way, within
    ``angle_tolerance`` degrees. Only directions are compared. Additive
    drawings of distributive lattices score 1.0; for other lattices 1.0 may
    be out of reach (in M3, ``0 -> a`` transposes to both ``b -> 1`` and
    ``c -> 1``), so compare drawings of the same lattice. 1.0 if there are no
    transposed pairs.
    """
    _check_angle_tolerance(angle_tolerance)
    drawing = _check_drawing(drawing)
    pairs = _transposed_cover_pairs(drawing)
    if not pairs:
        return 1.0
    directions = _signed_cover_directions(drawing)

    def parallel(e, f) -> bool:
        if e not in directions or f not in directions:
            return False  # a zero-length edge has no direction
        difference = abs(directions[e] - directions[f]) % 360.0
        return min(difference, 360.0 - difference) <= angle_tolerance

    return sum(parallel(e, f) for e, f in pairs) / len(pairs)


def slope_standard_score(drawing, *, canonical_angles=DEFAULT_CANONICAL_ANGLES) -> float:
    """
    How close edge slopes are to standard ones.

    ``1 - mean deviation / largest possible deviation``, measuring each
    edge's angle to the horizontal against the nearest of
    ``canonical_angles`` (degrees in [0, 90]). With the defaults (arctan(4/5),
    45°, arctan(3/2), 90°) a horizontal edge is the worst case. 1.0 for a
    drawing without edges.
    """
    canonical = np.sort(np.asarray(canonical_angles, dtype=float).ravel())
    if len(canonical) == 0 or not ((canonical >= 0) & (canonical <= 90)).all():
        raise ValueError("canonical_angles must be a non-empty collection of angles in [0, 90].")
    drawing = _check_drawing(drawing)
    slopes = _unsigned_slopes(drawing)
    if len(slopes) == 0:
        return 1.0

    worst = max(float(canonical[0]), float(90.0 - canonical[-1]), *(np.diff(canonical) / 2.0))
    if worst == 0:
        return 1.0  # every angle is canonical
    deviation = np.abs(slopes[:, None] - canonical[None, :]).min(axis=1)
    return float(np.clip(1.0 - float(np.mean(deviation)) / worst, 0.0, 1.0))


def slope_verticality_score(drawing) -> float:
    """
    ``sin(2 * min(θ, 45°))`` for the smallest angle ``θ`` of an edge to the
    horizontal.

    1 when every edge is at least 45° steep, 0 when one is horizontal: the
    vertical position signals which end of an edge is the upper one. The
    score rises quickly (a 30° edge scores about 0.87), so only edges close
    to horizontal are penalised. 1.0 for a drawing without edges.
    """
    slopes = _unsigned_slopes(_check_drawing(drawing))
    if len(slopes) == 0:
        return 1.0
    return float(np.sin(np.radians(2.0 * min(float(slopes.min()), 45.0))))


def bottleneck_clearance_radius(drawing) -> float:
    """
    The largest radius of discs around all nodes at once that neither
    overlap each other nor cross a cover edge the node is not an endpoint
    of: the minimum over the nodes of ``min(d_node / 2, d_edge)``. ``inf``
    if there is nothing to collide with.
    """
    drawing = _check_drawing(drawing)
    return float(
        np.min(np.minimum(drawing.nearest_node_distances / 2.0, drawing.nearest_edge_distances))
    )


def nested_suitability_score(drawing, *, r_min_nest_factor: float = 0.5) -> float:
    """
    How much room every node has for a nested element (an inner diagram).

    Scores ``bottleneck_clearance_radius`` against ``r_min_nest_factor``
    times the average cover edge length, with the penalty of
    ``node_node_conflict_score``: 1 when there is that much room around
    every node, 0 when two nodes coincide or a node lies on an edge. Only
    the tightest node counts, since a nested element needs the same room
    everywhere. With the default of 1/2, a perfect score needs every cover
    edge to be at least as long as the average.
    """
    check_scalar(
        r_min_nest_factor, "r_min_nest_factor", Real, min_val=0, include_boundaries="neither"
    )
    drawing = _check_drawing(drawing)
    radius = bottleneck_clearance_radius(drawing)
    return _conflict_score([radius], r_min_nest_factor * drawing.average_cover_edge_length)


def _ratio_balance(left: float, right: float) -> float:
    total = left + right
    if total <= 1e-12:
        return 1.0
    return float(1.0 - abs(left - right) / total)


def vertical_axis_balance_score(drawing) -> float:
    """
    Left-right balance around the vertical line through the top element.

    The product of three parts: about as many nodes left of the axis as right
    of it; about the same total horizontal distance from the axis on both
    sides; and the bottom element on the axis, with full penalty at one
    average cover edge length away. The poset need not be symmetric.

    Needs a unique top and bottom element, so not iceberg lattices.
    """
    drawing = _check_drawing(drawing)
    graph = drawing.graph
    tops = [x for x in graph.nodes if graph.out_degree(x) == 0]
    bottoms = [x for x in graph.nodes if graph.in_degree(x) == 0]
    if len(tops) != 1 or len(bottoms) != 1:
        raise ValueError(
            "vertical_axis_balance_score needs a unique top and bottom element, "
            f"found {len(tops)} maximal and {len(bottoms)} minimal elements."
        )
    top, bottom = tops[0], bottoms[0]

    x = drawing.xy[:, 0]
    offsets = np.delete(x - x[top], top)
    count_balance = _ratio_balance(int((offsets < 0).sum()), int((offsets > 0).sum()))
    mass_balance = _ratio_balance(
        float(-offsets[offsets < 0].sum()), float(offsets[offsets > 0].sum())
    )

    bottom_offset = abs(float(x[bottom] - x[top]))
    tolerance = drawing.average_cover_edge_length
    if tolerance > 0:
        bottom_alignment = 1.0 - min(bottom_offset / tolerance, 1.0)
    else:
        bottom_alignment = 1.0 if bottom_offset == 0 else 0.0

    return float(count_balance * mass_balance * bottom_alignment)


_SCORES = {
    "visual_chain_linearity": visual_chain_linearity_score,
    "node_node_conflict": node_node_conflict_score,
    "node_edge_conflict": node_edge_conflict_score,
    "edge_edge_conflict": edge_edge_conflict_score,
    "edge_crossing": edge_crossing_score,
    "crossing_angle": crossing_angle_score,
    "edge_length_uniformity": edge_length_uniformity_score,
    "layer_consistency": layer_consistency_score,
    "slope_economy": slope_economy_score,
    "slope_parallelism": slope_parallelism_score,
    "slope_standard": slope_standard_score,
    "slope_verticality": slope_verticality_score,
    "nested_suitability": nested_suitability_score,
    "vertical_axis_balance": vertical_axis_balance_score,
}

# Scores that average over nodes, vertices or crossings, and so also have a
# worst case.
_WORST_CASE_SCORES = (
    "node_node_conflict",
    "node_edge_conflict",
    "edge_edge_conflict",
    "crossing_angle",
)


def drawing_quality_scores(drawing, *, worst_case: bool = False) -> dict[str, float]:
    """
    All drawing quality metrics of ``drawing`` with default parameters.

    Returns a dict from metric name (the function name without ``_score``)
    to score. With ``worst_case=True`` it also has ``"<name>_min"`` for the
    metrics that average over nodes, vertices or crossings: the worst single
    term, on the same scale. The structures the metrics share are computed
    only once.
    """
    drawing = _check_drawing(drawing)
    scores = {name: metric(drawing) for name, metric in _SCORES.items()}
    if worst_case:
        for name in _WORST_CASE_SCORES:
            scores[f"{name}_min"] = _SCORES[name](drawing, aggregate="min")
    return scores
