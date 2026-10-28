"""
Large induced bipartite subgraphs: which vertices to delete so that a graph
becomes bipartite.

From Dürrschnabel, Hanika and Stumme, "Discovering Locally Maximal
Bipartite Subgraphs" (arXiv:2211.10446, 2022), used by Ord2Factor:

- ``minimum_bipartite_deletion``: a smallest such set (the paper's SAT
  formulation, Definition 1, solved with MiniSat 2.2; exponential in the
  worst case, as the problem is NP-hard);
- ``greedy_bipartite_deletion``: a locally minimal set (Algorithm 1);
- ``annealing_bipartite_deletion``: a locally minimal set improved by
  simulated annealing (Algorithm 2).

"Locally minimal" is the guarantee of the paper's ``greedy-fill``: every
deleted vertex has a neighbour on both sides of the bipartition that was
built, so it cannot be added to that bipartition. (The paper calls the
result inclusion-minimal; occasionally a deleted vertex could be kept with a
*different* 2-colouring, which the algorithm does not search for.)

Graphs are dicts mapping each vertex to the set of its neighbours, as
returned by ``incompatibility_graph``. Each function returns the set D of
vertices to delete.
"""

from __future__ import annotations

import math
from collections.abc import Hashable

from sklearn.utils import check_random_state

Graph = dict[Hashable, set]


def is_bipartite(graph: Graph, deleted=frozenset()) -> bool:
    """Whether the graph without the ``deleted`` vertices is bipartite."""
    color: dict = {}
    for start in graph:
        if start in deleted or start in color:
            continue
        color[start] = 0
        stack = [start]
        while stack:
            vertex = stack.pop()
            for neighbour in graph[vertex]:
                if neighbour in deleted:
                    continue
                if neighbour not in color:
                    color[neighbour] = 1 - color[vertex]
                    stack.append(neighbour)
                elif color[neighbour] == color[vertex]:
                    return False
    return True


# ---------------------------------------------------------------------------
# Exact: SAT (Definition 1)
# ---------------------------------------------------------------------------


def _deletable_with_at_most(graph: Graph, vertices: list, k: int):
    """D with |D| <= k making the graph bipartite, or None (MiniSat 2.2)."""
    from pysat.card import CardEnc, EncType
    from pysat.solvers import Minisat22

    index = {vertex: i for i, vertex in enumerate(vertices)}

    def var(i: int, part: int) -> int:  # V_{i,1}, V_{i,2}, V_{i,3}
        return 3 * i + part

    # (i) every vertex is in A, B or D.
    clauses = [[var(i, 1), var(i, 2), var(i, 3)] for i in range(len(vertices))]
    # (ii) no edge inside A or inside B.
    for vertex in vertices:
        i = index[vertex]
        for neighbour in graph[vertex]:
            j = index[neighbour]
            if i < j:
                clauses.append([-var(i, 1), -var(j, 1)])
                clauses.append([-var(i, 2), -var(j, 2)])
    # At most k vertices in D (Sinz's sequential counter).
    at_most = CardEnc.atmost(
        lits=[var(i, 3) for i in range(len(vertices))],
        bound=k,
        top_id=3 * len(vertices),
        encoding=EncType.seqcounter,
    )
    with Minisat22(bootstrap_with=clauses + at_most.clauses) as solver:
        if not solver.solve():
            return None
        true = {literal for literal in solver.get_model() if literal > 0}
    return {vertex for vertex in vertices if var(index[vertex], 3) in true}


def minimum_bipartite_deletion(graph: Graph) -> set:
    """
    A smallest set of vertices whose deletion makes the graph bipartite.

    The paper's SAT formulation: variables ``V_i1, V_i2, V_i3`` put vertex
    ``i`` into A, B or D; adjacent vertices are not both in A or both in B;
    at most ``k`` vertices are in D. The smallest ``k`` is found as in the
    paper: ``k`` is doubled until the instance is satisfiable, then the gap
    is closed by binary search.
    """
    vertices = sorted(graph, key=repr)
    if is_bipartite(graph):
        return set()

    # k = 0 is unsatisfiable (the graph is not bipartite).
    low, k = 0, 1
    best = _deletable_with_at_most(graph, vertices, k)
    while best is None:
        low, k = k, 2 * k
        best = _deletable_with_at_most(graph, vertices, k)
    high = k
    while high - low > 1:
        middle = (low + high) // 2
        found = _deletable_with_at_most(graph, vertices, middle)
        if found is None:
            low = middle
        else:
            high, best = middle, found
    return best


# ---------------------------------------------------------------------------
# Heuristics: greedy (Algorithm 1), simulated annealing (Algorithm 2)
# ---------------------------------------------------------------------------


def _greedy_fill(graph: Graph, a: set, b: set, d: set, rng) -> tuple[set, set, set]:
    """
    ``greedy-fill``: move the vertices of D, in random order, into A or B
    (chosen at random among those without a neighbour of the vertex). If A
    and B are both empty, the vertex goes to A.
    """
    candidates = sorted(d, key=repr)
    for position in rng.permutation(len(candidates)).tolist():
        vertex = candidates[position]
        options = [side for side in (a, b) if not graph[vertex] & side]
        if not options:
            continue
        d.discard(vertex)
        if not a and not b:
            a.add(vertex)
        else:
            options[rng.randint(len(options))].add(vertex)
    return a, b, d


def greedy_bipartite_deletion(graph: Graph, random_state=None) -> set:
    """
    A locally minimal set of vertices whose deletion makes the graph
    bipartite (the paper's Algorithm 1: ``greedy-fill`` starting with every
    vertex in D). Polynomial time.
    """
    rng = check_random_state(random_state)
    _, _, d = _greedy_fill(graph, set(), set(), set(graph), rng)
    return d


def annealing_bipartite_deletion(
    graph: Graph,
    n_iterations: int = 1000,
    temperature: float = 1.0,
    random_state=None,
) -> set:
    """
    Simulated annealing (the paper's Algorithm 2), starting from the greedy
    solution. A neighbouring solution moves a random vertex ``u`` of D into
    A or B (at random), moves its neighbours there into D, and refills with
    ``greedy-fill``. A smaller D is always accepted, a larger one with
    probability ``exp(-(growth) / t)``, where the temperature falls linearly
    from ``temperature`` to 0 over ``n_iterations`` steps. As in the paper,
    the final D is returned; it is locally minimal.
    """
    rng = check_random_state(random_state)
    a, b, d = _greedy_fill(graph, set(), set(), set(graph), rng)

    for i in range(n_iterations):
        if not d:
            break
        candidates = sorted(d, key=repr)
        u = candidates[rng.randint(len(candidates))]
        new_a, new_b, new_d = set(a), set(b), set(d)
        for neighbour in graph[u]:
            if neighbour in new_a or neighbour in new_b:
                new_a.discard(neighbour)
                new_b.discard(neighbour)
                new_d.add(neighbour)
        new_d.discard(u)
        (new_a if rng.randint(2) == 0 else new_b).add(u)
        new_a, new_b, new_d = _greedy_fill(graph, new_a, new_b, new_d, rng)

        change = len(d) - len(new_d)
        current_temperature = temperature * (1 - i / n_iterations)
        if change > 0 or (
            current_temperature > 0 and math.exp(change / current_temperature) > rng.random_sample()
        ):
            a, b, d = new_a, new_b, new_d
    return d
