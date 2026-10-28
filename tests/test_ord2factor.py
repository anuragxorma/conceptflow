"""
Ord2Factor (Dürrschnabel & Stumme 2023, Algorithm 2) and the bipartite
subgraph methods it uses (Dürrschnabel, Hanika & Stumme 2022).
"""

from itertools import combinations

import numpy as np
import pytest

from conceptflow import FormalContext
from conceptflow.decomposition import (
    ExactOrdinalTwoFactorizer,
    Ord2Factor,
    incidence_pairs,
    incompatibility_graph,
    is_ferrers_relation,
)
from conceptflow.decomposition.bipartite import (
    _greedy_fill,
    annealing_bipartite_deletion,
    greedy_bipartite_deletion,
    is_bipartite,
    minimum_bipartite_deletion,
)


def _random_graph(seed, max_vertices=9):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(1, max_vertices))
    p = rng.uniform(0.2, 0.7)
    graph = {v: set() for v in range(n)}
    for u, v in combinations(range(n), 2):
        if rng.random() < p:
            graph[u].add(v)
            graph[v].add(u)
    return graph


def _smallest_deletion_size(graph):
    vertices = list(graph)
    for size in range(len(vertices) + 1):
        for deleted in combinations(vertices, size):
            if is_bipartite(graph, frozenset(deleted)):
                return size
    raise AssertionError("unreachable")


# ---------------------------------------------------------------------------
# Bipartite subgraphs
# ---------------------------------------------------------------------------


def test_is_bipartite():
    triangle = {0: {1, 2}, 1: {0, 2}, 2: {0, 1}}
    square = {0: {1, 3}, 1: {0, 2}, 2: {1, 3}, 3: {0, 2}}
    assert not is_bipartite(triangle)
    assert is_bipartite(square)
    assert is_bipartite(triangle, deleted=frozenset({0}))


@pytest.mark.parametrize("seed", range(60))
def test_sat_deletion_is_minimum(seed):
    graph = _random_graph(seed)
    deleted = minimum_bipartite_deletion(graph)

    assert is_bipartite(graph, frozenset(deleted))
    assert len(deleted) == _smallest_deletion_size(graph)


@pytest.mark.parametrize("method", ["greedy", "annealing"])
@pytest.mark.parametrize("seed", range(40))
def test_heuristic_deletions_make_the_graph_bipartite(seed, method):
    graph = _random_graph(seed, max_vertices=14)
    if method == "greedy":
        deleted = greedy_bipartite_deletion(graph, random_state=seed)
    else:
        deleted = annealing_bipartite_deletion(graph, n_iterations=200, random_state=seed)

    assert is_bipartite(graph, frozenset(deleted))
    assert len(deleted) >= len(minimum_bipartite_deletion(graph))


@pytest.mark.parametrize("seed", range(40))
def test_greedy_fill_leaves_only_vertices_conflicting_with_both_sides(seed):
    graph = _random_graph(seed, max_vertices=14)
    a, b, d = _greedy_fill(graph, set(), set(), set(graph), np.random.RandomState(seed))

    assert a | b | d == set(graph) and not (a & b or a & d or b & d)
    assert all(not graph[v] & a for v in a) and all(not graph[v] & b for v in b)
    for vertex in d:
        assert graph[vertex] & a and graph[vertex] & b


def test_heuristics_are_reproducible():
    graph = _random_graph(3, max_vertices=14)
    assert greedy_bipartite_deletion(graph, random_state=1) == (
        greedy_bipartite_deletion(graph, random_state=1)
    )
    assert annealing_bipartite_deletion(graph, random_state=1) == (
        annealing_bipartite_deletion(graph, random_state=1)
    )


# ---------------------------------------------------------------------------
# Ord2Factor
# ---------------------------------------------------------------------------


def _two_factorizable(seed):
    # The complement of a grid is two-factorizable (order dimension two).
    rng = np.random.default_rng(seed)
    rows, columns = int(rng.integers(2, 5)), int(rng.integers(2, 5))
    complement = [
        [i >= a for a in range(1, rows)] + [j >= b for b in range(1, columns)]
        for i in range(rows)
        for j in range(columns)
    ]
    return 1 - np.array(complement, dtype=int)


@pytest.mark.parametrize("method", ["exact", "greedy", "annealing"])
@pytest.mark.parametrize("seed", range(5))
def test_two_factorizable_contexts_are_factorized_completely(seed, method):
    X = _two_factorizable(seed)
    model = Ord2Factor(method=method, random_state=0).fit(X)
    exact = ExactOrdinalTwoFactorizer().fit(X)

    assert model.coverage_ == 1.0
    assert model.n_rounds_ == 0
    assert model.removed_incidence_ == frozenset()
    assert [f.relation for f in model.factors_] == [f.relation for f in exact.factors_]


@pytest.mark.parametrize("method", ["exact", "greedy", "annealing"])
@pytest.mark.parametrize("seed", range(25))
def test_factors_are_ferrers_and_cover_the_kept_crosses(seed, method):
    rng = np.random.default_rng(100 + seed)
    X = rng.random((int(rng.integers(2, 9)), int(rng.integers(2, 7)))) < rng.uniform(0.3, 0.7)
    context = FormalContext.from_array(X)
    model = Ord2Factor(method=method, n_iterations=100, random_state=seed).fit(X)

    crosses = incidence_pairs(context)
    first, second = model.factors_
    assert is_ferrers_relation(first.relation, context)
    assert is_ferrers_relation(second.relation, context)
    assert model.covered_incidence_ == crosses - model.removed_incidence_
    assert first.relation | second.relation == model.covered_incidence_
    assert model.coverage_ == pytest.approx(len(model.covered_incidence_) / max(len(crosses), 1))
    # What is left is two-factorizable: its incompatibility graph is bipartite.
    assert is_bipartite(incompatibility_graph(model.reduced_context_))


@pytest.mark.parametrize("seed", range(20))
def test_exact_first_round_drops_fewest_crosses(seed):
    rng = np.random.default_rng(200 + seed)
    X = rng.random((6, 5)) < 0.5
    exact = Ord2Factor(method="exact").fit(X)
    greedy = Ord2Factor(method="greedy", random_state=seed).fit(X)
    if exact.n_rounds_ == greedy.n_rounds_ == 1:
        assert len(exact.removed_incidence_) <= len(greedy.removed_incidence_)


def test_invalid_parameters():
    with pytest.raises(ValueError, match="method"):
        Ord2Factor(method="genetic").fit(np.eye(3))
    with pytest.raises(ValueError, match="n_iterations"):
        Ord2Factor(method="annealing", n_iterations=-1).fit(np.eye(3))
