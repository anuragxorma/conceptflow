import numpy as np
import pytest

from conceptflow import FormalContext
from conceptflow.decomposition import (
    ExactOrdinalTwoFactorizer,
    bipartite_coloring,
    complement_context,
    incompatibility_graph,
    incidence_pairs,
    is_ferrers_relation,
)
from conceptflow.decomposition.ordinal_factorization import Ord2Factor


def make_chain_context():
    return FormalContext.from_array(
        np.array([
            [1, 0, 0],
            [1, 1, 0],
            [1, 1, 1],
        ]),
        objects=["g1", "g2", "g3"],
        attributes=["m1", "m2", "m3"],
    )


def test_incidence_pairs():
    ctx = make_chain_context()

    assert incidence_pairs(ctx) == frozenset({
        (0, 0),
        (1, 0),
        (1, 1),
        (2, 0),
        (2, 1),
        (2, 2),
    })


def test_complement_context():
    ctx = FormalContext.from_array(
        np.array([
            [1, 0],
            [0, 1],
        ]),
        objects=["g1", "g2"],
        attributes=["m1", "m2"],
    )

    complement = complement_context(ctx)

    assert complement.objects == ctx.objects
    assert complement.attributes == ctx.attributes
    assert complement.incidence.tolist() == [
        [False, True],
        [True, False],
    ]


def test_chain_context_is_ferrers():
    ctx = make_chain_context()
    relation = incidence_pairs(ctx)

    assert is_ferrers_relation(relation, ctx)


def test_non_ferrers_relation():
    ctx = FormalContext.from_array(
        np.array([
            [1, 0],
            [0, 1],
        ])
    )

    relation = frozenset({
        (0, 0),
        (1, 1),
    })

    assert not is_ferrers_relation(relation, ctx)


def test_incompatibility_graph_empty_for_ferrers_chain():
    ctx = make_chain_context()
    graph = incompatibility_graph(ctx)

    assert all(len(neighbors) == 0 for neighbors in graph.values())


def test_bipartite_coloring_simple_graph():
    graph = {
        (0, 0): {(1, 1)},
        (1, 1): {(0, 0)},
    }

    coloring = bipartite_coloring(graph)

    assert coloring[(0, 0)] != coloring[(1, 1)]


def test_bipartite_coloring_rejects_odd_cycle():
    graph = {
        (0, 0): {(1, 1), (2, 2)},
        (1, 1): {(0, 0), (2, 2)},
        (2, 2): {(0, 0), (1, 1)},
    }

    with pytest.raises(ValueError):
        bipartite_coloring(graph)


def test_exact_ordinal_two_factorizer_fits_ferrers_chain_context():
    ctx = make_chain_context()

    model = ExactOrdinalTwoFactorizer()
    model.fit(ctx)

    assert model.coverage_ == 1.0
    assert len(model.factors_) == 2
    assert model.covered_incidence_ == incidence_pairs(ctx)

    factor_1, factor_2 = model.factors_

    assert is_ferrers_relation(factor_1.relation, ctx)
    assert is_ferrers_relation(factor_2.relation, ctx)


def test_exact_ordinal_two_factorizer_fits_contranominal_dimension_three():
    ctx = FormalContext.from_array(
        np.array([
            [1, 1, 0],
            [1, 0, 1],
            [0, 1, 1],
        ])
    )

    model = ExactOrdinalTwoFactorizer()
    model.fit(ctx)

    assert model.coverage_ == 1.0
    assert model.covered_incidence_ == incidence_pairs(ctx)

    factor_1, factor_2 = model.factors_

    assert is_ferrers_relation(factor_1.relation, ctx)
    assert is_ferrers_relation(factor_2.relation, ctx)


def test_ord2factor_stub_is_not_implemented():
    ctx = make_chain_context()
    model = Ord2Factor()

    with pytest.raises(NotImplementedError):
        model.fit(ctx)