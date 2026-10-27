import itertools

import numpy as np
import pytest

from conceptflow import ConceptLattice, FormalContext
from conceptflow.core import (
    apposition,
    complement,
    direct_product,
    direct_sum,
    dual,
    subposition,
)
from conceptflow.exploration import nested_view, realized_pairs
from conceptflow.visualization import plot_nested


def pair(seed):
    rng = np.random.default_rng(seed)
    g1, g2, m1, m2 = (int(x) for x in rng.integers(1, 6, 4))
    first = FormalContext.from_array(
        rng.random((g1, m1)) < 0.5, [f"g{i}" for i in range(g1)], [f"a{j}" for j in range(m1)]
    )
    second = FormalContext.from_array(
        rng.random((g2, m2)) < 0.5, [f"h{i}" for i in range(g2)], [f"b{j}" for j in range(m2)]
    )
    return first, second


def n_concepts(context):
    return ConceptLattice.from_context(context).n_concepts


@pytest.mark.parametrize("seed", range(20))
def test_direct_sum_lattice_is_the_product_of_lattices(seed):
    first, second = pair(seed)
    assert n_concepts(direct_sum(first, second)) == n_concepts(first) * n_concepts(second)


@pytest.mark.parametrize("seed", range(10))
def test_direct_product_definition(seed):
    first, second = pair(seed)
    product = direct_product(first, second)

    assert product.n_objects == first.n_objects * second.n_objects
    for (g1, g2), (m1, m2) in itertools.product(
        itertools.product(range(first.n_objects), range(second.n_objects)),
        itertools.product(range(first.n_attributes), range(second.n_attributes)),
    ):
        row, column = g1 * second.n_objects + g2, m1 * second.n_attributes + m2
        expected = first.incidence[g1, m1] or second.incidence[g2, m2]
        assert product.incidence[row, column] == expected
    assert product.objects[0] == f"{first.objects[0]}|{second.objects[0]}"


def test_apposition_matches_objects_by_name():
    left = FormalContext.from_array([[1, 0], [0, 1]], objects=["x", "y"], attributes=["a", "b"])
    right = FormalContext.from_array([[1], [0]], objects=["y", "x"], attributes=["c"])

    combined = apposition(left, right)

    assert combined.attributes == ("a", "b", "c")
    assert combined.incidence.tolist() == [[True, False, False], [False, True, True]]


def test_subposition_and_prefixes():
    ctx = FormalContext.from_array([[1, 0]], objects=["x"], attributes=["a", "b"])

    with pytest.raises(ValueError, match="prefixes"):
        subposition(ctx, ctx)
    stacked = subposition(ctx, ctx, prefixes=("1:", "2:"))
    assert stacked.objects == ("1:x", "2:x")

    with pytest.raises(ValueError, match="same objects"):
        apposition(ctx, ctx.rename(objects={"x": "z"}), prefixes=("1:", "2:"))


def test_complement_and_dual():
    ctx = FormalContext.from_array([[1, 0, 1]], attributes=["a", "b", "c"])
    assert complement(ctx).incidence.tolist() == [[False, True, False]]
    assert dual(ctx) == ctx.transpose()


@pytest.mark.parametrize("seed", range(10))
def test_nested_view_realizes_the_apposition(seed):
    first, _ = pair(seed)
    rng = np.random.default_rng(100 + seed)
    second = FormalContext.from_array(
        rng.random((first.n_objects, 3)) < 0.5, list(first.objects)[::-1], ["b0", "b1", "b2"]
    )

    view = nested_view(first, second)
    inner = view.children[view.lattice.top().stable_id()]

    assert len(realized_pairs(view.lattice, inner.lattice)) == n_concepts(apposition(first, second))


def test_plot_nested_from_two_contexts():
    rng = np.random.default_rng(0)
    objects = [f"o{i}" for i in range(8)]
    outer = FormalContext.from_array(rng.random((8, 3)) < 0.5, objects, ["x", "y", "z"])
    inner = FormalContext.from_array(rng.random((8, 2)) < 0.5, objects, ["p", "q"])

    assert type(plot_nested(outer, inner=inner)).__name__ == "HTMLFigure"
    assert type(plot_nested(outer, inner=inner, backend="tikz")).__name__ == "TikZFigure"

    with pytest.raises(ValueError, match="disjoint"):
        nested_view(outer, outer)
