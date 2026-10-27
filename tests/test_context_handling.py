import numpy as np
import pytest

from conceptflow import ConceptLattice, FormalContext
from conceptflow.algorithms import clarify_context, reduce_context
from conceptflow.io import (
    context_from_json,
    context_to_json,
    read_csv,
    read_json,
    write_csv,
    write_json,
)


def random_context(seed):
    rng = np.random.default_rng(seed)
    g, m = int(rng.integers(0, 10)), int(rng.integers(0, 7))
    return FormalContext.from_array(rng.random((g, m)) < rng.uniform(0.1, 0.9))


@pytest.mark.parametrize("seed", range(40))
def test_reduction_removes_exactly_the_reducible_elements(seed):
    ctx = random_context(seed)
    clarified = clarify_context(ctx).context
    lattice = ConceptLattice.from_context(clarified)
    result = reduce_context(ctx)

    # In a finite lattice an element is join-irreducible iff it has exactly
    # one lower neighbour (meet-irreducible: exactly one upper neighbour).
    irreducible_objects = {
        name
        for name in clarified.objects
        if len(lattice.children(lattice.object_concept(name))) == 1
    }
    irreducible_attributes = {
        name
        for name in clarified.attributes
        if len(lattice.parents(lattice.attribute_concept(name))) == 1
    }

    assert set(result.context.objects) == irreducible_objects
    assert set(result.context.attributes) == irreducible_attributes
    assert (
        ConceptLattice.from_context(result.context).n_concepts
        == ConceptLattice.from_context(ctx).n_concepts
    )

    again = reduce_context(result.context)
    assert again.context == result.context
    assert again.removed_objects == again.removed_attributes == ()


def test_editing_methods():
    ctx = FormalContext.from_array(
        [[1, 0, 1], [0, 1, 1]], objects=["x", "y"], attributes=["a", "b", "c"]
    )

    assert ctx.transpose().objects == ("a", "b", "c")
    assert ctx.transpose().transpose() == ctx
    assert ctx.subcontext(objects=["y"], attributes=["c", "a"]).incidence.tolist() == [
        [True, False]
    ]
    assert ctx.remove_objects(["x"]).objects == ("y",)
    assert ctx.remove_attributes([0, "c"]).attributes == ("b",)
    assert ctx.add_objects(["z"], [[1, 1, 0]]).incidence[-1].tolist() == [True, True, False]
    assert ctx.add_attributes(["d"], [[1], [None]]).incidence[:, -1].tolist() == [True, False]
    assert ctx.rename(attributes={"a": "A"}).attributes == ("A", "b", "c")
    assert ctx.rename(objects=str.upper).objects == ("X", "Y")
    assert ctx.subcontext(objects=[]).n_objects == 0


@pytest.mark.parametrize("seed", range(20))
def test_json_and_csv_round_trips(seed, tmp_path):
    ctx = random_context(seed)

    write_json(ctx, tmp_path / "ctx.json")
    write_csv(ctx, tmp_path / "ctx.csv")

    assert read_json(tmp_path / "ctx.json") == ctx
    assert read_csv(tmp_path / "ctx.csv") == ctx
    assert context_from_json(context_to_json(ctx)) == ctx


def test_json_reads_fcart_and_plain_layouts():
    fcart = (
        '[{"ObjNames":["g one","g2"],"Params":{"AttrNames":["a","b","c"]}},'
        '{"Count":2,"Data":[{"Count":2,"Inds":[0,2]},{"Count":2,"Inds":[1,2]}]}]'
    )
    plain = '{"objects":["g one","g2"],"attributes":["a","b","c"],"incidence":[[1,0,1],[0,1,1]]}'

    assert context_from_json(fcart) == context_from_json(plain)
    with pytest.raises(ValueError, match="Unrecognised"):
        context_from_json("[1, 2]")


def test_csv_custom_values_and_strictness(tmp_path):
    path = tmp_path / "ctx.csv"
    path.write_text(",a,b\no1,yes,no\no2,no,yes\n")

    with pytest.raises(ValueError, match="Line 2, attribute 'a'"):
        read_csv(path)

    ctx = read_csv(path, true_values=["yes"], false_values=["no"])
    assert ctx.incidence.tolist() == [[True, False], [False, True]]
