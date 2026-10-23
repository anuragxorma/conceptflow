import numpy as np
import pandas as pd
import pytest

from conceptflow import ConceptLattice, FormalContext, ManyValuedContext
from conceptflow.exploration import (
    ExplorationBuilder,
    nested_combinations,
    realized_pairs,
)
from conceptflow.preprocessing import NominalScale, OrdinalScale
from conceptflow.visualization.nested import exploration_view_to_nested_data


def make_nested_view():
    mvc = ManyValuedContext.from_dataframe(
        pd.DataFrame(
            {
                "size": ["S", "M", "L", "L", "M"],
                "kind": ["x", "x", "y", "x", "y"],
            },
            index=["g1", "g2", "g3", "g4", "g5"],
        )
    )
    builder = ExplorationBuilder(mvc)
    root = builder.root("size", [OrdinalScale("size", ["S", "M", "L"])])
    builder.expand(root, root.lattice.top(), "kind", [NominalScale("kind")])
    return root


@pytest.mark.parametrize("seed", range(20))
def test_realized_pairs_biject_with_concepts_of_apposition(seed):
    rng = np.random.default_rng(seed)
    incidence = rng.random((8, 6)) < 0.5
    objects = [f"g{i}" for i in range(8)]
    outer = ConceptLattice.from_context(
        FormalContext.from_array(incidence[:, :3], objects, ["a0", "a1", "a2"])
    )
    inner = ConceptLattice.from_context(
        FormalContext.from_array(incidence[:, 3:], objects, ["b0", "b1", "b2"])
    )
    full = ConceptLattice.from_context(FormalContext.from_array(incidence, objects))

    pairs = realized_pairs(outer, inner)
    full_intents = {c.intent for c in full.concepts}
    combined = {o.intent | {3 + m for m in i.intent} for o, i in pairs}

    assert combined == full_intents


def test_nested_combinations_table():
    root = make_nested_view()
    table = nested_combinations(root)
    inner = root.children[root.lattice.top().stable_id()]

    assert len(table) == root.lattice.n_concepts * inner.lattice.n_concepts
    assert set(table.columns) >= {
        "outer_intent", "inner_intent", "realized", "n_objects", "introduced_objects",
    }

    # size>=L and kind=y are both held by g3 only.
    row = table[
        (table.outer_intent == ("size>=S", "size>=M", "size>=L"))
        & (table.inner_intent == ("kind=y",))
    ].iloc[0]
    assert row.realized
    assert row.objects == ("g3",)
    assert row.introduced_objects == ("g3",)

    # Every object is introduced exactly once, always at a realized pair.
    introduced = [g for gs in table.introduced_objects for g in gs]
    assert sorted(introduced) == ["g1", "g2", "g3", "g4", "g5"]
    assert table[table.introduced_objects.map(len) > 0].realized.all()


def test_nested_combinations_match_filled_inner_nodes():
    root = make_nested_view()
    table = nested_combinations(root)
    data = exploration_view_to_nested_data(root)

    filled = {
        (node["id"], inner_id)
        for node in data["nodes"]
        for inner_id in node["filled_inner"]
    }
    realized = set(
        table[table.realized][["outer_id", "inner_id"]].itertuples(index=False, name=None)
    )
    assert filled == realized


def test_nested_combinations_requires_inner_view():
    mvc = ManyValuedContext.from_dataframe(
        pd.DataFrame({"kind": ["x", "y"]}, index=["g1", "g2"])
    )
    root = ExplorationBuilder(mvc).root("kind", [NominalScale("kind")])

    with pytest.raises(ValueError, match="inner view"):
        nested_combinations(root)
