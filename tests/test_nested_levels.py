"""
Click-through levels of nested diagrams against their definition.

Every level reached by clicking through ``exploration_view_to_nested_data``
is part of one nested line diagram of all the scales on the path. A node,
given by one concept per level, is filled iff the objects having all these
intents have exactly each of them (in the context of its scale over all
objects), each object is labelled once, at its own object concepts, and
the details of a node list the objects having all these intents.
"""

import numpy as np
import pandas as pd
import pytest

from conceptflow import ExplorationBuilder, ManyValuedContext
from conceptflow.preprocessing import (
    ConceptualScaler,
    ContranominalScale,
    NominalScale,
    OrdinalScale,
    ThresholdScale,
)
from conceptflow.visualization.nested import exploration_view_to_nested_data

LEVEL_SCALES = [
    [NominalScale("color")],
    [ThresholdScale("size", [33, 66])],
    [ContranominalScale("shape")],
    [OrdinalScale("grade", ["low", "mid", "high"])],
]


def random_data(seed, n):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "color": rng.choice(["red", "blue", "green"], n),
            "size": rng.uniform(0, 100, n).round(),
            "shape": rng.choice(["round", "square", "oval"], n),
            "grade": rng.choice(["low", "mid", "high"], n),
        },
        index=[f"o{i}" for i in range(n)],
    )


def expand_everything(builder, view, levels):
    if view.depth + 1 >= len(levels):
        return
    for child in builder.expand_all(view, scales=levels[view.depth + 1]):
        expand_everything(builder, child, levels)


def check_levels(nodes, level, path, contexts, labels, stats):
    """Walk the click-through data; ``path`` holds the intents above."""
    deepest = len(contexts) - 1
    for node in nodes:
        here = (*path, frozenset(node["metadata"]["intent"]))
        objects = set(contexts[0].objects)
        for context, intent in zip(contexts, here):
            objects &= {context.objects[g] for g in context.attribute_derivation(intent)}
        realized = all(
            {context.attributes[j] for j in context.object_derivation(objects)} == intent
            for context, intent in zip(contexts, here)
        )
        stats["levels"].add(level)
        # The details panel lists the objects having all these intents.
        assert set(node["metadata"]["extent"]) == objects, here
        assert node["metadata"]["extent_size"] == len(objects)
        if level > 0:
            stats["nodes"] += 1
            assert node["is_filled"] == realized, here
        for name in node["new_objects"]:
            assert all(
                {context.attributes[j] for j in context.object_derivation([name])} == intent
                for context, intent in zip(contexts, here)
            ), (name, here)
            if level == deepest:
                labels.append(name)
        if node["has_child"]:
            assert level == 0 or realized, here
            check_levels(node["child"]["nodes"], level + 1, here, contexts, labels, stats)
        elif level < deepest and objects and realized:
            pytest.fail(f"realized node {here} with objects cannot be entered")


@pytest.mark.parametrize("depth", [3, 4])
@pytest.mark.parametrize("seed", range(4))
def test_click_through_levels_are_one_nested_diagram(seed, depth):
    data = random_data(seed, n=8 + 3 * seed)
    levels = LEVEL_SCALES[:depth]
    builder = ExplorationBuilder(ManyValuedContext.from_dataframe(data))
    root = builder.root("root", levels[0])
    expand_everything(builder, root, levels)

    nested = exploration_view_to_nested_data(root, timeout_ms=50)

    contexts = [
        ConceptualScaler(scales=scales, output="context").fit_transform(data)
        for scales in levels
    ]
    labels, stats = [], {"nodes": 0, "levels": set()}
    check_levels(nested["nodes"], 0, (), contexts, labels, stats)
    assert sorted(labels) == sorted(data.index)  # every object exactly once
    assert stats["levels"] == set(range(depth))  # the walk reached every level
    assert stats["nodes"] > 20


def test_child_views_keep_the_values_their_objects_lack():
    data = pd.DataFrame(
        {"color": ["red", "red", "blue"], "shape": ["round", "round", "square"]},
        index=["a", "b", "c"],
    )
    builder = ExplorationBuilder(ManyValuedContext.from_dataframe(data))
    root = builder.root("color", [NominalScale("color")])
    red = next(c for c in root.lattice.concepts if len(c.extent) == 2)
    child = builder.expand(root, red, "red", [ContranominalScale("shape")])
    assert child.context.attributes == ("shape!=round", "shape!=square")
    # Both red objects lack "square", so both have "shape != square".
    assert child.lattice.top().intent == frozenset({1})


def test_details_of_a_node_inside_an_outer_concept():
    # The details panel showed the inner concept's extent over all objects.
    data = pd.DataFrame(
        {"color": ["red", "red", "blue"], "size": [10, 90, 90]}, index=["a", "b", "c"]
    )
    builder = ExplorationBuilder(ManyValuedContext.from_dataframe(data))
    root = builder.root("color", [NominalScale("color")])
    builder.expand_all(root, scales=[ThresholdScale("size", [50])])
    nested = exploration_view_to_nested_data(root, timeout_ms=50)
    red = next(n for n in nested["nodes"] if n["metadata"]["intent"] == ["color=red"])
    big = next(
        n for n in red["child"]["nodes"] if n["metadata"]["intent"] == ["size>=50"]
    )
    assert big["metadata"]["extent"] == ["b"]
    assert big["metadata"]["extent_size"] == 1


def test_max_depth_must_be_a_non_negative_integer():
    builder = ExplorationBuilder(ManyValuedContext.from_dataframe(random_data(0, 6)))
    root = builder.root("root", LEVEL_SCALES[0])
    for bad in (-1, 1.5, "1"):
        with pytest.raises((ValueError, TypeError), match="max_depth"):
            exploration_view_to_nested_data(root, max_depth=bad, timeout_ms=50)


def test_nested_data_is_the_same_in_every_run():
    # filled_inner came from a set, so its order changed with the hash seed.
    data = random_data(1, 12)
    builder = ExplorationBuilder(ManyValuedContext.from_dataframe(data))
    root = builder.root("root", LEVEL_SCALES[0])
    expand_everything(builder, root, LEVEL_SCALES[:3])
    nested = exploration_view_to_nested_data(root, timeout_ms=50)
    lists = []

    def collect(nodes):
        for node in nodes:
            lists.append(node["filled_inner"])
            if node["has_child"]:
                collect(node["child"]["nodes"])

    collect(nested["nodes"])
    assert any(len(filled) > 1 for filled in lists)
    assert all(filled == sorted(filled) for filled in lists)
