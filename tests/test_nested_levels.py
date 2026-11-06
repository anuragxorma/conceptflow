"""
Click-through levels of nested diagrams against their definition.

Every level reached by clicking through ``exploration_view_to_nested_data``
is part of one nested line diagram of all the scales on the path. A node,
given by one concept per level, is filled iff the objects having all these
intents have exactly each of them (in the context of its scale over all
objects), and each object is labelled once, at its own object concepts.
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
