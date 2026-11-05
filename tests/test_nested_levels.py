"""Views of an ExplorationBuilder have the attributes of a view of all objects."""

import pandas as pd

from conceptflow import ExplorationBuilder, ManyValuedContext
from conceptflow.preprocessing import ContranominalScale, NominalScale


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
