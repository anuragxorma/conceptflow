import pandas as pd
import pytest

from conceptflow import ManyValuedContext
from conceptflow.exploration import ExplorationBuilder, ExplorationView
from conceptflow.preprocessing import NominalScale


def make_many_valued_context():
    df = pd.DataFrame(
        {
            "region": ["EU", "EU", "US", "US"],
            "sector": ["energy", "tech", "energy", "tech"],
            "risk": ["low", "high", "low", "high"],
        },
        index=["g1", "g2", "g3", "g4"],
    )

    return ManyValuedContext.from_dataframe(df)


def test_exploration_builder_creates_root_view():
    mvc = make_many_valued_context()

    builder = ExplorationBuilder(mvc)

    root = builder.root(
        name="Region view",
        scales=[NominalScale("region")],
    )

    assert isinstance(root, ExplorationView)
    assert root.name == "Region view"
    assert root.depth == 0
    assert root.scale_names == ("region",)
    assert root.context.n_objects == 4
    assert root.lattice.n_concepts > 0


def test_exploration_view_adds_and_retrieves_child():
    mvc = make_many_valued_context()
    builder = ExplorationBuilder(mvc)

    root = builder.root(
        name="Region view",
        scales=[NominalScale("region")],
    )

    concept = next(
        concept
        for concept in root.lattice.concepts
        if len(concept.extent) > 0
    )

    child = builder.expand(
        parent=root,
        concept=concept,
        name="Sector detail",
        scales=[NominalScale("sector")],
    )

    concept_id = concept.stable_id()

    assert root.has_child(concept_id)
    assert root.child(concept_id) is child
    assert child.depth == 1
    assert child.parent_concept_id == concept_id
    assert child.scale_names == ("sector",)


def test_exploration_expand_restricts_to_parent_extent_objects():
    mvc = make_many_valued_context()
    builder = ExplorationBuilder(mvc)

    root = builder.root(
        name="Region view",
        scales=[NominalScale("region")],
    )

    concept = next(
        concept
        for concept in root.lattice.concepts
        if 0 < len(concept.extent) < root.context.n_objects
    )

    child = builder.expand(
        parent=root,
        concept=concept,
        name="Sector detail",
        scales=[NominalScale("sector")],
    )

    expected_objects = {
        root.context.objects[index]
        for index in concept.extent
    }

    assert set(child.context.objects) == expected_objects


def test_exploration_expand_rejects_empty_extent():
    mvc = make_many_valued_context()
    builder = ExplorationBuilder(mvc)

    root = builder.root(
        name="Region view",
        scales=[NominalScale("region")],
    )

    empty_concept = next(
        concept
        for concept in root.lattice.concepts
        if len(concept.extent) == 0
    )

    with pytest.raises(ValueError):
        builder.expand(
            parent=root,
            concept=empty_concept,
            name="Impossible child",
            scales=[NominalScale("sector")],
        )


def test_exploration_expand_all_creates_children():
    mvc = make_many_valued_context()
    builder = ExplorationBuilder(mvc)

    root = builder.root(
        name="Region view",
        scales=[NominalScale("region")],
    )

    children = builder.expand_all(
        parent=root,
        scales=[NominalScale("sector")],
        include_bottom=False,
    )

    assert len(children) > 0
    assert len(root.children) == len(children)

    for child in children:
        assert child.depth == 1
        assert child.scale_names == ("sector",)


def test_exploration_repr_contains_debug_information():
    mvc = make_many_valued_context()
    builder = ExplorationBuilder(mvc)

    root = builder.root(
        name="Region view",
        scales=[NominalScale("region")],
    )

    text = repr(root)

    assert "ExplorationView" in text
    assert "Region view" in text
    assert "n_concepts" in text


def test_expand_rejects_a_concept_of_another_view():
    builder = ExplorationBuilder(make_many_valued_context())
    root = builder.root("region", scales=[NominalScale("region")])
    eu = next(c for c in root.lattice.concepts if len(c.extent) == 2)
    child = builder.expand(root, eu, "EU", scales=[NominalScale("sector")])
    other = builder.root("sector", scales=[NominalScale("sector")])

    for concept in root.lattice.concepts:  # indices of root, not of child
        if concept not in set(child.lattice.concepts):
            with pytest.raises(ValueError, match="not a concept of the parent view 'EU'"):
                builder.expand(child, concept, "wrong", scales=[NominalScale("risk")])
    foreign = next(c for c in other.lattice.concepts if c not in set(root.lattice.concepts))
    with pytest.raises(ValueError, match="parent.lattice.concepts"):
        builder.expand(root, foreign, "wrong", scales=[NominalScale("risk")])


def test_expanding_again_replaces_the_child_view():
    builder = ExplorationBuilder(make_many_valued_context())
    root = builder.root("region", scales=[NominalScale("region")])
    top = root.lattice.top()
    builder.expand(root, top, "first", scales=[NominalScale("sector")])
    builder.expand(root, top, "second", scales=[NominalScale("risk")])
    assert root.child(top.stable_id()).name == "second"
    assert len(root.children) == 1


@pytest.mark.parametrize(
    "kwargs, error, match",
    [
        ({"min_extent_size": -1}, ValueError, "min_extent_size"),
        ({"min_extent_size": 1.5}, TypeError, "min_extent_size"),
        ({"max_extent_size": -1}, ValueError, "max_extent_size"),
        ({"name_template": "{nope}"}, ValueError, "may use"),
        ({"name_template": "{0}"}, ValueError, "may use"),
        ({"name_template": "{label"}, ValueError, "may use"),
        ({"predicate": 5}, TypeError, "predicate"),
    ],
)
def test_expand_all_rejects_invalid_options(kwargs, error, match):
    builder = ExplorationBuilder(make_many_valued_context())
    root = builder.root("region", scales=[NominalScale("region")])
    with pytest.raises(error, match=match):
        builder.expand_all(root, scales=[NominalScale("sector")], **kwargs)
    assert root.children == {}


def test_expand_all_name_template_fields():
    builder = ExplorationBuilder(make_many_valued_context())
    root = builder.root("region", scales=[NominalScale("region")])
    children = builder.expand_all(
        root, scales=[NominalScale("sector")],
        name_template="{label}|{intent}|{extent_size}|{depth}", max_extent_size=2,
    )
    assert sorted(child.name for child in children) == [
        "region=EU|region=EU|2|1", "region=US|region=US|2|1",
    ]
