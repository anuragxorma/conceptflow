"""
Property-based tests of file formats, incomplete contexts and nested
diagrams.

- Files: writing and reading a context gives it back, whatever its names
  (spaces, separators, quotes, non-ASCII and control characters), or fails
  with the documented ``ValueError``; it never comes back changed.
- Incomplete contexts: "certain" and "possible" mean "in every completion"
  and "in some completion", checked by listing the completions.
- Nested diagrams: the filled (realized) combinations of concepts are the
  concepts of the combined context, on every level.
"""

import copy
import itertools
import pickle

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("hypothesis")

from hypothesis import assume, given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

from conceptflow import (  # noqa: E402
    ConceptLattice,
    ExplorationBuilder,
    FormalContext,
    IncompleteContext,
    ManyValuedContext,
)
from conceptflow.core import apposition  # noqa: E402
from conceptflow.exploration import (  # noqa: E402
    nested_combinations,
    nested_view,
    realized_pairs,
)
from conceptflow.io import (  # noqa: E402
    context_from_json,
    context_to_json,
    read_csv,
    read_cxt,
    read_json,
    write_csv,
    write_cxt,
    write_json,
)
from conceptflow.preprocessing import (  # noqa: E402
    ContranominalScale,
    NominalScale,
    OrdinalScale,
    ThresholdScale,
)
from conceptflow.visualization.nested import exploration_view_to_nested_data  # noqa: E402
from tests.property_helpers import (  # noqa: E402
    Reference,
    all_subsets,
    context_of,
    incidences,
    named_contexts,
    to_set,
)

# ---------------------------------------------------------------------------
# File formats
# ---------------------------------------------------------------------------


def line_break_in_names(context):
    return any("\n" in n or "\r" in n for n in context.objects + context.attributes)


@given(named_contexts())
def test_cxt_files_round_trip_or_refuse_line_breaks(tmp_path_factory, context):
    path = tmp_path_factory.mktemp("cxt") / "context.cxt"
    if line_break_in_names(context):
        with pytest.raises(ValueError, match="line break"):
            write_cxt(context, path)
        return
    write_cxt(context, path)
    assert read_cxt(path) == context


@given(named_contexts())
def test_json_round_trips(tmp_path_factory, context):
    assert context_from_json(context_to_json(context)) == context
    path = tmp_path_factory.mktemp("json") / "context.json"
    write_json(context, path)
    assert read_json(path) == context


@given(named_contexts(), st.sampled_from([",", ";", "\t", "|"]))
def test_csv_files_round_trip(tmp_path_factory, context, delimiter):
    path = tmp_path_factory.mktemp("csv") / "context.csv"
    write_csv(context, path, delimiter=delimiter)
    assert read_csv(path, delimiter=delimiter) == context


@given(named_contexts())
def test_dataframes_round_trip(context):
    assert FormalContext.from_dataframe(context.to_dataframe()) == context


# ---------------------------------------------------------------------------
# Incomplete contexts
# ---------------------------------------------------------------------------


@st.composite
def incomplete_tables(draw, max_objects=4, max_attributes=4, max_unknown=6):
    g = draw(st.integers(0, max_objects), label="objects")
    m = draw(st.integers(0, max_attributes), label="attributes")
    cells = draw(st.lists(st.sampled_from([0, 1, None]), min_size=g * m, max_size=g * m))
    table = np.array(cells, dtype=object).reshape(g, m)
    assume(sum(cell is None for cell in cells) <= max_unknown)
    return table


def completions(table):
    unknown = [(i, j) for i, j in itertools.product(*map(range, table.shape))
               if table[i, j] is None]
    known = np.array([[cell == 1 for cell in row] for row in table], dtype=bool).reshape(
        table.shape
    )
    for bits in itertools.product([False, True], repeat=len(unknown)):
        completion = known.copy()
        for (i, j), bit in zip(unknown, bits):
            completion[i, j] = bit
        yield Reference(completion)


@given(incomplete_tables())
def test_incomplete_contexts_answer_for_every_completion(table):
    context = IncompleteContext.from_array(table)
    worlds = list(completions(table))
    g, m = table.shape
    assert context.n_unknown == sum(cell is None for cell in table.flat)

    for attributes in all_subsets(range(m)):
        mask = sum(1 << j for j in attributes)
        extents = [to_set(world.extent(mask)) for world in worlds]
        assert context.certain_extent(attributes) == frozenset.intersection(*extents)
        assert context.possible_extent(attributes) == frozenset.union(*extents)
    for objects in all_subsets(range(g)):
        mask = sum(1 << i for i in objects)
        intents = [to_set(world.intent(mask)) for world in worlds]
        assert context.certain_intent(objects) == frozenset.intersection(*intents)
        assert context.possible_intent(objects) == frozenset.union(*intents)

    for premise, conclusion in itertools.product(list(all_subsets(range(m))), repeat=2):
        p = sum(1 << j for j in premise)
        c = sum(1 << j for j in conclusion)
        broken = [to_set(w.extent(p) & ~w.extent(p | c)) for w in worlds]
        certain, possible = context.implication_counterexamples(premise, conclusion)
        assert certain == frozenset.intersection(*broken)
        assert possible == frozenset.union(*broken)
        holds = [not b for b in broken]
        expected = "valid" if all(holds) else "possible" if any(holds) else "refuted"
        assert context.implication_status(premise, conclusion) == expected


@given(incomplete_tables())
def test_incomplete_contexts_convert_and_survive_pickling(table):
    context = IncompleteContext.from_array(table)
    assert IncompleteContext.from_dataframe(context.to_dataframe()) == context
    assert (context.certain_context().incidence <= context.possible_context().incidence).all()
    if context.is_complete:
        complete = IncompleteContext.from_formal_context(context.certain_context())
        assert complete == context
        assert context.certain_context() == context.possible_context()
    for copied in (pickle.loads(pickle.dumps(context)), copy.deepcopy(context)):
        assert copied == context and hash(copied) == hash(context)
        assert not copied.certain.flags.writeable and not copied.unknown.flags.writeable


# ---------------------------------------------------------------------------
# Nested diagrams, two levels
# ---------------------------------------------------------------------------


@st.composite
def two_views(draw):
    """Two contexts over the same objects, the inner one in another order."""
    outer_cells = draw(incidences(max_objects=6, max_attributes=4))
    g = outer_cells.shape[0]
    inner_cells = draw(incidences(min_objects=g, max_objects=g, max_attributes=4))
    outer = context_of(outer_cells).rename(attributes=lambda name: f"o{name}")
    inner = context_of(inner_cells).rename(attributes=lambda name: f"i{name}")
    order = draw(st.permutations(range(g)), label="inner object order")
    return outer, inner.subcontext(objects=list(order))


def apposition_intents(outer, inner):
    together = apposition(outer, inner.subcontext(objects=list(outer.objects)))
    split = outer.n_attributes
    return {
        (frozenset(j for j in b if j < split), frozenset(j - split for j in b if j >= split))
        for _, b in Reference(together.incidence).concept_sets()
    }


@given(two_views())
def test_realized_pairs_are_the_concepts_of_the_apposition(views):
    outer, inner = views
    inner = inner.subcontext(objects=list(outer.objects))
    pairs = realized_pairs(ConceptLattice.from_context(outer), ConceptLattice.from_context(inner))
    assert {(a.intent, b.intent) for a, b in pairs} == apposition_intents(outer, inner)


@given(two_views())
def test_nested_combinations_describe_every_pair(views):
    outer, inner = views
    table = nested_combinations(nested_view(outer, inner))
    outer_lattice = ConceptLattice.from_context(outer)
    inner_lattice = ConceptLattice.from_context(inner)
    assert len(table) == outer_lattice.n_concepts * inner_lattice.n_concepts

    realized = apposition_intents(outer, inner)
    outer_index = {name: j for j, name in enumerate(outer.attributes)}
    inner_index = {name: j for j, name in enumerate(inner.attributes)}
    introduced = []
    for row in table.itertuples():
        a = frozenset(outer_index[n] for n in row.outer_intent)
        b = frozenset(inner_index[n] for n in row.inner_intent)
        having = set(outer.objects[g] for g in outer.attribute_derivation(a)) & set(
            inner.objects[g] for g in inner.attribute_derivation(b)
        )
        assert set(row.objects) == having and row.n_objects == len(having)
        assert row.realized == ((a, b) in realized)
        for name in row.introduced_objects:
            assert row.realized
            assert outer.object_derivation([name]) == a
            assert inner.object_derivation([name]) == b
        introduced.extend(row.introduced_objects)
    assert sorted(introduced) == sorted(outer.objects)


@given(two_views())
def test_nested_view_matches_objects_by_name(views):
    outer, inner = views
    root = nested_view(outer, inner)
    child = root.children[root.lattice.top().stable_id()]
    assert child.context.objects == outer.objects
    assert child.context == inner.subcontext(objects=list(outer.objects))


# ---------------------------------------------------------------------------
# Nested diagrams, three and four levels
# ---------------------------------------------------------------------------

LEVEL_SCALES = [
    [NominalScale("color")],
    [ThresholdScale("size", [33, 66])],
    [ContranominalScale("shape")],
    [OrdinalScale("grade", ["low", "mid", "high"])],
]


@st.composite
def many_valued_data(draw):
    n = draw(st.integers(1, 9), label="objects")
    column = st.lists
    return pd.DataFrame(
        {
            "color": draw(column(st.sampled_from(["red", "blue", "green"]),
                                 min_size=n, max_size=n)),
            "size": draw(column(st.sampled_from([0.0, 20.0, 33.0, 50.0, 66.0, 90.0]),
                                min_size=n, max_size=n)),
            "shape": draw(column(st.sampled_from(["round", "square", "oval"]),
                                 min_size=n, max_size=n)),
            "grade": draw(column(st.sampled_from(["low", "mid", "high"]),
                                 min_size=n, max_size=n)),
        },
        index=[f"o{i}" for i in range(n)],
    )


def expand_everything(builder, view, levels):
    if view.depth + 1 >= len(levels):
        return
    for child in builder.expand_all(view, scales=levels[view.depth + 1]):
        expand_everything(builder, child, levels)


def walk(nodes, level, path, contexts, labels):
    """Every node against the definition; ``path`` holds the intents above."""
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
        assert set(node["metadata"]["extent"]) == objects, here
        assert node["metadata"]["extent_size"] == len(objects)
        if level > 0:
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
            walk(node["child"]["nodes"], level + 1, here, contexts, labels)
        else:
            assert not (level < deepest and objects and realized), (
                f"realized node {here} with objects cannot be entered"
            )


@settings(max_examples=25)
@given(many_valued_data(), st.integers(2, 4))
def test_every_nested_level_is_one_nested_diagram(data, depth):
    from conceptflow.preprocessing import ConceptualScaler

    levels = LEVEL_SCALES[:depth]
    builder = ExplorationBuilder(ManyValuedContext.from_dataframe(data))
    root = builder.root("root", levels[0])
    expand_everything(builder, root, levels)
    nested = exploration_view_to_nested_data(root, timeout_ms=50)
    contexts = [
        ConceptualScaler(scales=scales, output="context").fit_transform(data)
        for scales in levels
    ]
    labels = []
    walk(nested["nodes"], 0, (), contexts, labels)
    assert sorted(labels) == sorted(data.index)  # every object exactly once
