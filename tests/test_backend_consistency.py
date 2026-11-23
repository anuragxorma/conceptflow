"""
The backends draw the same diagram.

For one drawing, the interactive page (d3), matplotlib, TikZ and Graphviz
must show the same nodes, edges and labels, with each attribute at its
attribute concept and each object at its object concept (reduced
labelling, from the definition), shortened to the same ``max_label_items``
names; and matplotlib and TikZ must put the nodes where ``layout_positions``
says. For nested diagrams, matplotlib and TikZ show the same labels, and each
object is labelled once, inside its outer object concept, at its inner
object concept.
"""

import collections
import json
import math
import re

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("hypothesis")

from hypothesis import given  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

from conceptflow import (  # noqa: E402
    ConceptLattice,
    ExplorationBuilder,
    FormalContext,
    ManyValuedContext,  # noqa: E402
)
from conceptflow.preprocessing import (  # noqa: E402
    ConceptualScaler,
    ContranominalScale,
    NominalScale,
    ThresholdScale,
)
from conceptflow.visualization import (  # noqa: E402
    graph_data_to_tikz,
    lattice_to_dot,
    lattice_to_graph_data,
    layered_layout,
    layout_positions,
    nested_data_to_tikz,
    render_graph_data_html,
    render_lattice_matplotlib,
    render_nested_matplotlib,
)
from conceptflow.visualization.nested import exploration_view_to_nested_data  # noqa: E402
from tests.property_helpers import incidences  # noqa: E402

LETTERS = "qwertyuiopasdfgh"


@st.composite
def lattices(draw):
    incidence = draw(incidences(min_objects=1, max_objects=6, min_attributes=1,
                                max_attributes=6))
    g, m = incidence.shape
    names = draw(st.lists(st.text(LETTERS, min_size=1, max_size=3), min_size=g + m,
                          max_size=g + m, unique=True), label="names")
    context = FormalContext.from_array(incidence, [f"O{n}" for n in names[:g]],
                                       [f"A{n}" for n in names[g:]])
    min_support = draw(st.sampled_from([None, None, 2]), label="min_support")
    return ConceptLattice.from_context(context, min_support=min_support)


def joined(names, max_items):
    names = sorted(names)
    if max_items is not None and len(names) > max_items:
        names = [*names[:max_items], f"+{len(names) - max_items}"]
    return ", ".join(names)


def expected_labels(lattice, max_items):
    """node index -> (attribute text, object text), from the definition."""
    context = lattice.context
    attributes = collections.defaultdict(list)
    objects = collections.defaultdict(list)
    position = {c.extent: i for i, c in enumerate(lattice.concepts)}
    for m, name in enumerate(context.attributes):
        extent = context.attribute_derivation([m])
        if extent in position:  # mu(m) is drawn
            attributes[position[extent]].append(name)
    for g, name in enumerate(context.objects):
        extent = context.attribute_derivation(context.object_derivation([g]))
        if extent in position:  # gamma(g) is drawn (not below an iceberg)
            objects[position[extent]].append(name)
    return {
        i: (joined(attributes[i], max_items), joined(objects[i], max_items))
        for i in range(lattice.n_concepts)
    }


def similar(points, other, tolerance=1e-6):
    """Whether ``other`` is ``points`` moved and uniformly scaled (no rotation)."""
    keys = list(points)
    a = np.array([points[k] for k in keys], dtype=float)
    b = np.array([other[k] for k in keys], dtype=float)
    a, b = a - a.mean(axis=0), b - b.mean(axis=0)
    size_a, size_b = np.abs(a).max(), np.abs(b).max()
    if size_a < 1e-12 or size_b < 1e-12:
        return size_a < 1e-12 and size_b < 1e-12
    return np.allclose(a / size_a, b / size_b, atol=tolerance)


@given(lattices(), st.sampled_from([None, 1, 2, 6]), st.floats(0, 2 * math.pi))
def test_every_backend_shows_the_same_labels_and_positions(lattice, max_items, angle):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    expected = expected_labels(lattice, max_items)
    graph_data = layered_layout(lattice_to_graph_data(lattice))
    if not lattice.is_iceberg and lattice.n_concepts > 1:
        # Stored at an angle, as DimFlux layouts are: drawn upright.
        from dataclasses import replace

        cos_a, sin_a = math.cos(angle), math.sin(angle)
        graph_data = replace(
            graph_data, metadata={**graph_data.metadata, "layout": "custom"},
            nodes=[replace(n, x=cos_a * n.x - sin_a * n.y, y=sin_a * n.x + cos_a * n.y)
                   for n in graph_data.nodes],
        )
    index = {n.node_id: i for i, n in enumerate(graph_data.nodes)}
    assert [n.concept for n in graph_data.nodes] == list(lattice.concepts)
    positions = layout_positions(graph_data)
    drawn = {index[n.node_id]: positions[frozenset(n.metadata["extent"])]
             for n in graph_data.nodes}
    edges = {(index[e.source], index[e.target]) for e in graph_data.edges}

    # The interactive page.
    html = render_graph_data_html(graph_data, max_label_items=max_items)
    page = json.loads(re.search(r"const graphData = (.*?);\n", html).group(1))
    assert {i: (node["attribute_text"], node["object_text"])
            for i, node in enumerate(page["nodes"])} == expected
    assert {(index[e["source"]], index[e["target"]]) for e in page["edges"]} == edges

    # matplotlib: nodes where layout_positions says, labels at their nodes.
    figure = render_lattice_matplotlib(graph_data, max_label_items=max_items)
    ax = figure.axes[0]
    points = {tuple(np.round(p, 6)) for p in ax.collections[1].get_offsets()}
    assert points == {tuple(np.round(p, 6)) for p in drawn.values()}
    at_point = {tuple(np.round(p, 6)): i for i, p in drawn.items()}
    shown = collections.defaultdict(list)
    for text in ax.texts:
        shown[at_point[tuple(np.round(text.xy, 6))]].append(text.get_text())
    plt.close(figure)
    for i, (attributes, objects) in expected.items():
        assert sorted(shown.get(i, [])) == sorted(t for t in (attributes, objects) if t), i

    # TikZ: the same labels, nodes placed like layout_positions.
    tikz = graph_data_to_tikz(graph_data, max_label_items=max_items)
    nodes = re.findall(r"\\node\[concept(.*?)\] \((\w+)\) at \(([-\d.]+), ([-\d.]+)\)", tikz)
    assert len(nodes) == lattice.n_concepts
    tikz_points = {}
    for options, name, x, y in nodes:
        i = index[name]
        tikz_points[i] = (float(x), float(y))
        attributes = re.search(r"\[attr\]above:\{(.*?)\}\}", options)
        objects = re.search(r"\[obj\]below:\{(.*?)\}\}", options)
        assert (attributes.group(1) if attributes else "",
                objects.group(1) if objects else "") == expected[i], name
    assert similar(drawn, tikz_points, tolerance=1e-3)  # TikZ rounds coordinates

    # Graphviz lays the lattice out itself; the labels and edges are the same.
    dot = lattice_to_dot(lattice, max_label_items=max_items)
    for i in range(lattice.n_concepts):
        line = re.search(rf"^  c{i}( \[xlabel=<(.*)>\])?;$", dot, re.MULTILINE)
        fonts = dict(re.findall(r'<FONT COLOR="(#\w+)">(.*?)</FONT>', line.group(2) or ""))
        assert (fonts.get("#1f4e79", ""), fonts.get("#333333", "")) == expected[i], i
    position = {c: i for i, c in enumerate(lattice.concepts)}
    dot_edges = {tuple(int(x) for x in pair) for pair in re.findall(r"c(\d+) -> c(\d+)", dot)}
    assert dot_edges == {(position[a], position[b]) for a, b in lattice.edges}


# ---------------------------------------------------------------------------
# Nested diagrams
# ---------------------------------------------------------------------------


@st.composite
def nested_views(draw):
    n = draw(st.integers(1, 8), label="objects")
    rng = np.random.default_rng(draw(st.integers(0, 2**32 - 1), label="seed"))
    data = pd.DataFrame(
        {"color": rng.choice(["red", "blue", "green"], n),
         "size": rng.choice([10.0, 40.0, 80.0], n),
         "shape": rng.choice(["round", "square"], n)},
        index=[f"o{i}" for i in range(n)],
    )
    builder = ExplorationBuilder(ManyValuedContext.from_dataframe(data))
    root = builder.root("color", [NominalScale("color")])
    inner = draw(st.sampled_from(["none", "top", "all"]), label="inner views")
    inner_scales = draw(st.sampled_from([[ThresholdScale("size", [33, 66])],
                                         [ContranominalScale("shape")]]), label="inner")
    if inner == "top":
        builder.expand(root, root.lattice.top(), "inner", inner_scales)
    elif inner == "all":
        builder.expand_all(root, scales=inner_scales)
    return data, root, inner, inner_scales


def matplotlib_labels(data, **options):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure = render_nested_matplotlib(data, **options)
    texts = sorted(text.get_text() for text in figure.axes[0].texts)
    plt.close(figure)
    return texts


@given(nested_views(), st.sampled_from([None, 1, 3]), st.booleans())
def test_nested_backends_show_the_same_labels(view, max_items, objects):
    _, root, _, _ = view
    data = exploration_view_to_nested_data(root, timeout_ms=50)
    options = {"max_label_items": max_items, "show_object_labels": objects}
    tikz = nested_data_to_tikz(data, **options)
    tikz_texts = sorted(re.findall(r"\\node\[(?:attr|obj), (?:above|below)\] at .*? \{(.*)\};",
                                   tikz))
    assert matplotlib_labels(data, **options) == tikz_texts


@given(nested_views())
def test_static_nested_figures_label_each_object_once_at_its_place(view):
    from conceptflow.visualization.static import _nested_geometry

    data_frame, root, inner, inner_scales = view
    data = exploration_view_to_nested_data(root, timeout_ms=50)
    tikz = nested_data_to_tikz(data, show_object_labels=True, max_label_items=None)
    outer_context = root.context
    outer_index = {node["id"]: k for k, node in enumerate(data["nodes"])}
    outer_intent = {k: frozenset(node["metadata"]["intent"])
                    for k, node in enumerate(data["nodes"])}
    places = {}
    for match in re.finditer(r"\\node\[obj, below\] at \((\w+)\.south\) \{(.*)\};", tikz):
        for name in match.group(2).split(", "):
            assert name not in places, f"{name} labelled twice"
            places[name] = match.group(1)
    assert sorted(places) == sorted(data_frame.index)

    geometry = _nested_geometry(data)
    for name, place in places.items():
        g = outer_context.objects.index(name)
        own_outer = frozenset(outer_context.attributes[j]
                              for j in outer_context.object_derivation([g]))
        if geometry["template"] is None:
            k = int(place[1:])
            assert place.startswith("o") and outer_intent[k] == own_outer, (name, place)
            continue
        k, j = (int(part) for part in place[1:].split("x"))
        assert outer_intent[k] == own_outer, (name, place)
        # The inner node: the object's own intent in the inner scale.
        inner_context = ConceptualScaler(scales=inner_scales, output="context").fit_transform(
            data_frame
        )
        own_inner = frozenset(inner_context.attributes[m]
                              for m in inner_context.object_derivation([g]))
        node = data["nodes"][k]
        inner_ids = list(geometry["inner_places"][node["id"]])
        template = {n["id"]: frozenset(n["intent"])
                    for n in geometry["template"]["nodes"]}
        assert template[inner_ids[j]] == own_inner, (name, place)
    assert outer_index  # the figure has outer nodes
