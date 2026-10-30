import numpy as np

from conceptflow import ConceptLattice, FormalContext
from conceptflow.visualization import (
    HTMLFigure,
    graph_data_to_d3_data,
    lattice_to_graph_data,
    plot_lattice,
    render_graph_data_html,
    render_with_d3,
)


def make_lattice():
    ctx = FormalContext.from_array(
        np.array([
            [1, 0],
            [1, 1],
        ]),
        objects=["g1", "g2"],
        attributes=["m1", "m2"],
    )

    return ConceptLattice.from_context(ctx)


def test_graph_data_to_d3_data_contains_nodes_edges_and_metadata():
    lattice = make_lattice()
    graph_data = lattice_to_graph_data(lattice)

    data = graph_data_to_d3_data(graph_data)

    assert "nodes" in data
    assert "edges" in data
    assert "metadata" in data
    assert len(data["nodes"]) == lattice.n_concepts
    assert len(data["edges"]) == len(lattice.edges)


def test_render_graph_data_html_returns_html_string():
    lattice = make_lattice()
    graph_data = lattice_to_graph_data(lattice)

    html = render_graph_data_html(graph_data)

    assert "<!DOCTYPE html>" in html
    assert "d3@7" in html
    assert "Concept lattice" in html


def test_render_with_d3_returns_html_figure():
    lattice = make_lattice()
    graph_data = lattice_to_graph_data(lattice)

    fig = render_with_d3(graph_data)

    assert isinstance(fig, HTMLFigure)
    assert "Concept lattice" in fig.html


def test_plot_lattice_d3_backend_returns_html_figure():
    lattice = make_lattice()

    fig = plot_lattice(
        lattice,
        backend="d3",
        layout="none",
    )

    assert isinstance(fig, HTMLFigure)
    assert "Concept lattice" in fig.html

def test_render_graph_data_html_accepts_custom_title():
    lattice = make_lattice()
    graph_data = lattice_to_graph_data(lattice)

    html = render_graph_data_html(
        graph_data,
        title="My lattice",
    )

    assert "My lattice" in html


def test_plot_lattice_d3_backend_accepts_custom_size_and_title():
    lattice = make_lattice()

    fig = plot_lattice(
        lattice,
        backend="d3",
        layout="none",
        width=1000,
        height=800,
        title="Custom lattice",
    )

    assert isinstance(fig, HTMLFigure)
    assert "Custom lattice" in fig.html
    assert "width=\"720\"" in fig.html
    assert "min-height: 800px" in fig.html


def _page_data(html):
    """The JSON the page draws from."""
    import json
    import re

    return json.loads(re.search(r"const graphData = (.*);\n", html).group(1))


def test_d3_uses_reduced_labels():
    ctx = FormalContext.from_array(
        np.array([[1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]]),
        objects=["g1", "g2", "g3", "g4"],
        attributes=["a", "b", "c"],
    )
    lattice = ConceptLattice.from_context(ctx)
    graph_data = lattice_to_graph_data(lattice)
    nodes = _page_data(render_graph_data_html(graph_data))["nodes"]
    for node, concept in zip(nodes, lattice.concepts):
        attributes = [
            a for j, a in enumerate(ctx.attributes) if lattice.attribute_concept(j) == concept
        ]
        objects = [
            g for i, g in enumerate(ctx.objects) if lattice.object_concept(i) == concept
        ]
        assert node["attribute_text"] == ", ".join(sorted(attributes))
        assert node["object_text"] == ", ".join(sorted(objects))
    assert "labelForNode(d));" not in render_graph_data_html(graph_data)


def test_d3_label_options():
    graph_data = lattice_to_graph_data(make_lattice())
    nodes = _page_data(
        render_graph_data_html(graph_data, show_attribute_labels=False, object_label=str.upper)
    )["nodes"]
    assert all(node["attribute_text"] == "" for node in nodes)
    assert sorted(node["object_text"] for node in nodes if node["object_text"]) == ["G1", "G2"]
    many = FormalContext.from_array(np.ones((1, 8)), objects=["g"], attributes=list("abcdefgh"))
    html = plot_lattice(ConceptLattice.from_context(many), layout="layered", max_label_items=2).html
    assert "a, b, +6" in html
