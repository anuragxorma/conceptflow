"""
Tests for nested visualization data.

These tests do not check actual D3 rendering yet.

They check the intermediate data structure:

    ExplorationView tree
        -> nested JSON-like dictionary

This is the safe bridge between the mathematical exploration layer and the
future renderer.
"""

import json

import pandas as pd
import pytest

from conceptflow import ManyValuedContext
from conceptflow.exploration import ExplorationBuilder
from conceptflow.preprocessing import NominalScale
from conceptflow.visualization.html_figure import HTMLFigure
from conceptflow.visualization.nested import (
    exploration_view_to_nested_data,
    plot_nested,
    render_nested_html,
)


def make_region_mvc():
    df = pd.DataFrame(
        {
            "region": ["EU", "EU", "US", "US"],
            "sector": ["energy", "tech", "energy", "tech"],
        },
        index=["g1", "g2", "g3", "g4"],
    )
    return ManyValuedContext.from_dataframe(df)


def make_root_view(mvc=None):
    if mvc is None:
        mvc = make_region_mvc()

    builder = ExplorationBuilder(mvc)

    return builder.root(
        name="Region view",
        scales=[NominalScale("region")],
    )


def make_expanded_view(mvc=None):
    if mvc is None:
        mvc = make_region_mvc()

    builder = ExplorationBuilder(mvc)

    root = builder.root(
        name="Region view",
        scales=[NominalScale("region")],
    )

    candidate = next(
        concept
        for concept in root.lattice.concepts
        if 0 < len(concept.extent) < len(mvc.objects)
    )

    builder.expand(
        parent=root,
        concept=candidate,
        name="Sector detail",
        scales=[NominalScale("sector")],
    )

    return root


# Backward-readable helper name for preview tests.
def make_root_view_with_child():
    return make_expanded_view()


# ---------------------------------------------------------------------------
# exploration_view_to_nested_data — structure
# ---------------------------------------------------------------------------


def test_nested_data_has_required_top_level_keys():
    data = exploration_view_to_nested_data(make_root_view())

    for key in ("name", "depth", "scale_names", "nodes", "edges", "metadata"):
        assert key in data


def test_nested_data_name_depth_scale_names():
    data = exploration_view_to_nested_data(make_root_view())

    assert data["name"] == "Region view"
    assert data["depth"] == 0
    assert data["scale_names"] == ["region"]


def test_nested_data_node_count_matches_lattice():
    view = make_root_view()
    data = exploration_view_to_nested_data(view)

    assert len(data["nodes"]) == len(view.lattice.concepts)


def test_nested_data_node_fields():
    data = exploration_view_to_nested_data(make_root_view())

    for node in data["nodes"]:
        assert "id" in node
        assert "label" in node
        assert "x" in node
        assert "y" in node
        assert "has_child" in node
        assert "child" in node
        assert isinstance(node["id"], str)
        assert isinstance(node["has_child"], bool)


def test_nested_data_edge_fields_reference_valid_node_ids():
    data = exploration_view_to_nested_data(make_root_view())
    node_ids = {node["id"] for node in data["nodes"]}

    for edge in data["edges"]:
        assert "source" in edge
        assert "target" in edge
        assert edge["source"] in node_ids
        assert edge["target"] in node_ids


def test_nested_data_metadata_fields():
    view = make_root_view()
    data = exploration_view_to_nested_data(view)
    metadata = data["metadata"]

    assert "context_size" in metadata
    assert "n_concepts" in metadata
    assert "n_children" in metadata
    assert metadata["n_concepts"] == view.lattice.n_concepts
    assert metadata["n_children"] == 0
    assert metadata["context_size"] == (
        view.context.n_objects,
        view.context.n_attributes,
    )


def test_nested_data_no_child_nodes_when_unexpanded():
    data = exploration_view_to_nested_data(make_root_view())

    assert all(not node["has_child"] for node in data["nodes"])
    assert all(node["child"] is None for node in data["nodes"])


# ---------------------------------------------------------------------------
# exploration_view_to_nested_data — child view
# ---------------------------------------------------------------------------


def test_nested_data_contains_child_data():
    root = make_expanded_view()
    data = exploration_view_to_nested_data(root)

    child_nodes = [
        node
        for node in data["nodes"]
        if node["has_child"]
    ]

    assert len(child_nodes) == 1

    child_node = child_nodes[0]
    assert child_node["child"] is not None
    assert child_node["child"]["name"] == "Sector detail"
    assert child_node["child"]["depth"] == 1
    assert child_node["child"]["scale_names"] == ["sector"]


def test_nested_data_child_has_own_nodes_and_edges():
    root = make_expanded_view()
    data = exploration_view_to_nested_data(root)

    child = next(
        node["child"]
        for node in data["nodes"]
        if node["has_child"]
    )

    assert len(child["nodes"]) > 0
    assert isinstance(child["edges"], list)


def test_nested_data_child_metadata_n_children_zero():
    root = make_expanded_view()
    data = exploration_view_to_nested_data(root)

    child = next(
        node["child"]
        for node in data["nodes"]
        if node["has_child"]
    )

    assert child["metadata"]["n_children"] == 0


def test_nested_data_root_metadata_n_children_one():
    root = make_expanded_view()
    data = exploration_view_to_nested_data(root)

    assert data["metadata"]["n_children"] == 1


# ---------------------------------------------------------------------------
# render_nested_html
# ---------------------------------------------------------------------------


def test_render_nested_html_returns_string():
    data = exploration_view_to_nested_data(make_root_view())
    html = render_nested_html(data)

    assert isinstance(html, str)


def test_render_nested_html_contains_d3_script():
    data = exploration_view_to_nested_data(make_root_view())
    html = render_nested_html(data)

    assert "d3@7" in html or "d3.js" in html or "d3.min.js" in html or "d3@" in html


def test_render_nested_html_embeds_json_data():
    data = exploration_view_to_nested_data(make_root_view())
    html = render_nested_html(data)

    assert "rootData" in html
    assert json.dumps(data["name"]) in html


def test_render_nested_html_custom_dimensions():
    data = exploration_view_to_nested_data(make_root_view())
    html = render_nested_html(data, width=1200, height=800)

    assert "1200" in html
    assert "800" in html


def test_render_nested_html_contains_back_and_root_buttons():
    data = exploration_view_to_nested_data(make_root_view())
    html = render_nested_html(data)

    assert "Back" in html
    assert "Root" in html


def test_render_nested_html_accepts_custom_title():
    root = make_root_view()
    data = exploration_view_to_nested_data(root)

    html = render_nested_html(
        data,
        title="Custom nested title",
    )

    assert "Custom nested title" in html


def test_render_nested_html_can_disable_child_previews():
    root = make_root_view_with_child()
    data = exploration_view_to_nested_data(root)

    html = render_nested_html(
        data,
        show_child_previews=False,
    )

    assert "const showChildPreviews = false;" in html


# ---------------------------------------------------------------------------
# plot_nested
# ---------------------------------------------------------------------------


def test_plot_nested_returns_html_figure():
    fig = plot_nested(make_root_view())

    assert isinstance(fig, HTMLFigure)


def test_plot_nested_html_figure_has_html():
    fig = plot_nested(make_root_view())

    assert isinstance(fig.html, str)
    assert len(fig.html) > 0


def test_plot_nested_custom_size():
    fig = plot_nested(make_root_view(), width=1000, height=600)

    assert "1000" in fig.html
    assert "600" in fig.html


def test_plot_nested_accepts_custom_title():
    root = make_root_view()

    fig = plot_nested(
        root,
        title="Custom nested title",
    )

    assert "Custom nested title" in fig.html


def test_plot_nested_accepts_navigator_mode():
    root = make_root_view()

    fig = plot_nested(
        root,
        mode="navigator",
    )

    assert "<!DOCTYPE html>" in fig.html


def test_plot_nested_rejects_unknown_mode():
    root = make_root_view()

    with pytest.raises(
        ValueError,
        match='Currently only "navigator" is supported',
    ):
        plot_nested(
            root,
            mode="embedded",
        )


def test_plot_nested_can_disable_child_previews():
    root = make_root_view_with_child()

    fig = plot_nested(
        root,
        show_child_previews=False,
    )

    assert "const showChildPreviews = false;" in fig.html