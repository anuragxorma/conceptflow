"""
Invalid plotting options fail with a clear error instead of drawing nonsense,
and labels reach the page as given.
"""

import numpy as np
import pandas as pd
import pytest

from conceptflow import ConceptLattice, FormalContext, ManyValuedContext, plot_lattice
from conceptflow.exploration import ExplorationBuilder
from conceptflow.preprocessing import NominalScale
from conceptflow.visualization import lattice_to_graph_data, layered_layout, plot_nested
from tests.test_nested_combinations import make_nested_view


def make_lattice():
    return ConceptLattice.from_context(
        FormalContext.from_array(
            np.array([[1, 1, 0], [1, 0, 1], [0, 1, 1]]),
            objects=["g1", "g2", "g3"],
            attributes=["a", "b", "c"],
        )
    )


@pytest.mark.parametrize("width", [0, -5])
def test_html_width_must_be_positive(width):
    with pytest.raises(ValueError, match="width"):
        plot_lattice(make_lattice(), layout="layered", width=width)
    with pytest.raises(ValueError, match="width"):
        plot_nested(make_nested_view(), width=width)


@pytest.mark.parametrize("width, drawing_width", [(100, 100), (559, 559), (560, 280), (900, 620)])
def test_narrow_pages_put_the_details_panel_below_the_drawing(width, drawing_width):
    for figure, svg_id in (
        (plot_lattice(make_lattice(), layout="layered", width=width), "conceptflow-svg"),
        (plot_nested(make_nested_view(), width=width), "conceptflow-nested-svg"),
    ):
        assert f'<svg id="{svg_id}" width="{drawing_width}"' in figure.html
        assert ("border-top" in figure.html) == (width < 560)


def test_html_size_rejects_text_and_zero_height():
    with pytest.raises(TypeError, match="width"):
        plot_lattice(make_lattice(), layout="layered", width="900")
    with pytest.raises(ValueError, match="height"):
        plot_lattice(make_lattice(), layout="layered", height=0)


@pytest.mark.parametrize("backend", ["matplotlib", "tikz", "graphviz"])
@pytest.mark.parametrize(
    "value, error", [(-1, ValueError), (2.5, TypeError), ("3", TypeError)]
)
def test_max_label_items_is_validated(backend, value, error):
    if backend == "matplotlib":
        pytest.importorskip("matplotlib")
    with pytest.raises(error, match="max_label_items"):
        plot_lattice(make_lattice(), backend=backend, layout="layered", max_label_items=value)


@pytest.mark.parametrize("backend", ["matplotlib", "tikz", "graphviz"])
def test_object_label_may_return_any_value(backend):
    if backend == "matplotlib":
        pytest.importorskip("matplotlib").use("Agg")
    plot_lattice(make_lattice(), backend=backend, layout="layered", object_label=len)


@pytest.mark.parametrize("width_cm", [0, -5])
def test_tikz_width_must_be_positive(width_cm):
    with pytest.raises(ValueError, match="width_cm"):
        plot_lattice(make_lattice(), backend="tikz", layout="layered", width_cm=width_cm)


@pytest.mark.parametrize(
    "options",
    [{"x_spacing": 0}, {"x_spacing": -10}, {"y_spacing": 0}, {"y_spacing": -5}, {"sweeps": -1}],
)
def test_layered_layout_rejects_invalid_spacing(options):
    with pytest.raises(ValueError, match=next(iter(options))):
        layered_layout(lattice_to_graph_data(make_lattice()), **options)


def test_nested_page_shows_attribute_names_verbatim():
    outer = FormalContext.from_array(
        [[1, 0, 1], [0, 1, 1], [1, 1, 0]],
        objects=["p", "q", "r"],
        attributes=["color!=red", "a=yes", "has_limbs"],
    )
    inner = FormalContext.from_array([[1], [0], [1]], objects=["p", "q", "r"], attributes=["b=yes"])
    html = plot_nested(outer, inner=inner).html
    assert "formatAttrLabel" not in html
    assert '"color!=red"' in html and '"has_limbs"' in html


def test_nested_preview_lets_clicks_through_to_the_bubble():
    html = plot_nested(make_nested_view()).html
    assert ".child-preview {\n    pointer-events: none;" in html


def test_expand_all_rejects_scales_that_are_not_a_list_of_scales():
    df = pd.DataFrame({"region": ["EU", "US"], "sector": ["energy", "tech"]}, index=["g1", "g2"])
    builder = ExplorationBuilder(ManyValuedContext.from_dataframe(df))
    root = builder.root("region", [NominalScale("region")])
    with pytest.raises(TypeError, match="scales"):
        builder.expand_all(root, "sector", [NominalScale("sector")])
    with pytest.raises(TypeError, match="scales"):
        builder.root("region", NominalScale("region"))


@pytest.mark.parametrize(
    "name, shown",
    [
        ("T4: (0,0),(2,0),(1,\\sqrt{3})", "T4: (0,0),(2,0),(1,√3)"),
        ("$\\alpha \\leq 5$", "α ≤ 5"),
        ("\\sqrt{x+1}", "√(x+1)"),
        ("10^{-3}", "10⁻³"),
        ("has_limbs", "has_limbs"),
        ("item_1", "item_1"),
        ("price $5", "price $5"),
        ("a\\b", "a\\b"),
    ],
)
def test_latex_in_names_is_shown_as_symbols(name, shown):
    from conceptflow.visualization._text import display_text

    assert display_text(name) == shown


def test_latex_in_names_in_every_backend():
    from conceptflow.datasets import load_context

    lattice = ConceptLattice.from_context(load_context("triangles_en"))
    assert "(1,\\u221a3)" in plot_lattice(lattice, layout="layered").html
    assert "(1,√3)" in plot_lattice(lattice, backend="graphviz").source
    assert "(1,$\\sqrt{3}$)" in plot_lattice(lattice, backend="tikz", layout="layered").source


def test_latex_in_names_on_the_interactive_nested_page():
    outer = FormalContext.from_array(
        [[1, 0], [0, 1], [1, 1]], objects=["x^2", "\\sqrt{2}", "p"], attributes=["\\alpha", "b_1"]
    )
    inner = FormalContext.from_array(
        [[1], [0], [1]], objects=["x^2", "\\sqrt{2}", "p"], attributes=["$\\leq 5$"]
    )
    html = plot_nested(outer, inner=inner).html
    assert "\\\\alpha" not in html and "\\\\sqrt" not in html
    assert all(s in html for s in ("\\u03b1", "\\u2264 5", "x\\u00b2", "\\u221a2", "b_1"))


@pytest.mark.parametrize("name", ["debug_filled_pairs", "debug_bottom_outer"])
def test_debug_printers_are_deprecated(name, capsys):
    import conceptflow.visualization as visualization

    with pytest.warns(DeprecationWarning, match=rf"{name} is deprecated.*0\.6\.0"):
        getattr(visualization, name)(make_nested_view())
    assert capsys.readouterr().out  # still works until it is removed
