import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from matplotlib.figure import Figure  # noqa: E402

from conceptflow import ConceptLattice, FormalContext, plot_lattice  # noqa: E402
from conceptflow.visualization import plot_nested, render_lattice_matplotlib  # noqa: E402
from conceptflow.visualization.static import _upright  # noqa: E402
from tests.test_nested_combinations import make_nested_view  # noqa: E402


def make_lattice():
    return ConceptLattice.from_context(
        FormalContext.from_array(
            np.array([[1, 1, 0], [1, 0, 1], [0, 1, 1]]),
            attributes=["a", "b", "c"],
        )
    )


@pytest.mark.parametrize("suffix", ["svg", "pdf", "png"])
def test_plot_lattice_matplotlib_saves_static_files(tmp_path, suffix):
    fig = plot_lattice(make_lattice(), backend="matplotlib", layout="none")

    assert isinstance(fig, Figure)
    path = tmp_path / f"lattice.{suffix}"
    fig.savefig(path)
    assert path.stat().st_size > 0


def test_plot_lattice_matplotlib_draws_reduced_labels():
    fig = plot_lattice(make_lattice(), backend="matplotlib", layout="none")
    texts = {t.get_text() for t in fig.axes[0].texts}

    assert {"a", "b", "c", "g0", "g1", "g2"} <= texts


def test_plot_lattice_matplotlib_draws_on_given_axes():
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    assert plot_lattice(make_lattice(), backend="matplotlib", layout="none", ax=ax) is fig


def test_plot_nested_matplotlib(tmp_path):
    fig = plot_nested(make_nested_view(), backend="matplotlib", show_object_labels=True)

    assert isinstance(fig, Figure)
    fig.savefig(tmp_path / "nested.pdf")
    texts = {t.get_text() for t in fig.axes[0].texts}
    assert "kind=x" in texts and "size>=M" in texts


def test_plot_nested_rejects_unknown_backend():
    with pytest.raises(ValueError, match="backend"):
        plot_nested(make_nested_view(), backend="powerpoint")


def test_upright_puts_top_above_bottom():
    positions = {"top": (3.0, -1.0), "bottom": (-2.0, 4.0), "mid": (0.0, 0.0)}
    rotated = _upright(positions, "top", "bottom")

    (tx, ty), (bx, by) = rotated["top"], rotated["bottom"]
    assert tx == pytest.approx(bx)
    assert ty > by


def test_static_lattice_matches_interactive_orientation():
    # Same context the audit used to show a left-right mirror.
    lattice = ConceptLattice.from_context(
        FormalContext.from_array(
            np.array([[1, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 0, 0], [0, 0, 1, 1]]),
            attributes=list("abcd"),
        )
    )
    # One layout for both: DimFlux is time-limited, so two separate runs can
    # differ slightly on a slow machine, which is not what is tested here.
    graph_data = plot_lattice(lattice, backend="graph_data")
    fig = render_lattice_matplotlib(graph_data)

    static_x = fig.axes[0].collections[1].get_offsets()[:, 0]
    interactive_x = [node.x for node in graph_data.nodes]
    # The upright rotation may only add a constant offset, never mirror.
    assert np.allclose(
        static_x - static_x[0], np.subtract(interactive_x, interactive_x[0]), atol=1e-6
    )


@pytest.mark.parametrize("data", [np.tril(np.ones((5, 5))), np.array([[1, 0]])])
def test_chains_do_not_render_blank(data):
    lattice = ConceptLattice.from_context(FormalContext.from_array(data))
    fig = plot_lattice(lattice, backend="matplotlib")
    fig.canvas.draw()

    assert fig.axes[0].get_window_extent().width > 50


def test_layered_layout_handles_very_deep_lattices():
    from conceptflow.visualization.graph_data import GraphData, GraphEdge, GraphNode
    from conceptflow.visualization.layered_layout import layered_layout

    # Bottom-first node order used to recurse once per level.
    n = 3000
    nodes = [
        GraphNode(f"n{i}", "", "", concept=None, metadata={}) for i in reversed(range(n))
    ]
    edges = [GraphEdge(f"n{i + 1}", f"n{i}") for i in range(n - 1)]

    laid_out = layered_layout(GraphData(nodes=nodes, edges=edges)).node_map()

    # Screen coordinates: the top (n0) has the smallest y.
    assert laid_out["n0"].y < laid_out[f"n{n - 1}"].y


def test_label_truncation_and_default_title():
    from conceptflow.visualization.static import _join_labels

    assert _join_labels(["a", "b", "c"], 0) == "+3"
    assert _join_labels(["a", "b", "c"], 2) == "a, b, +1"

    fig = plot_lattice(make_lattice(), backend="matplotlib", layout="none")
    assert fig.axes[0].get_title() == ""


def test_static_nested_only_lays_out_drawn_levels():
    from conceptflow.visualization.nested import exploration_view_to_nested_data

    data = exploration_view_to_nested_data(make_nested_view(), max_depth=1)

    assert data["inner_template"] is not None
    for node in data["nodes"]:
        for inner in node["child"]["nodes"]:
            assert inner["child"] is None
