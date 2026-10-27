import shutil

import numpy as np
import pytest

from conceptflow import ConceptLattice, FormalContext, plot_lattice
from conceptflow.visualization import TikZFigure, latex_escape, plot_nested
from tests.test_nested_combinations import make_nested_view

needs_latex = pytest.mark.skipif(
    shutil.which("pdflatex") is None and shutil.which("lualatex") is None,
    reason="LaTeX not installed",
)


def make_lattice():
    return ConceptLattice.from_context(
        FormalContext.from_array(
            np.array([[1, 1, 0], [1, 0, 1], [0, 1, 1]]),
            objects=["frog_1", "50% & more", "x"],
            attributes=["a", "b", "c"],
        )
    )


def test_latex_escape():
    assert latex_escape("a_b & 50% #1 $x$ {y} ~ ^ \\") == (
        r"a\_b \& 50\% \#1 \$x\$ \{y\} \textasciitilde{} \textasciicircum{} "
        r"\textbackslash{}"
    )


@pytest.mark.parametrize("layout", ["layered", "none", "dimflux"])
def test_lattice_tikz_structure(layout):
    lattice = make_lattice()
    fig = plot_lattice(lattice, backend="tikz", layout=layout)

    assert isinstance(fig, TikZFigure)
    assert fig.source.startswith("\\begin{tikzpicture}")
    assert fig.source.rstrip().endswith("\\end{tikzpicture}")
    assert fig.source.count("\\node[concept") == lattice.n_concepts
    assert fig.source.count("\\draw[edge]") == len(lattice.edges)
    assert "frog\\_1" in fig.source and "50\\% \\& more" in fig.source
    assert "\\documentclass" not in fig.source


def test_standalone_document_written_for_tex(tmp_path):
    path = plot_lattice(make_lattice(), backend="tikz", layout="layered").write(tmp_path / "l.tex")
    text = path.read_text()
    assert text.startswith("\\documentclass[tikz")
    assert "\\begin{tikzpicture}" in text and text.rstrip().endswith("\\end{document}")


@needs_latex
def test_lattice_tikz_compiles(tmp_path):
    path = plot_lattice(make_lattice(), backend="tikz", layout="layered").write(tmp_path / "l.pdf")
    assert path.read_bytes().startswith(b"%PDF")


def test_nested_tikz_structure():
    view = make_nested_view()
    fig = plot_nested(view, backend="tikz", show_object_labels=True)

    assert isinstance(fig, TikZFigure)
    n_outer = view.lattice.n_concepts
    assert fig.source.count("\\node[outer") == n_outer
    assert "\\node[filled]" in fig.source


@needs_latex
def test_nested_tikz_compiles(tmp_path):
    fig = plot_nested(make_nested_view(), backend="tikz", show_object_labels=True)
    assert fig.write(tmp_path / "n.pdf").read_bytes().startswith(b"%PDF")
