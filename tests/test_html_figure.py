from pathlib import Path

import pytest

from conceptflow.visualization import HTMLFigure


def test_html_figure_to_html_returns_html_string():
    fig = HTMLFigure("<div>Hello</div>")

    assert fig.to_html() == "<div>Hello</div>"


def test_html_figure_repr_html_returns_html_string():
    fig = HTMLFigure("<div>Hello</div>")

    assert fig._repr_html_() == "<div>Hello</div>"


def test_html_figure_write_html_writes_file(tmp_path):
    fig = HTMLFigure("<div>Hello</div>")
    output_path = tmp_path / "figure.html"

    written_path = fig.write_html(output_path)

    assert written_path == output_path.resolve()
    assert output_path.read_text(encoding="utf-8") == "<div>Hello</div>"


def test_html_figure_write_html_respects_overwrite_false(tmp_path):
    fig = HTMLFigure("<div>Hello</div>")
    output_path = tmp_path / "figure.html"

    output_path.write_text("existing", encoding="utf-8")

    with pytest.raises(FileExistsError):
        fig.write_html(output_path, overwrite=False)


def test_html_figure_export_html_backward_compat_alias(tmp_path):
    fig = HTMLFigure("<div>Hello</div>")
    output_path = tmp_path / "figure.html"

    written_path = fig.write_html(output_path)

    assert written_path == output_path.resolve()
    assert output_path.read_text(encoding="utf-8") == "<div>Hello</div>"