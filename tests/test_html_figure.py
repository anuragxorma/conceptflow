
import pytest

from conceptflow.visualization import HTMLFigure


def test_html_figure_to_html_returns_html_string():
    fig = HTMLFigure("<div>Hello</div>")

    assert fig.to_html() == "<div>Hello</div>"


def test_html_figure_repr_html_isolates_the_page_in_an_iframe():
    fig = HTMLFigure('<div class="x">Hello & bye</div>', height=300)

    shown = fig._repr_html_()

    assert shown.startswith('<iframe srcdoc="&lt;div class=&quot;x&quot;&gt;Hello &amp; bye')
    assert "height: 400px" in shown


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


def test_html_figure_show_displays_without_warnings(monkeypatch):
    pytest.importorskip("IPython")
    import warnings

    import IPython.display

    shown = []
    monkeypatch.setattr(IPython.display, "display", lambda obj: shown.append(obj))
    fig = HTMLFigure("<div>Hello</div>", height=300)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fig.show()
    assert shown == [fig]
