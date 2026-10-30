"""
Small HTML figure wrapper for custom visualizations.
"""

from __future__ import annotations

import html
import json
import os
import webbrowser
from pathlib import Path


def script_json(value) -> str:
    """
    JSON for a ``<script>`` block. ``<`` is escaped, so names such as
    ``"</script>"`` cannot end the block early.
    """
    return json.dumps(value).replace("<", "\\u003c")


class HTMLFigure:
    """
    Lightweight wrapper around an HTML string.

    Plotly-style methods:
    - to_html()
    - write_html(path)
    - show()

    Backward-compatible aliases:
    - export_html(path)
    - display()
    - open(path)
    """

    def __init__(self, html: str, height: int | None = None):
        self.html = html
        # Expected page height in pixels, for the notebook frame before the
        # page has loaded and measured itself.
        self.height = height

    def _repr_html_(self) -> str:
        """
        The page in its own ``<iframe>``, so that several figures in one
        notebook do not share ids, scripts or styles with each other or with
        the notebook. The frame grows to the page's height once loaded.
        """
        height = (self.height or 700) + 100
        return (
            f'<iframe srcdoc="{html.escape(self.html, quote=True)}" '
            f'style="width: 100%; height: {height}px; border: none;" '
            'onload="this.style.height = '
            "(this.contentDocument.documentElement.scrollHeight + 4) + 'px';\"></iframe>"
        )

    def to_html(self) -> str:
        """
        Return the raw HTML string.
        """
        return self.html

    def write_html(
        self,
        path: str | Path,
        overwrite: bool = True,
    ) -> Path:
        """
        Write the HTML figure to disk.
        """
        output_path = Path(path).expanduser().resolve()

        if output_path.exists() and not overwrite:
            raise FileExistsError(f"File already exists: {output_path}")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.html, encoding="utf-8")

        return output_path

    def show(self) -> None:
        """
        Display the figure in IPython/Jupyter.
        """
        from IPython.display import display

        # display(self) uses _repr_html_; IPython's HTML() would warn about
        # the <iframe>.
        display(self)

    def open(
        self,
        path: str | Path = "conceptflow_figure.html",
        overwrite: bool = True,
    ) -> Path:
        """
        Write the HTML figure to disk and open it in the browser.
        """
        os.environ.setdefault("NO_AT_BRIDGE", "1")

        output_path = self.write_html(path, overwrite=overwrite)
        webbrowser.open(output_path.as_uri(), new=2)

        return output_path

    # Backward-compatible aliases

    def display(self) -> None:
        """
        Alias for show().
        """
        self.show()

    def export_html(
        self,
        path: str | Path,
        overwrite: bool = True,
    ) -> Path:
        """
        Alias for write_html().
        """
        return self.write_html(path, overwrite=overwrite)
