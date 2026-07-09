"""
Small HTML figure wrapper for custom visualizations.
"""

from __future__ import annotations

import os
import webbrowser
from pathlib import Path


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

    def __init__(self, html: str):
        self.html = html

    def _repr_html_(self) -> str:
        return self.html

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
        from IPython.display import HTML, display

        display(HTML(self.html))

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