"""
Graphviz (DOT) export of concept lattices.

``lattice_to_dot`` writes DOT text with no extra dependency; Graphviz then
lays out and renders it (e.g. ``dot -Tpdf lattice.dot -o lattice.pdf``).
``GraphvizFigure`` wraps the text and renders it through the ``dot``
program when that is installed, and shows inline in Jupyter.
"""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path

from conceptflow.core import ConceptLattice

_ATTRIBUTE_COLOR = "#1f4e79"
_OBJECT_COLOR = "#333333"


def _html_escape(text: str) -> str:
    # Backslashes are doubled: Graphviz expands escapes such as \G (graph
    # name) even inside HTML-like labels, and "\\" is its literal backslash.
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("\\", "\\\\")
    )


def _join(items: list[str], max_items: int | None) -> str:
    if max_items is not None and len(items) > max_items:
        items = [*items[:max_items], f"+{len(items) - max_items}"]
    return ", ".join(items)


def reduced_labels(
    lattice: ConceptLattice,
) -> tuple[dict, dict]:
    """
    Reduced labelling: attribute ``m`` at μ(m), object ``g`` at γ(g).

    Returns two dicts, concept -> attribute names and concept -> object
    names. On an iceberg lattice, labels whose concept lies below the
    iceberg are left out.
    """
    attributes: dict = {}
    objects: dict = {}

    for name in lattice.context.attributes:
        try:
            attributes.setdefault(lattice.attribute_concept(name), []).append(name)
        except ValueError:
            pass

    for name in lattice.context.objects:
        try:
            objects.setdefault(lattice.object_concept(name), []).append(name)
        except ValueError:
            pass

    return attributes, objects


def lattice_to_dot(
    lattice: ConceptLattice,
    show_attribute_labels: bool = True,
    show_object_labels: bool = True,
    max_label_items: int | None = 6,
    object_label: Callable[[str], str] | None = None,
    title: str | None = None,
) -> str:
    """
    DOT source for a concept lattice with reduced labels.

    The top concept is drawn at the top. Graphviz computes the layout, so
    this works for lattices far too large for DimFlux.
    """
    attributes_at, objects_at = reduced_labels(lattice)
    format_object = object_label or (lambda name: name)
    node_id = {concept: f"c{i}" for i, concept in enumerate(lattice.concepts)}

    lines = [
        "digraph lattice {",
        "  rankdir=BT;",
        '  node [shape=circle, width=0.15, height=0.15, fixedsize=true, label="", '
        "style=filled, fillcolor=white];",
        "  edge [arrowhead=none];",
        "  forcelabels=true;",
    ]
    if title:
        # An HTML-like label: backslashes and quotes in the title stay literal.
        lines.append(f"  label=<{_html_escape(title)}>; labelloc=t;")

    for concept, identifier in node_id.items():
        parts = []
        if show_attribute_labels and concept in attributes_at:
            text = _join(sorted(attributes_at[concept]), max_label_items)
            if text.strip():  # Graphviz rejects an empty <FONT></FONT>
                parts.append(f'<FONT COLOR="{_ATTRIBUTE_COLOR}">{_html_escape(text)}</FONT>')
        if show_object_labels and concept in objects_at:
            names = sorted(format_object(name) for name in objects_at[concept])
            text = _join(names, max_label_items)
            if text.strip():
                parts.append(f'<FONT COLOR="{_OBJECT_COLOR}">{_html_escape(text)}</FONT>')
        xlabel = f" [xlabel=<{'<BR/>'.join(parts)}>]" if parts else ""
        lines.append(f"  {identifier}{xlabel};")

    for lower, upper in lattice.edges:
        lines.append(f"  {node_id[lower]} -> {node_id[upper]};")

    lines.append("}")
    return "\n".join(lines) + "\n"


class GraphvizFigure:
    """
    DOT source with rendering helpers.

    - ``source``: the DOT text.
    - ``write(path)``: ``.dot``/``.gv`` writes the text; any other suffix
      (``.pdf``, ``.svg``, ``.png``, ...) renders it with Graphviz's ``dot``.
    - Shows as SVG in Jupyter when ``dot`` is installed.
    """

    def __init__(self, source: str):
        self.source = source

    def __str__(self) -> str:
        return self.source

    @staticmethod
    def _dot_executable() -> str:
        dot = shutil.which("dot")
        if dot is None:
            raise RuntimeError(
                "Rendering needs Graphviz's `dot` program (e.g. `apt install "
                "graphviz` or `brew install graphviz`). The DOT text itself is "
                "available as `.source`."
            )
        return dot

    def render(self, fmt: str = "svg", timeout: float | None = None) -> bytes:
        """Render with Graphviz and return the output bytes."""
        result = subprocess.run(
            [self._dot_executable(), f"-T{fmt}"],
            input=self.source.encode("utf-8"),
            capture_output=True,
            check=True,
            timeout=timeout,
        )
        return result.stdout

    def write(self, path: str | Path) -> Path:
        """Write DOT text (``.dot``/``.gv``) or a rendered file (other suffixes)."""
        path = Path(path)
        suffix = path.suffix.lower().lstrip(".")
        if suffix in {"dot", "gv", ""}:
            path.write_text(self.source, encoding="utf-8")
        else:
            path.write_bytes(self.render(suffix))
        return path

    def _repr_svg_(self) -> str | None:
        # Notebook preview: give up (and show nothing) rather than hang on a
        # lattice too large for Graphviz.
        if shutil.which("dot") is None:
            return None
        try:
            return self.render("svg", timeout=30).decode("utf-8")
        except subprocess.TimeoutExpired:
            return None
