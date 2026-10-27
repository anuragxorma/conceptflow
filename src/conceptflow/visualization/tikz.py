"""
TikZ/LaTeX export of concept lattices and nested line diagrams.

The output is plain TikZ (no special LaTeX packages) with one style per kind
of element, so the look can be adjusted in the ``.tex`` file:

- ``concept``: lattice nodes, ``edge``: cover lines,
- ``attr`` / ``obj``: attribute labels above, object labels below,
- ``outer`` / ``outeredge`` / ``inner`` / ``filled`` / ``inneredge``:
  nested diagrams.

The geometry is the same as the matplotlib export.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from conceptflow.visualization.graph_data import GraphData

_LATEX_SPECIALS = {
    "\\": r"\textbackslash{}",
    "{": r"\{",
    "}": r"\}",
    "$": r"\$",
    "&": r"\&",
    "#": r"\#",
    "^": r"\textasciicircum{}",
    "_": r"\_",
    "%": r"\%",
    "~": r"\textasciitilde{}",
}


def latex_escape(text: str) -> str:
    """Escape LaTeX special characters; newlines and other control
    characters become spaces (a blank line would end the paragraph)."""
    return "".join(
        " " if ch.isspace() or ord(ch) < 32 else _LATEX_SPECIALS.get(ch, ch)
        for ch in str(text)
    )


def _join(items: list[str], max_items: int | None) -> str:
    if max_items is not None and len(items) > max_items:
        items = [*items[:max_items], f"+{len(items) - max_items}"]
    return ", ".join(latex_escape(item) for item in items)


def _scaling(points: dict, width_cm: float):
    """
    The map that centres ``points`` and scales them to ``width_cm`` (along
    the longer side), and its scale factor.
    """
    xs = [x for x, _ in points.values()] or [0.0]
    ys = [y for _, y in points.values()] or [0.0]
    span = max(max(xs) - min(xs), max(ys) - min(ys))
    factor = width_cm / span if span > 0 else 1.0
    cx, cy = (max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2

    def transform(point):
        return ((point[0] - cx) * factor, (point[1] - cy) * factor)

    return transform, factor


def _fmt(value: float) -> str:
    return f"{value:.3f}".rstrip("0").rstrip(".") if value else "0"


_LATTICE_STYLES = r"""  concept/.style={circle, draw, fill=white, inner sep=0pt, minimum size=5pt},
  edge/.style={draw=black!70},
  attr/.style={font=\scriptsize, text=blue!60!black, inner sep=1pt},
  obj/.style={font=\scriptsize, text=black!80, inner sep=1pt},"""

_NESTED_STYLES = r"""  outer/.style={circle, draw, thick, fill=white, inner sep=0pt},
  outeredge/.style={draw=black, thick},
  inner/.style={circle, draw=black!60, fill=white, inner sep=0pt, minimum size=3pt},
  filled/.style={circle, draw=blue!50!black, fill=blue!50!black, inner sep=0pt, minimum size=3pt},
  inneredge/.style={draw=black!50, very thin},
  attr/.style={font=\scriptsize, text=blue!60!black, inner sep=1pt},
  obj/.style={font=\tiny, text=black!80, inner sep=1pt},"""


def standalone_document(picture: str) -> str:
    """
    Wrap a ``tikzpicture`` into a document that compiles on its own.

    T1 font encoding covers accented Latin letters with pdflatex. Names in
    other scripts need a matching LaTeX setup (e.g. lualatex with suitable
    fonts); ``TikZFigure.write("x.pdf")`` uses lualatex automatically for
    non-ASCII names when it is installed.
    """
    return (
        "\\documentclass[tikz, border=4pt]{standalone}\n"
        "\\usepackage[T1]{fontenc}\n"
        "\\begin{document}\n" + picture + "\\end{document}\n"
    )


def _wrap(body: list[str], styles: str, standalone: bool, title: str | None = None) -> str:
    if title:
        body = [
            *body,
            "\\node[font=\\bfseries, above] at (current bounding box.north) "
            f"{{{latex_escape(title)}}};",
        ]
    picture = (
        "\\begin{tikzpicture}[\n" + styles + "\n]\n" + "\n".join(body) + "\n\\end{tikzpicture}\n"
    )
    return standalone_document(picture) if standalone else picture


def graph_data_to_tikz(
    graph_data: GraphData,
    width_cm: float = 12.0,
    show_attribute_labels: bool = True,
    show_object_labels: bool = True,
    max_label_items: int | None = 6,
    object_label: Callable[[str], str] | None = None,
    standalone: bool = False,
    title: str | None = None,
) -> str:
    """
    TikZ code for a lattice ``GraphData`` (from any layout).

    Nodes without coordinates get the layered layout. Reduced labels:
    attributes above the node introducing them, objects below.
    """
    from conceptflow.visualization.static import _lattice_positions, _reduced_labels

    graph_data, positions = _lattice_positions(graph_data)
    transform, _ = _scaling(positions, width_cm)
    positions = {node_id: transform(point) for node_id, point in positions.items()}
    attributes_at, objects_at = _reduced_labels(graph_data)
    format_object = object_label or (lambda name: name)
    name = {node.node_id: f"c{i}" for i, node in enumerate(graph_data.nodes)}

    body = []
    for node in graph_data.nodes:
        x, y = positions[node.node_id]
        labels = []
        if show_attribute_labels and attributes_at[node.node_id]:
            text = _join(sorted(attributes_at[node.node_id]), max_label_items)
            labels.append(f"label={{[attr]above:{{{text}}}}}")
        if show_object_labels and objects_at[node.node_id]:
            names = sorted(format_object(o) for o in objects_at[node.node_id])
            text = _join(names, max_label_items)
            labels.append(f"label={{[obj]below:{{{text}}}}}")
        options = ", ".join(["concept", *labels])
        body.append(f"\\node[{options}] ({name[node.node_id]}) at ({_fmt(x)}, {_fmt(y)}) {{}};")

    for edge in graph_data.edges:
        body.append(f"\\draw[edge] ({name[edge.source]}) -- ({name[edge.target]});")

    return _wrap(body, _LATTICE_STYLES, standalone, title)


def nested_data_to_tikz(
    data: dict[str, Any],
    width_cm: float = 14.0,
    show_attribute_labels: bool = True,
    show_object_labels: bool = False,
    max_label_items: int | None = 4,
    standalone: bool = False,
    title: str | None = None,
) -> str:
    """
    TikZ code for a two-level nested line diagram.

    ``data`` is the output of ``exploration_view_to_nested_data``. Realized
    (outer, inner) pairs are drawn filled, unrealized ones hollow.
    """
    from conceptflow.visualization.static import _nested_geometry

    geometry = _nested_geometry(data)
    scale_point, factor = _scaling(geometry["centers"], width_cm)
    centers = {node_id: scale_point(point) for node_id, point in geometry["centers"].items()}
    radius = geometry["bubble"] * factor

    body = []
    for (start, end) in geometry["outer_segments"]:
        (x1, y1), (x2, y2) = scale_point(start), scale_point(end)
        body.append(f"\\draw[outeredge] ({_fmt(x1)}, {_fmt(y1)}) -- ({_fmt(x2)}, {_fmt(y2)});")

    template = geometry["template"]
    for k, node in enumerate(data["nodes"]):
        cx, cy = centers[node["id"]]
        body.append(
            f"\\node[outer, minimum size={_fmt(2 * radius)}cm] (o{k}) "
            f"at ({_fmt(cx)}, {_fmt(cy)}) {{}};"
        )
        if show_attribute_labels and node["new_attrs"]:
            body.append(
                f"\\node[attr, above] at (o{k}.north) "
                f"{{{_join(node['new_attrs'], max_label_items)}}};"
            )

        if not template:
            if show_object_labels and node["new_objects"]:
                body.append(
                    f"\\node[obj, below] at (o{k}.south) "
                    f"{{{_join(node['new_objects'], max_label_items)}}};"
                )
            continue

        places = {iid: scale_point(p) for iid, p in geometry["inner_places"][node["id"]].items()}
        inner_name = {iid: f"i{k}x{j}" for j, iid in enumerate(places)}
        filled = set(node["filled_inner"])

        for iid, (x, y) in places.items():
            style = "filled" if iid in filled else "inner"
            body.append(f"\\node[{style}] ({inner_name[iid]}) at ({_fmt(x)}, {_fmt(y)}) {{}};")
        for edge in template["edges"]:
            body.append(
                f"\\draw[inneredge] ({inner_name[edge['source']]}) -- "
                f"({inner_name[edge['target']]});"
            )

        if show_attribute_labels and node["id"] == geometry["top_id"]:
            for inode in template["nodes"]:
                if inode["new_attrs"]:
                    body.append(
                        f"\\node[attr, above] at ({inner_name[inode['id']]}.north) "
                        f"{{{_join(inode['new_attrs'], max_label_items)}}};"
                    )

        if show_object_labels and node.get("child"):
            for cnode in node["child"]["nodes"]:
                if cnode["new_objects"]:
                    body.append(
                        f"\\node[obj, below] at ({inner_name[cnode['id']]}.south) "
                        f"{{{_join(cnode['new_objects'], max_label_items)}}};"
                    )

    return _wrap(body, _NESTED_STYLES, standalone, title)


class TikZFigure:
    """
    TikZ source with helpers.

    - ``source``: the ``tikzpicture`` (or a full document if ``standalone``).
    - ``write(path)``: ``.tex`` writes the source (as a standalone document,
      so it compiles on its own); ``.pdf`` compiles it with ``pdflatex``.
    """

    def __init__(self, source: str):
        self.source = source
        self._standalone = (
            source if "\\documentclass" in source else standalone_document(source)
        )

    def __str__(self) -> str:
        return self.source

    def write(self, path: str | Path) -> Path:
        path = Path(path)
        if path.suffix.lower() == ".pdf":
            path.write_bytes(self._compile())
        else:
            path.write_text(self._standalone, encoding="utf-8")
        return path

    def _compile(self) -> bytes:
        non_ascii = any(ord(ch) > 127 for ch in self._standalone)
        if non_ascii and shutil.which("lualatex"):
            latex = shutil.which("lualatex")
        else:
            latex = shutil.which("pdflatex") or shutil.which("lualatex")
        if latex is None:
            raise RuntimeError(
                "Compiling to PDF needs a LaTeX installation (pdflatex). The "
                "TikZ code itself is available as `.source`."
            )
        with tempfile.TemporaryDirectory() as directory:
            tex = Path(directory) / "figure.tex"
            tex.write_text(self._standalone, encoding="utf-8")
            result = subprocess.run(
                [latex, "-interaction=nonstopmode", "-halt-on-error", tex.name],
                cwd=directory,
                capture_output=True,
                text=True,
            )
            pdf = tex.with_suffix(".pdf")
            if result.returncode != 0 or not pdf.exists():
                tail = "\n".join(result.stdout.splitlines()[-15:])
                raise RuntimeError(f"LaTeX failed to compile the figure:\n{tail}")
            return pdf.read_bytes()
