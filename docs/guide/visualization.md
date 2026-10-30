# Visualization

`plot_lattice(lattice, backend=..., layout=...)` draws a concept lattice with
reduced labelling: each attribute above the node that introduces it, each
object below.

```python
import conceptflow as cf
from conceptflow.datasets import load_context

# Ganter & Wille's "living beings and water", bundled with ConceptFlow.
lattice = cf.ConceptLattice.from_context(load_context("livingbeings_en"))
```

## Backends

| `backend=` | Returns | Use for |
|---|---|---|
| `"d3"` (default) | `HTMLFigure` | interactive exploration in a browser or notebook |
| `"matplotlib"` | matplotlib `Figure` | static PDF/SVG/PNG, subplots |
| `"tikz"` | `TikZFigure` | LaTeX documents |
| `"graphviz"` | `GraphvizFigure` | Graphviz's own layout and renderers |
| `"graph_data"` | `GraphData` | your own renderer |

```python
html = cf.plot_lattice(lattice, layout="layered")
html.write_html("lattice.html")

fig = cf.plot_lattice(lattice, backend="matplotlib", layout="layered", title="Living beings")
fig.savefig("lattice.pdf")

tikz = cf.plot_lattice(lattice, backend="tikz", layout="layered")
tikz.write("lattice.tex")            # standalone LaTeX document
print(tikz.source.splitlines()[0])  # \begin{tikzpicture}[
```

## Layouts

| `layout=` | What it is | When |
|---|---|---|
| `"dimflux"` (default) | DimFlux additive, force-directed drawing | small lattices; the most readable drawings |
| `"layered"` | Sugiyama-style rows with crossing reduction | large lattices: tens of thousands of concepts in seconds |
| `"none"` | no coordinates | renderers that need them fall back to `"layered"` |

Both layouts move a concept slightly sideways when a straight edge would
pass through it, so the drawing never suggests a cover relation that does
not exist; heights are kept, so every edge still points upward.

DimFlux takes seconds for small lattices but can take minutes beyond a few
hundred concepts; ConceptFlow warns above 150 concepts. `timeout_ms`
(default 1000) limits each of its optimization phases loosely, not the total
run time. Because of that time limit, DimFlux coordinates can differ
slightly between runs on a slow or busy machine; for exactly reproducible
coordinates use `layout="layered"` or `timeout_ms=None` (no limit, slower).

```python
fig = cf.plot_lattice(lattice, backend="matplotlib")   # DimFlux layout
fig.savefig("lattice_dimflux.svg")
```

## LaTeX (TikZ)

The TikZ export is plain TikZ with one style per kind of element, so the
look can be changed in the `.tex` file:

- `concept`, `edge`, `attr`, `obj` for lattices;
- `outer`, `outeredge`, `inner`, `filled`, `inneredge` for nested diagrams.

```python
tikz = cf.plot_lattice(lattice, backend="tikz", layout="layered", width_cm=10)
picture = tikz.source        # \begin{tikzpicture} ... \end{tikzpicture}, to paste into a paper
```

LaTeX special characters in names are escaped. `tikz.write("lattice.pdf")`
compiles the figure if `pdflatex` is installed.

## Graphviz

<!-- no-test -->
```python
dot = cf.plot_lattice(lattice, backend="graphviz")
dot.write("lattice.dot")     # DOT text, no dependency
dot.write("lattice.pdf")     # rendered with Graphviz's `dot` program
```

Graphviz computes its own layout. It produces good drawings but becomes slow
for thousands of concepts; `layout="layered"` with matplotlib or TikZ is much
faster there.

## Interactive (d3) options

The interactive page shows the same reduced labels; a concept's full extent
and intent appear in its tooltip and, on click, in the details panel. Large
lattices are drawn with smaller nodes so none overlap; zoom in with the
mouse wheel to bring them back to full size. `show_attribute_labels=`,
`show_object_labels=`, `max_label_items=` and `object_label=` work as for
matplotlib below. `width` includes the 280 px details panel, which moves
below the drawing in pages narrower than 560 px.

## matplotlib options

`render_lattice_matplotlib` options can be passed to `plot_lattice`:
`ax=` (draw into existing axes), `figsize=`, `show_attribute_labels=`,
`show_object_labels=`, `max_label_items=` (truncate long label lists),
`object_label=` (a function shortening object names), `node_size=`,
`font_size=`.

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
cf.plot_lattice(lattice, backend="matplotlib", layout="layered", ax=axes[0])
cf.plot_lattice(lattice, backend="matplotlib", layout="layered", ax=axes[1],
                show_object_labels=False)
fig.savefig("side_by_side.png")
```

Nested line diagrams have the same backends; see
[Nested line diagrams](nested.md).

## Known limitations

- **DimFlux coordinates are not exactly reproducible** on slow or busy
  machines, since its optimization is time-limited (see *Layouts* above);
  `layout="layered"` is deterministic.
- **Crowded layouts.** No edge is drawn through a concept it does not
  connect, but in crowded rows of the layered layout an edge can pass close
  to one: a concept is only moved where it keeps its distance to all others.
  Lattices above 2000 concepts are not adjusted at all.
- **Very large lattices in the interactive page.** Nodes shrink so that none
  overlap; on a few hundred concepts they are tiny until you zoom in.
- **Labels can overlap** in dense diagrams, in every backend. `max_label_items=`,
  `object_label=` and hiding one kind of label (`show_object_labels=False`)
  help; a larger `width`/`height` (or `figsize`) spreads the drawing.
- **The interactive pages need internet access**: they load D3 (and, for the
  nested PDF export, jsPDF and svg2pdf.js) from public CDNs, in exact
  versions.
- **Notebooks.** Each interactive figure is shown in its own frame. In a
  notebook run by someone else, Jupyter does not run its HTML until you trust
  the notebook (*File → Trust Notebook*) or rerun the cells, so the frames
  stay empty until then. A matplotlib figure returned as the last line of a
  cell is shown twice, as for any matplotlib figure; end the line with `;`.
- **LaTeX in names** is shown for common commands only (`\sqrt{3}` → √3,
  Greek letters, `\leq`, `x^2`, ...); anything else stays as written. TikZ
  typesets it in math mode.
- **Nested PDF export** is vector for names in Latin-1; other names (for
  example with √ or α) give an image PDF, since jsPDF's standard fonts cannot
  show them.
