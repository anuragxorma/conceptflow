# Visualization

`plot_lattice(lattice, backend=..., layout=...)` draws a concept lattice with
reduced labelling: each attribute above the node that introduces it, each
object below.

```python
import conceptflow as cf
from conceptflow.io import read_cxt

lattice = cf.ConceptLattice.from_context(read_cxt("data/living_beings_and_water.cxt"))
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

DimFlux takes seconds for small lattices but can take minutes beyond a few
hundred concepts; ConceptFlow warns above 150 concepts. `timeout_ms`
(default 1000) limits each of its optimization phases loosely, not the total
run time.

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
