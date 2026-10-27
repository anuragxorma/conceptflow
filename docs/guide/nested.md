# Nested line diagrams

A nested line diagram draws the concept lattice of two attribute sets at
once: each concept of the **outer** lattice is a circle containing a copy of
the **inner** lattice. Filled inner nodes are combinations that occur in the
data; hollow ones never occur. This is the Toscana-style tool for exploring
data along two conceptual dimensions. The [tutorial](../nested_diagram_tutorial.md)
explains the mathematics.

## From two contexts

The quickest way: two formal contexts over the same objects.

```python
import conceptflow as cf
from conceptflow.visualization import plot_nested

objects = ["p1", "p2", "p3", "p4", "p5", "p6"]
diet = cf.FormalContext.from_array(
    [[1, 0], [1, 0], [0, 1], [0, 1], [1, 1], [0, 0]], objects, ["vegetarian", "spicy"]
)
price = cf.FormalContext.from_array(
    [[1, 0], [0, 1], [1, 0], [1, 1], [0, 1], [1, 0]], objects, ["cheap", "popular"]
)

html = plot_nested(diet, inner=price)
html.write_html("nested.html")
```

Every outer concept shows the same inner lattice, so only one inner lattice
is computed. Use `ExplorationBuilder` (below) for deeper nesting.

## From many-valued data

`ExplorationBuilder` scales one many-valued table into the outer and inner
contexts, and can nest further levels:

```python
import pandas as pd
from conceptflow import ExplorationBuilder, ManyValuedContext
from conceptflow.preprocessing import DichotomicScale, GeneralScale

patients = pd.DataFrame(
    {
        "coma": ["no", "no", "yes", "yes", "no", "yes"],
        "pH": ["normal", "dangerous", "pathological", "dangerous", "pathological", "normal"],
    },
    index=[f"patient {i}" for i in range(6)],
)

builder = ExplorationBuilder(ManyValuedContext.from_dataframe(patients))
outer = builder.root("coma", [DichotomicScale("coma", true_value="yes", false_value="no")])
builder.expand_all(
    outer,
    scales=[
        GeneralScale(
            "pH",
            mapping={
                "normal": [],
                "pathological": ["pathological pH"],
                "dangerous": ["pathological pH", "dangerous pH"],
            },
        )
    ],
)

html = plot_nested(outer, title="Coma and pH")
```

`expand_all` takes filters (`min_extent_size=`, `include_bottom=`,
`predicate=`); `expand(parent, concept, name, scales)` expands one concept.
Many-valued data with missing values needs
`ExplorationBuilder(..., missing="false")`; see
[Conceptual scaling](scaling.md#missing-values).

## Static figures

`backend="matplotlib"` and `backend="tikz"` draw the first nesting level as a
publication figure:

```python
fig = plot_nested(outer, backend="matplotlib", show_object_labels=True)
fig.savefig("nested.pdf")

plot_nested(outer, backend="tikz").write("nested.tex")
```

## Realized combinations as data

The filled and hollow inner nodes are available as a table, one row per
(outer concept, inner concept) pair:

```python
from conceptflow.exploration import nested_combinations

table = nested_combinations(outer)
never = table[~table.realized]
print(set(table.columns) >= {"outer_intent", "inner_intent", "realized", "objects"})  # True
```

Unrealized pairs are attribute combinations that never occur together — often
the most interesting result of a nested diagram.
