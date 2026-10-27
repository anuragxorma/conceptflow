# Getting started

## Installation

```bash
pip install conceptflow
```

For development, from a clone of the repository:

```bash
pip install -e ".[dev]"        # tests and linting
pip install -e ".[docs]"       # this documentation (mkdocs serve)
```

## A formal context

A formal context is a table of objects and the attributes they have. The
running example of this documentation is the classic "living beings and
water" context from Ganter & Wille's *Formal Concept Analysis*:

```python
import pandas as pd
import conceptflow as cf

table = pd.DataFrame(
    [
        # water  in_water on_land chloro dicot monocot moves limbs suckles
        [1, 1, 0, 0, 0, 0, 1, 0, 0],  # leech
        [1, 1, 0, 0, 0, 0, 1, 1, 0],  # bream
        [1, 1, 1, 0, 0, 0, 1, 1, 0],  # frog
        [1, 0, 1, 0, 0, 0, 1, 1, 1],  # dog
        [1, 1, 0, 1, 0, 1, 0, 0, 0],  # spike-weed
        [1, 1, 1, 1, 0, 1, 0, 0, 0],  # reed
        [1, 0, 1, 1, 1, 0, 0, 0, 0],  # bean
        [1, 0, 1, 1, 0, 1, 0, 0, 0],  # maize
    ],
    index=["leech", "bream", "frog", "dog", "spike-weed", "reed", "bean", "maize"],
    columns=[
        "needs water", "lives in water", "lives on land", "chlorophyll",
        "dicotyledon", "monocotyledon", "can move", "has limbs", "suckles",
    ],
)

ctx = cf.FormalContext.from_dataframe(table)
print(ctx)  # FormalContext(n_objects=8, n_attributes=9)
```

Contexts can also be read from Burmeister `.cxt`, JSON and CSV files; see
[Formal contexts](guide/contexts.md).

## Concepts and the concept lattice

A *formal concept* is a pair (extent, intent): a set of objects and exactly
the attributes they all share, such that no further object has all of them.

```python
lattice = cf.ConceptLattice.from_context(ctx)
print(lattice.n_concepts)  # 19

frog = lattice.object_concept("frog")
print([ctx.attributes[m] for m in sorted(frog.intent)])
print([ctx.objects[g] for g in sorted(frog.extent)])
```

Concepts store extents and intents as sets of indices into
`ctx.objects` / `ctx.attributes`.

## Implications

An implication *A → B* holds if every object with all attributes in *A* also
has all attributes in *B*. The canonical basis is the smallest set of
implications from which all others follow:

```python
from conceptflow.algorithms import compute_canonical_basis

for imp in compute_canonical_basis(ctx):
    print(set(imp.premise_names(ctx)), "->", set(imp.conclusion_names(ctx)))
```

## Drawing the lattice

```python
fig = cf.plot_lattice(lattice, backend="matplotlib", layout="layered")
fig.savefig("lattice.pdf")
```

Other backends give an interactive HTML page (`"d3"`), Graphviz output and
TikZ code for LaTeX; see [Visualization](guide/visualization.md).

## In a scikit-learn pipeline

```python
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from conceptflow.cluster import ConceptLatticeEstimator

X = ctx.incidence                       # objects x attributes, boolean
y = [0, 0, 0, 0, 1, 1, 1, 1]            # animals vs plants

model = make_pipeline(
    ConceptLatticeEstimator(select="stability", n_concepts=5),
    LogisticRegression(),
).fit(X, y)
print(model.predict(X))
```

The concept features are interpretable: each column is one concept, true for
the objects having all attributes of its intent. See
[scikit-learn](guide/sklearn.md).
