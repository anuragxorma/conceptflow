# ConceptFlow

ConceptFlow is a Python library for **Formal Concept Analysis (FCA)** that
fits into the scientific Python stack: contexts are NumPy/pandas-friendly,
and the machine-learning parts are proper
[scikit-learn](https://scikit-learn.org) estimators that pass scikit-learn's
estimator checks.

```bash
pip install conceptflow
```

ConceptFlow requires Python 3.12 or newer.

## What it covers

| Area | Highlights |
|---|---|
| [Formal contexts](guide/contexts.md) | building, editing, clarification and reduction, apposition and other context operations, `.cxt` / JSON / CSV, incomplete (yes/no/unknown) contexts |
| [Concept lattices](guide/lattices.md) | NextClosure and Close-by-One on bitsets, iceberg lattices, navigation (join, meet, up-sets, object and attribute concepts) |
| [Implications and rules](guide/rules.md) | canonical (Duquenne-Guigues) basis, proper-premise basis, near-implications (Luxenburger basis), minimal generators |
| [Concept measures](guide/measures.md) | stability (exact and bounds), Δ-measure, entropy, information gain, lattice indices |
| [Conceptual scaling](guide/scaling.md) | seven scale types and explicit missing-value handling for many-valued data |
| [scikit-learn](guide/sklearn.md) | scaling, concept and implication features, and an implication-based outlier detector |
| [Visualization](guide/visualization.md) | interactive HTML, DimFlux and layered layouts, matplotlib, Graphviz and TikZ export |
| [Nested line diagrams](guide/nested.md) | Toscana-style nested diagrams, realized/unrealized combinations as data |
| [Data quality](guide/quality.md) | rule violations, redundant objects and attributes, explanations |

## A first look

```python
import conceptflow as cf
from conceptflow.algorithms import compute_canonical_basis

# Each object has either all three attributes or exactly one.
ctx = cf.FormalContext.from_array(
    [[1, 1, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
    objects=["g1", "g2", "g3", "g4"],
    attributes=["a", "b", "c"],
)

lattice = cf.ConceptLattice.from_context(ctx)
print(lattice.n_concepts)  # 5

for implication in compute_canonical_basis(ctx):
    print(implication.premise_names(ctx), "->", implication.conclusion_names(ctx))
# ('a', 'b') -> ('c',)
# ('a', 'c') -> ('b',)
# ('b', 'c') -> ('a',)
```

Continue with [Getting started](getting-started.md).

## Citing and license

ConceptFlow is released under the BSD 3-Clause license. Lattice layouts use
[DimFlux](https://arxiv.org/abs/2603.16366) (the `dim-flux` package, CC BY
4.0), installed automatically.
