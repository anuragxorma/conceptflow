# ConceptFlow

![Python](https://img.shields.io/badge/python-3.12%2B-blue)
![License](https://img.shields.io/badge/license-BSD--3--Clause-green)
![Status](https://img.shields.io/badge/status-experimental-orange)

ConceptFlow is a Python library for **Formal Concept Analysis (FCA)** that
fits into the scientific Python stack. Its machine-learning parts are proper
[scikit-learn](https://scikit-learn.org) estimators that pass scikit-learn's
estimator checks.

```bash
pip install conceptflow
```

Requires Python 3.12 or newer.

## Example

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

cf.plot_lattice(lattice, backend="tikz", layout="layered").write("lattice.pdf")
```

## Features

- **Formal contexts:** building and editing, clarification and reduction,
  apposition, subposition, direct sum and product; `.cxt`, JSON (FCART/fcapy)
  and CSV files; incomplete (yes/no/unknown) contexts.
- **Concept lattices:** NextClosure and Close-by-One on bitsets, iceberg
  lattices, navigation (join, meet, up-sets, object and attribute concepts),
  explicit opt-in limits for very large lattices.
- **Implications and rules:** canonical (Duquenne-Guigues) basis,
  proper-premise basis, near-implications with exceptions (Luxenburger basis),
  minimal generators.
- **Measures:** stability (exact and bounds), Δ-measure, entropy, information
  gain, linearity and distributivity indices.
- **Conceptual scaling** of many-valued data with seven scale types and
  explicit missing-value handling.
- **scikit-learn:** scaling, concept and implication features, and an
  outlier detector that explains its decisions with implications.
- **Visualization:** interactive HTML, DimFlux and layered layouts,
  matplotlib, Graphviz and TikZ/LaTeX export; Toscana-style nested line
  diagrams.
- **Data quality:** rule violations, redundant objects and attributes,
  explanations of objects and rules.

## Documentation

The documentation lives in [`docs/`](https://github.com/anuragxorma/conceptflow/blob/main/docs/index.md):

- [Getting started](https://github.com/anuragxorma/conceptflow/blob/main/docs/getting-started.md)
- User guide: [contexts](https://github.com/anuragxorma/conceptflow/blob/main/docs/guide/contexts.md),
  [lattices](https://github.com/anuragxorma/conceptflow/blob/main/docs/guide/lattices.md),
  [implications](https://github.com/anuragxorma/conceptflow/blob/main/docs/guide/rules.md),
  [measures](https://github.com/anuragxorma/conceptflow/blob/main/docs/guide/measures.md),
  [scaling](https://github.com/anuragxorma/conceptflow/blob/main/docs/guide/scaling.md),
  [scikit-learn](https://github.com/anuragxorma/conceptflow/blob/main/docs/guide/sklearn.md),
  [visualization](https://github.com/anuragxorma/conceptflow/blob/main/docs/guide/visualization.md),
  [nested diagrams](https://github.com/anuragxorma/conceptflow/blob/main/docs/guide/nested.md),
  [data quality](https://github.com/anuragxorma/conceptflow/blob/main/docs/guide/quality.md)
- [Changelog](https://github.com/anuragxorma/conceptflow/blob/main/CHANGELOG.md)

To build the documentation site locally (with the API reference):

```bash
pip install -e ".[docs]"
mkdocs serve
```

Runnable examples, including a full case study, are in
[`examples/`](https://github.com/anuragxorma/conceptflow/tree/main/examples).

## License

BSD 3-Clause; see [LICENSE](https://github.com/anuragxorma/conceptflow/blob/main/LICENSE).
Lattice layouts use [DimFlux](https://arxiv.org/abs/2603.16366) (the
`dim-flux` package, CC BY 4.0), installed automatically.
