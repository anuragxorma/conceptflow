# Factorization

Factorization explains a binary table by a few *factors*. ConceptFlow has two
kinds: Boolean factorization (GreConD, ASSO) and ordinal two-factorization.

## Boolean factorization

A Boolean factorization writes a binary table `X` (objects × attributes) as a
Boolean product `X ≈ S ∘ B`:

- `B` (factors × attributes) says which attributes make up each factor;
- `S` (objects × factors) says which objects have each factor;
- an object has an attribute if it has some factor that contains it.

Both estimators work like scikit-learn's `NMF`: `fit` learns the factors as
`components_` (`B`), `transform` gives the factor memberships (`S`) of any
data, and `inverse_transform` rebuilds the table from them.

```python
import pandas as pd
from conceptflow.decomposition import ASSO, GreConD

baskets = pd.DataFrame(
    [[1, 1, 1, 0, 0, 0], [1, 1, 1, 0, 0, 0], [1, 1, 0, 0, 0, 0],
     [0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 1, 1], [0, 0, 0, 0, 1, 1]],
    columns=["bread", "butter", "jam", "chips", "salsa", "soda"],
    index=["c1", "c2", "c3", "c4", "c5", "c6"],
)

def show(model):
    return [[name for name, used in zip(baskets.columns, row) if used] for row in model.components_]

grecond = GreConD().fit(baskets)
print(grecond.n_components_)  # 4
print(show(grecond))  # [['bread', 'butter'], ['chips', 'salsa', 'soda'], ['bread', 'butter', 'jam'], ['salsa', 'soda']]
print(grecond.reconstruction_error_)  # 0
```

### GreConD

`GreConD` (Belohlavek & Vychodil, 2010) uses **formal concepts as factors**.
Each new factor is grown attribute by attribute, always taking the attribute
whose concept covers the most still-uncovered 1s. Because factors are
concepts, they never cover a 0, and they are easy to read: a factor is a set
of objects and exactly the attributes they share.

- `min_coverage` (default 1.0) stops once that share of the 1s is covered,
  for a shorter explanation of most of the data;
- `n_components` caps the number of factors;
- `factors_` lists the factors as `Concept`s of the training data;
- `transform` gives an object a factor iff it has all of the factor's
  attributes, so on the training data the memberships are the extents.

```python
print(GreConD(min_coverage=0.8).fit(baskets).n_components_)  # 3
print(grecond.transform(baskets)[:, 0].tolist())  # [True, True, True, False, False, False]
```

### ASSO

`ASSO` (Miettinen et al., 2008) builds candidate factors from association
rules: candidate `i` contains attribute `j` if at least a share `threshold`
of the objects with `i` also have `j`. It then picks factors greedily,
weighing covered 1s (`positive_weight`) against covered 0s
(`negative_weight`). Factors may cover some 0s, which often gives fewer,
broader factors:

```python
asso = ASSO(threshold=0.6).fit(baskets)
print(show(asso))  # [['bread', 'butter', 'jam'], ['chips', 'salsa', 'soda']]
print(asso.coverage_, asso.reconstruction_error_)  # 1.0 2
```

Two factors cover every 1, at the price of two 0s (`c3` without jam, `c6`
without chips). A larger `negative_weight` makes covering 0s costlier.

`coverage_` is the share of the training 1s covered; `reconstruction_error_`
counts the training cells that `inverse_transform(transform(X))` gets wrong.

Both estimators accept the usual inputs (arrays, DataFrames, sparse matrices,
`FormalContext`), support `output="dataframe"`/`"sparse"` and `set_output`,
and work in pipelines. The factor memberships make compact, readable features
for a downstream model.

## Ordinal factorization

`ExactOrdinalTwoFactorizer` implements the exact construction of Algorithm 1
in Dürrschnabel & Stumme's work on ordinal factorization: for a context whose
complement's concept lattice has order dimension at most 2, it finds two
linear extensions of that lattice and builds the two Ferrers factors from
them.

```python
import conceptflow as cf
from conceptflow.decomposition import ExactOrdinalTwoFactorizer

ctx = cf.FormalContext.from_array([[1, 1, 0], [1, 0, 1], [0, 1, 1]])
model = ExactOrdinalTwoFactorizer().fit(ctx)

first_factor, second_factor = model.factors_
print(model.coverage_)  # 1.0
```

`coverage_` is the share of crosses covered by the two factors. The two
linear extensions come from a transitive orientation of the lattice's
incomparability graph, which takes polynomial time.

### Ord2Factor: any context

Most contexts are not two-factorizable. `Ord2Factor` (Algorithm 2 of
Dürrschnabel & Stumme, "Maximal Ordinal Two-Factorizations", 2023) drops as
few crosses as it can to make the context two-factorizable, then factorizes
what remains. A context is two-factorizable exactly when its
*incompatibility graph* (crosses that cannot be in the same Ferrers relation
are adjacent) is bipartite, so it keeps the crosses of a large induced
bipartite subgraph and repeats, since dropping crosses can create new
incompatibilities.

Finding the largest bipartite subgraph is NP-hard; `method` chooses how,
following Dürrschnabel, Hanika & Stumme (2022):

- `"exact"` (default): the fewest crosses to drop, from a SAT formulation
  solved with MiniSat 2.2 — exponential in the worst case, meant for the
  small contexts ordinal factorization is usually applied to;
- `"greedy"`: a fast heuristic (every dropped cross conflicts with both
  sides of the bipartition it built);
- `"annealing"`: the greedy result improved by simulated annealing
  (`n_iterations`, `temperature`).

```python
from conceptflow.datasets import load_context
from conceptflow.decomposition import Ord2Factor

triangles = load_context("triangles_en")
exact = Ord2Factor().fit(triangles)
greedy = Ord2Factor(method="greedy", random_state=0).fit(triangles)

print(round(exact.coverage_, 2), len(exact.removed_incidence_))  # 0.87 3
print(round(greedy.coverage_, 2), len(greedy.removed_incidence_))  # 0.57 10
```

`factors_` are the two Ferrers relations, `removed_incidence_` the crosses
that are not covered, `coverage_` the share that is, and `n_rounds_` how
often crosses were dropped. The paper's third heuristic, a genetic
algorithm, is not included: its selection step is not fully specified.
