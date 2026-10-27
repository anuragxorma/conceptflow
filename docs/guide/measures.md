# Concept measures

Lattices of real data are large; interestingness measures help pick the
concepts worth looking at. Concept measures return one value per concept as a
NumPy array in the order of `lattice.concepts`.

```python
import numpy as np
import conceptflow as cf
from conceptflow.metrics import (
    concept_stability, stability_bounds, delta_measure,
    concept_entropy, concept_information_gain,
    linearity_index, distributivity_index,
)

data = np.random.default_rng(1).random((40, 8)) < 0.4
lattice = cf.ConceptLattice.from_context(cf.FormalContext.from_array(data))
```

## Stability

The (intensional) stability of a concept *(A, B)* is the share of subsets of
*A* whose common attributes are still exactly *B*: how well the concept
survives when objects are removed. Stable concepts are less likely to be
noise.

```python
stability = concept_stability(lattice)          # exact
lower, upper = stability_bounds(lattice)        # cheap bounds
assert np.all(lower <= stability + 1e-12) and np.all(stability <= upper + 1e-12)

most_stable = np.argsort(-stability)[:5]
for i in most_stable:
    concept = lattice.concepts[i]
    print(round(stability[i], 3), len(concept.extent), sorted(concept.intent))
```

Exact stability needs the full lattice, and its cost grows quadratically
with the number of concepts (it warns above 20,000). The bounds (Buzmakov,
Kuznetsov & Napoli, 2014) and the Δ-measure only need each concept's lower
neighbours: they are fast and also work on
[iceberg lattices](lattices.md#iceberg-lattices).

## Δ-measure

The Δ-measure is the number of objects a concept loses to its largest
subconcept — a fast stability proxy:

```python
delta = delta_measure(lattice)
print(delta.dtype, delta.min() >= 0)  # int64 True
```

## Label-aware measures

With class labels `y` (one per object), concepts can be scored by how pure
they are:

```python
y = np.random.default_rng(2).integers(0, 2, 40)

entropy = concept_entropy(lattice, y)            # 0 = all objects share one label
gain = concept_information_gain(lattice, y)      # information about y from membership
```

## Lattice indices

```python
# Share of comparable pairs of concepts:
print(round(linearity_index(lattice), 3))  # 0.226
# Share of pairs of intents whose union is again an intent:
print(round(distributivity_index(lattice), 3))
```

A chain has both indices equal to 1. The distributivity index depends on the
context, not only on the lattice's shape (an attribute no object has lowers
it), and it is quadratic for (near-)distributive lattices, where almost every
pair counts; it warns above 20,000 concepts.

## Selecting concepts as features

`ConceptLatticeEstimator` uses these measures to keep only the best concepts
as features; see [scikit-learn](sklearn.md):

```python
from conceptflow.cluster import ConceptLatticeEstimator

est = ConceptLatticeEstimator(select="stability", n_concepts=10).fit(data)
print(est.transform(data).shape)  # (40, 10)
```
