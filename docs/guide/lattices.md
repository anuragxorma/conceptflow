# Concept lattices

## Building a lattice

```python
import numpy as np
import conceptflow as cf

ctx = cf.FormalContext.from_array(
    [[1, 1, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]],
    objects=["g1", "g2", "g3", "g4", "g5"],
    attributes=["a", "b", "c"],
)

lattice = cf.ConceptLattice.from_context(ctx)               # NextClosure
lattice_cbo = cf.ConceptLattice.from_context(ctx, algorithm="closebyone")

print(lattice.n_concepts)  # 6
print(len(lattice.edges))  # 7
```

Algorithms (all on integer bitsets):

| `algorithm` | Order of `lattice.concepts` | Notes |
|---|---|---|
| `"nextclosure"` (default) | lectic order of intents | Ganter's NextClosure |
| `"closebyone"` | by extent size | usually the fastest |
| `"bruteforce"` | by extent size | tries every attribute subset; reference implementation for small contexts |

A `Concept` stores `extent` and `intent` as frozensets of indices into
`ctx.objects` and `ctx.attributes`. `lattice.edges` are the Hasse diagram's
cover pairs `(lower, upper)`; they are computed once and cached.

## Navigating

```python
top, bottom = lattice.top(), lattice.bottom()

ab = lattice.concept_from_attributes(["a", "b"])        # (B', B'')
print([ctx.objects[g] for g in sorted(ab.extent)])      # ['g1', 'g5']

g5 = lattice.object_concept("g5")                       # γ(g): smallest concept containing g
a = lattice.attribute_concept("a")                      # μ(m): largest concept with m in its intent

print(lattice.join(g5, lattice.object_concept("g4")) == top)  # True
print(lattice.meet(a, lattice.attribute_concept("c")) == lattice.object_concept("g1"))  # True

print(len(lattice.parents(g5)), len(lattice.children(top)))  # 2 3
print(len(lattice.upset(g5)), len(lattice.downset(top)))      # 4 6
print(len(lattice.atoms()), len(lattice.coatoms()))           # 2 3
```

Every lookup accepts names or indices. `to_networkx()` returns the Hasse
diagram as a NetworkX graph.

## Iceberg lattices

Often only frequent concepts matter. `min_support` computes just the
concepts with at least that many objects (an int) or that fraction of the
objects (a float):

```python
data = np.random.default_rng(0).random((200, 15)) < 0.3
big = cf.FormalContext.from_array(data)

iceberg = cf.ConceptLattice.from_context(big, min_support=0.1)
print(iceberg.is_iceberg)  # True
print(all(len(c.extent) >= 20 for c in iceberg.concepts))  # True
```

The frequent concepts form an up-set of the lattice, so joins stay inside.
The top concept is always included, even when not even it is frequent. A
lattice counts as an iceberg (`is_iceberg`) only if some concept was cut off.
Navigation that would leave the iceberg — the bottom concept, atoms, a meet
below the threshold — raises a `ValueError` that says so.

## Large lattices

Lattices can grow exponentially. Nothing is limited by default, but once an
enumeration passes 50,000 concepts it warns, once, while it continues:

```text
LargeLatticeWarning: The concept lattice has more than 50,000 concepts and is
still growing. Computation continues. To limit it, use min_support=... or
max_concepts=...
```

You then decide whether to do anything:

- `min_support=...` computes only the frequent concepts (above);
- `max_concepts=...` stops with `ConceptLimitExceeded` once the limit is
  passed, instead of running on.

```python
from conceptflow.algorithms.enumeration import ConceptLimitExceeded

try:
    cf.ConceptLattice.from_context(big, max_concepts=1000)
except ConceptLimitExceeded as error:
    print("too large:", error)
```

The same options exist on `ConceptLatticeEstimator`.

## Lower-level functions

`conceptflow.algorithms` has the building blocks: `enumerate_concepts`,
`compute_hasse_edges`, the derivation operators (`object_derivation`,
`attribute_derivation`, `attribute_closure`), `subconcept_of` and
`is_cover`.
