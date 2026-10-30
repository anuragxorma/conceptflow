# Scale-measures and order indices

A **scale-measure** of a formal context is a coarser view of it: the same
objects, fewer distinctions, so a smaller concept lattice whose extents are
all extents of the original (Hanika & Hirth, *On the lattice of conceptual
measurements*, 2022). Families of them give related views of one data set,
from a rough overview to the full context, and the concepts of a coarser
view reappear in every finer one.

These scales are not the conceptual scales of
[Conceptual scaling](scaling.md), which turn many-valued data into a formal
context.

```python
import conceptflow as cf
from conceptflow.core import is_scale_measure, refines, scale_measure, scale_measure_problems
from conceptflow.datasets import load_context

living = load_context("livingbeings_en")
```

## Building scale-measures

`scale_measure(context, attributes)` keeps the objects and uses the given
attributes. An entry is an original attribute (an *attribute projection*) or
a tuple of attributes, which becomes their *conjunction*: an object has it
when it has every part. Both always give a scale-measure (Hanika & Hirth,
Cor. 22 and Cor. 17).

```python
habitat = scale_measure(living, ["lives in water", "lives on land"])
mixed = scale_measure(
    living, ["lives in water", ("needs chlorophyll to produce food", "lives on land")]
)

print(mixed.attributes)  # ('lives in water', 'needs chlorophyll to produce food ∧ lives on land')
print(cf.ConceptLattice.from_context(habitat).n_concepts)  # 4
print(is_scale_measure(living, mixed))  # True
```

The attributes are put in a fixed order, by the positions of their parts in
the original context, so two scales of one context list their common
attributes in the same relative order. Layouts that depend on the order of
the columns then draw them comparably.

Disjunctions and negations are not offered: they give a scale-measure only
when the resulting extent happens to be an extent of the original (Prop. 16).
`is_scale_measure` checks any context on the same objects, one closure per
attribute (Prop. 20), and `scale_measure_problems` says which attributes
fail:

```python
water_or_plant = cf.FormalContext(
    objects=living.objects,
    attributes=("lives in water or needs chlorophyll",),
    incidence=(living.incidence[:, 1] | living.incidence[:, 3])[:, None],
)
print(is_scale_measure(living, water_or_plant))  # False
print(len(scale_measure_problems(living, water_or_plant)))  # 1
```

## Refinement pairs

`refines(coarse, fine)` tells whether every extent of `coarse` is an extent
of `fine`. Such a *refinement pair* has every concept of the coarser view
among the concepts of the finer one, matched by extent, so the two lattices
(and drawings of them) can be compared concept by concept. Every
scale-measure refines into its context; two scales need not refine into each
other.

```python
taxa = scale_measure(living, [
    "needs water to live", "needs chlorophyll to produce food", "two seed leaves",
    "one seed leave", "can move around", "has limbs", "suckles its offspring",
])
print(refines(taxa, living))  # True
print(refines(habitat, taxa))  # False
```

Contexts or full concept lattices can be given; a lattice saves computing it
again.

## Order indices

Three classic indices describe the shape of a lattice, next to
[`linearity_index`](measures.md):

- `lattice_width`: the largest antichain, the most concepts that are pairwise
  incomparable (Dilworth's theorem, with SciPy's matching);
- `lattice_height`: the number of concepts in a longest chain;
- `incomparable_pairs`: the pairs of concepts neither of which lies below
  the other. This is what a two-dimensional drawing has to decide, which of
  the two goes left, and it predicts the cost of layouts based on order
  dimension better than the number of concepts does.

`concept_levels` gives every concept's level: the number of concepts in a
longest chain from a minimal concept up to it. Concepts on one level are
incomparable, so a drawing may arrange each level freely.

```python
from conceptflow.metrics import (
    concept_levels,
    incomparable_pairs,
    lattice_height,
    lattice_width,
)

lattice = cf.ConceptLattice.from_context(living)
print(lattice.n_concepts, lattice_width(lattice), lattice_height(lattice))  # 19 6 6
print(incomparable_pairs(lattice))  # 91
print(sorted(set(concept_levels(lattice).tolist())))  # [1, 2, 3, 4, 5, 6]
```

All four also work on iceberg lattices, describing the drawn concepts.

## Comparing drawings of a refinement pair

`layout_positions(graph_data)` gives where each concept is drawn, as
`{extent: (x, y)}` with the extent as a set of object names. The coordinates
are those of the figure a reader sees: upright, y growing upwards, with
DimFlux layouts rotated as they are when drawn. Drawings of the two lattices
of a refinement pair can then be matched by extent:

```python
from conceptflow.visualization import layout_positions

fine = layout_positions(cf.plot_lattice(lattice, backend="graph_data", layout="layered"))
coarse = layout_positions(
    cf.plot_lattice(cf.ConceptLattice.from_context(taxa), backend="graph_data", layout="layered")
)
shared = [extent for extent in coarse if extent in fine]
print(len(shared), len(coarse))  # 8 8
```

With the shared concepts' coordinates, any comparison can follow, for
example whether the left-to-right order is kept, with Kendall's tau from
SciPy:

```python
from scipy.stats import kendalltau

tau = kendalltau([coarse[e][0] for e in shared], [fine[e][0] for e in shared]).statistic
print(round(tau, 2))  # 0.52
```
