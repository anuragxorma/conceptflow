# Formal contexts

A `FormalContext` is an immutable table of objects × attributes with a
boolean incidence. Methods that "change" a context return a new one.

## Creating contexts

```python
import numpy as np
import pandas as pd
import conceptflow as cf

ctx = cf.FormalContext.from_array(
    [[1, 1, 0], [1, 0, 1], [0, 1, 1]],
    objects=["g1", "g2", "g3"],
    attributes=["a", "b", "c"],
)

frame = pd.DataFrame(
    {"a": [True, False], "b": [True, True]}, index=["x", "y"]
)
ctx2 = cf.FormalContext.from_dataframe(frame)

print(ctx.n_objects, ctx.n_attributes)  # 3 3
```

How cell values are read — one rule per cell, whatever the type of the
table:

- **booleans** as they are;
- **numbers**: values greater than 0 are crosses (so 0/1 data, counts and
  ±1-encoded data all work as expected);
- **empty cells** (None, NaN, `pd.NA`) mean "no cross", as in a printed cross
  table;
- **strings** must be a known spelling: `"1"`, `"true"`, `"x"`, `"×"` for a
  cross, `"0"`, `"false"`, `"."` or `""` for none. Anything else (such as
  `"yes"`/`"no"`) raises instead of being guessed;
- the string `"?"` means "unknown" and is rejected: use an
  [`IncompleteContext`](#incomplete-contexts) for unknown entries.

Derivation operators accept attribute and object names or indices and return
index sets. Names are always strings, so an integer is always a position
(and must be in range):

```python
print(sorted(ctx.attribute_derivation(["a"])))   # [0, 1]
print(sorted(ctx.object_derivation(["g1", "g2"])))  # [0]
print(sorted(ctx.closure(["b"])))                 # [1]
```

## Editing

```python
smaller = ctx.subcontext(objects=["g1", "g3"], attributes=["a", "c"])
dual = ctx.transpose()                      # objects <-> attributes
renamed = ctx.rename(attributes={"a": "alpha"}, objects=str.upper)
grown = ctx.add_objects(["g4"], [[1, 1, 1]]).add_attributes(["d"], [[1], [0], [0], [1]])
pruned = grown.remove_attributes(["b"]).remove_objects(["g2"])

print(grown)  # FormalContext(n_objects=4, n_attributes=4)
print(renamed.objects)  # ('G1', 'G2', 'G3')
```

## Clarification and reduction

Duplicate rows and columns do not change the concept lattice; neither do
*reducible* objects and attributes (whose concept is a join or meet of
others). Removing them gives the smallest context with the same lattice:

```python
from conceptflow.algorithms import clarify_context, reduce_context

messy = cf.FormalContext.from_array(
    [[1, 1, 0], [1, 1, 0], [1, 0, 1], [1, 0, 0]],
    objects=["p", "p_copy", "q", "r"],
    attributes=["a", "b", "c"],
)

result = reduce_context(messy)
print(result.context.objects)      # ('p', 'q')
print(result.removed_objects)      # ('r',)
print(result.removed_attributes)   # ('a',)
print(result.object_groups["p"])   # ('p', 'p_copy')
```

`clarify_context` only merges duplicates.
[Data quality](quality.md) has a report that also *explains* redundancy.

## Context operations

Following Ganter & Wille:

```python
from conceptflow.core import (
    apposition, subposition, direct_sum, direct_product, complement, dual,
)

colour = cf.FormalContext.from_array(
    [[1, 0], [0, 1], [1, 0]], objects=["x", "y", "z"], attributes=["red", "blue"]
)
size = cf.FormalContext.from_array(
    [[1], [0], [1]], objects=["z", "y", "x"], attributes=["large"]
)

side_by_side = apposition(colour, size)        # same objects, matched by name
print(side_by_side.attributes)  # ('red', 'blue', 'large')

stacked = subposition(colour, colour, prefixes=("A:", "B:"))
print(stacked.objects[:2])  # ('A:x', 'A:y')

# Disjoint union whose lattice is the product of the two lattices.
summed = direct_sum(colour, size, prefixes=("C:", "S:"))
print(summed.objects)  # ('C:x', 'C:y', 'C:z', 'S:z', 'S:y', 'S:x')
print(complement(colour).incidence.tolist())  # [[False, True], [True, False], [False, True]]
```

When both contexts use the same names, `prefixes=` keeps them apart;
otherwise an error names the clash. `direct_product(K1, K2)` has object pairs
`"g1|g2"` and attribute pairs `"m1|m2"`.

## Files

```python
from conceptflow.io import read_cxt, write_cxt, read_json, write_json, read_csv, write_csv

write_cxt(ctx, "context.cxt")
write_json(ctx, "context.json")      # FCART / fcapy JSON layout
write_csv(ctx, "context.csv")

assert read_cxt("context.cxt") == ctx
assert read_json("context.json") == ctx
assert read_csv("context.csv") == ctx

triangles = read_cxt("data/triangles.cxt")
```

`read_csv` accepts `true_values=` / `false_values=` (defaults include
`1/0`, `True/False`, `x/X`, `.` and empty cells) and raises on anything else,
naming the line and attribute. JSON files written by fcapy can be read
directly.

## Many-valued contexts

Raw data with arbitrary values lives in a `ManyValuedContext`; conceptual
scaling turns it into a formal context (see
[Conceptual scaling](scaling.md)):

```python
mvc = cf.ManyValuedContext.from_dataframe(
    pd.DataFrame({"tempo": [92, 128], "mode": ["minor", "major"]}, index=["s1", "s2"])
)
print(mvc.n_objects, mvc.n_attributes)  # 2 2
```

## Incomplete contexts

An `IncompleteContext` has three kinds of entries: known yes, known no and
unknown. Questions are answered over all *completions* (ways of filling in
the unknowns):

```python
from conceptflow import IncompleteContext

partial = IncompleteContext.from_dataframe(
    pd.DataFrame(
        {"linear": [True, None, False], "polar": [True, True, None]},
        index=["CO", "C60", "H2O"],
    )
)

print(partial.n_unknown)  # 2
print(partial.implication_status(["linear"], ["polar"]))  # valid
# C60 might be linear and H2O might not be polar, so this could still hold:
print(partial.implication_status(["polar"], ["linear"]))  # possible

certain = partial.certain_context()      # unknown -> no
possible = partial.possible_context()    # unknown -> yes
```

`implication_status` returns `"valid"` (holds in every completion),
`"possible"` (in some) or `"refuted"` (in none);
`implication_counterexamples` gives the objects behind the answer. Known
values must be booleans or 0/1; text such as `"no"` raises instead of being
guessed.
