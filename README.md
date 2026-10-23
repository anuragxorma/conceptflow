# ConceptFlow

![Python](https://img.shields.io/badge/python-3.12%2B-blue)
![License](https://img.shields.io/badge/license-BSD--3--Clause-green)
![Status](https://img.shields.io/badge/status-experimental-orange)

ConceptFlow is a scikit-learn compatible Formal Concept Analysis (FCA) framework.

The goal is to combine FCA-native mathematical structures with modern Python machine learning workflows.

## Installation

```bash
pip install conceptflow
```

To install from source instead (e.g. for development):

```bash
git clone https://github.com/anuragxorma/conceptflow
cd conceptflow
pip install -e ".[dev]"
```

ConceptFlow requires Python 3.12 or newer. Lattice and nested-diagram layouts
use [DimFlux](https://arxiv.org/abs/2603.16366) (the `dim-flux` package, installed
automatically; licensed CC BY 4.0). matplotlib, used for static figures, is
installed with it.

See [CHANGELOG.md](https://github.com/anuragxorma/conceptflow/blob/main/CHANGELOG.md) for what changed between releases.

Runnable versions of every code snippet below live in [`examples/`](https://github.com/anuragxorma/conceptflow/tree/main/examples/),
including a full case study in [`examples/eurovision_nested_diagram.py`](https://github.com/anuragxorma/conceptflow/blob/main/examples/eurovision_nested_diagram.py).

## Current features

**Core FCA structures**
- Binary formal contexts
- Many-valued contexts
- Incomplete (three-valued: yes / no / unknown) contexts
- Formal concepts
- Concept lattice construction

**Algorithms**
- Brute-force, NextClosure, and CloseByOne concept enumeration

**Preprocessing**
- Conceptual scaling with seven scale types
- Explicit missing-value handling (error / closed world / unknown)
- scikit-learn compatible preprocessing

**scikit-learn estimators**
- scikit-learn compatible lattice estimator
- scikit-learn compatible implication (stem basis) estimator

**Visualization**
- Nested line diagrams (subdirect decomposition)
- Interactive nested diagram visualization
- Static SVG/PDF/PNG export of lattices and nested diagrams (matplotlib)
- Realized / unrealized nested-diagram combinations as a DataFrame

**I/O**
- Burmeister `.cxt` read/write support

**Metrics**
- Basic support and confidence metrics, exception objects of a rule

**Rules**
- Duquenne-Guigues (stem) basis / implication computation (NextClosure)
- Luxenburger basis of near-implications, with exception objects

**Decomposition**
- Ordinal two-factorization

## Basic usage

```python
import numpy as np
import conceptflow as cf
from conceptflow.cluster import ConceptLatticeEstimator

ctx = cf.FormalContext.from_array(
    np.array([
        [1, 1, 0],
        [1, 0, 1],
        [0, 1, 1],
    ]),
    objects=["g1", "g2", "g3"],
    attributes=["m1", "m2", "m3"],
)

lattice = ConceptLatticeEstimator().fit(ctx).get_lattice()

print(lattice.n_concepts)
print(lattice.edges)
```

## Concept enumeration algorithms

ConceptFlow currently supports three concept enumeration methods:

```python
lattice = ConceptLatticeEstimator(algorithm="nextclosure").fit(ctx).get_lattice()
```

Supported algorithms:

```text
bruteforce
nextclosure
closebyone
```

`bruteforce` is mainly intended as a correctness baseline for small contexts.
`nextclosure` is the default FCA-native enumeration method.
`closebyone` provides a recursive canonicity-based enumeration strategy.

## Many-valued contexts

`ManyValuedContext` stores a raw many-valued data table before conceptual scaling.

```python
import pandas as pd
import conceptflow as cf

data = pd.DataFrame(
    {
        "tempo":   [92, 128, 74],
        "mode":    ["minor", "major", "minor"],
        "country": ["DE", "SE", "NO"],
    },
    index=["song_a", "song_b", "song_c"],
)

mvc = cf.ManyValuedContext.from_dataframe(data)

print(mvc.n_objects)      # 3
print(mvc.n_attributes)   # 3
```

Pass a `ManyValuedContext` to `ConceptualScaler` or `ExplorationBuilder` to
produce binary formal contexts.

## Conceptual scaling

`ConceptualScaler` applies a list of `Scale` objects to a `ManyValuedContext`
and produces a binary `FormalContext`.

```python
import pandas as pd
import conceptflow as cf

from conceptflow.preprocessing import (
    ConceptualScaler,
    NominalScale,
    OrdinalScale,
)

data = pd.DataFrame(
    {
        "color": ["red", "blue", "red"],
        "risk":  ["low", "medium", "high"],
    },
    index=["g1", "g2", "g3"],
)

scaler = ConceptualScaler(
    scales=[
        NominalScale("color"),
        OrdinalScale(
            "risk",
            levels=["low", "medium", "high"],
            mode="ge",
        ),
    ],
    output="context",
)

formal_context = scaler.fit_transform(data)
```

Supported outputs:

```text
"context"    -> FormalContext
"dataframe"  -> pandas DataFrame
"array"      -> NumPy ndarray
"sparse"     -> scipy.sparse CSR matrix
```

### Missing values

In a binary `FormalContext`, empty cells (None, NaN, `pd.NA`) mean "no
cross", as in a printed cross table. In many-valued data a missing value
(None, NaN, NaT, `pd.NA`, or the string `"?"`) is ambiguous: the attribute may
not apply to the object, or its value may be unknown. It is never silently
turned into crosses or non-crosses; `ConceptualScaler(missing=...)` makes the
choice explicit:

```text
"error"    -> raise ValueError naming the attribute and objects (default)
"false"    -> the object gets none of that scale's attributes (closed world)
"unknown"  -> attributes true for every value of the scale stay known-yes,
              those true for none stay known-no, the rest become unknown
```

With `missing="unknown"`, `output="context"` returns an `IncompleteContext`,
`"array"` and `"dataframe"` return floats (1.0 / 0.0, NaN for unknown), the
scikit-learn convention for missing values.
Unknown entries are independent of each other, so an implication that follows
from a scale itself (such as `size>=L -> size>=M`) may be reported as
"possible" rather than "valid".

### Incomplete contexts

`IncompleteContext` stores yes / no / unknown entries. Questions are answered
over all *completions* (ways to fill in the unknowns):

```python
from conceptflow import IncompleteContext

ctx = IncompleteContext.from_dataframe(df)   # True/False/1/0; None, NaN or "?" = unknown

ctx.certain_context()     # FormalContext with unknown -> no
ctx.possible_context()    # FormalContext with unknown -> yes
ctx.implication_status(["linear"], ["polar"])          # "valid" | "possible" | "refuted"
ctx.implication_counterexamples(["linear"], ["polar"])  # (certain, possible) object indices
```

Only booleans and 0/1 are accepted as known values; strings such as `"no"`
raise instead of being coerced to True.

### Scale types

**`NominalScale(source_attribute)`**
Unordered categorical values. Produces one binary attribute per distinct value
(`color=red`, `color=blue`, …). Each object is true for exactly one attribute.

**`ContranominalScale(source_attribute)`**
Complement of a nominal scale. Produces `color!=red`, `color!=blue`, …. Each
object is true for all attributes *except* its own value.

**`DichotomicScale(source_attribute, true_value=True, false_value=False)`**
Explicit two-valued scale for yes/no or present/absent attributes. Equivalent to
a two-value nominal scale but more explicit in intent.

```python
DichotomicScale("minor_key", true_value="minor", false_value="major")
```

**`OrdinalScale(source_attribute, levels, mode="ge")`**
Ordered categorical values. `levels` lists values from lowest to highest.
`mode` controls the comparison direction:

```text
"ge"    -> attr>=level  (each object is true for its rank and all below)
"le"    -> attr<=level  (each object is true for its rank and all above)
"exact" -> attr=level   (each object is true for exactly its level)
```

```python
OrdinalScale("risk", levels=["low", "medium", "high"], mode="ge")
```

**`ThresholdScale(source_attribute, thresholds)`**
Numeric values compared against explicit thresholds. Produces `attr>=t` for
each threshold `t`. Thresholds are automatically sorted.

```python
ThresholdScale("bpm", thresholds=[100, 150])
# bpm=128 -> {bpm>=100: True, bpm>=150: False}
```

**`InterordinalScale(source_attribute, levels)`**
Combines `<=` and `>=` for every level — both directions simultaneously. Useful
when both upper and lower bounds on an ordered attribute are conceptually relevant.

```python
InterordinalScale("size", levels=["S", "M", "L"])
# size=M -> {size<=S: False, size<=M: True, size<=L: True,
#            size>=S: True,  size>=M: True, size>=L: False}
```

**`GeneralScale(source_attribute, mapping, attribute_order=None)`**
Explicit value-to-attribute mapping for domain-specific or paper-style scales.
Each source value maps to a set of binary attribute names that are true for it.

```python
GeneralScale(
    "support",
    mapping={
        "strong": {"regional", "cultural", "historical"},
        "moderate": {"cultural", "historical"},
        "weak": {"historical"},
    },
)
```

## scikit-learn style pipeline

```python
import pandas as pd
from sklearn.pipeline import Pipeline

from conceptflow.cluster import ConceptLatticeEstimator
from conceptflow.preprocessing import (
    ConceptualScaler,
    NominalScale,
    OrdinalScale,
)

data = pd.DataFrame(
    {
        "color": ["red", "blue", "red"],
        "risk": ["low", "medium", "high"],
    },
    index=["g1", "g2", "g3"],
)

pipe = Pipeline([
    (
        "scaling",
        ConceptualScaler(
            scales=[
                NominalScale("color"),
                OrdinalScale(
                    "risk",
                    levels=["low", "medium", "high"],
                    mode="ge",
                ),
            ],
            output="context",
        ),
    ),
    (
        "lattice",
        ConceptLatticeEstimator(algorithm="nextclosure"),
    ),
])

pipe.fit(data)

lattice_step = pipe.named_steps["lattice"]

print(lattice_step.concepts_)
print(lattice_step.edges_)
```

## Nested line diagrams

A nested line diagram visualises the subdirect decomposition of a combined
context into two factor lattices. It is the standard Toscana-style tool for
exploring many-valued data split across two conceptual domains.

### Building the exploration tree

`ExplorationBuilder` takes a `ManyValuedContext` and builds an `ExplorationView`
tree by applying separate scale sets to the outer (root) and inner (child) domains.

```python
import pandas as pd
import conceptflow as cf

from conceptflow.preprocessing import ThresholdScale, DichotomicScale, GeneralScale
from conceptflow import ExplorationBuilder

mvc = cf.ManyValuedContext.from_dataframe(data)

builder = ExplorationBuilder(mvc)

# Outer lattice: voting-pattern attributes
outer_view = builder.root(
    name="voting",
    scales=[
        GeneralScale("political_support", mapping={...}),
        GeneralScale("regional_support",  mapping={...}),
        GeneralScale("cultural_support",  mapping={...}),
        GeneralScale("historical_support", mapping={...}),
    ],
)

# Inner lattice: one child for each outer concept
# expand_all attaches a child view to every concept in the outer lattice
builder.expand_all(
    parent=outer_view,
    scales=[
        ThresholdScale("bpm", thresholds=[100, 150]),
        DichotomicScale("mode", true_value="minor", false_value="major"),
    ],
    name_template="{label}",
)
```

`expand_all` accepts optional filters:

```python
builder.expand_all(
    parent=outer_view,
    scales=[...],
    min_extent_size=2,     # skip concepts with fewer than 2 objects
    include_bottom=False,  # skip the empty-extent concept
    predicate=lambda c: len(c.intent) > 0,  # custom filter
)
```

To expand only a single concept:

```python
top_concept = outer_view.lattice.top()
builder.expand(
    parent=outer_view,
    concept=top_concept,
    name="inner",
    scales=[...],
)
```

### Rendering the diagram

```python
from conceptflow.visualization import plot_nested

fig = plot_nested(
    outer_view,
    title="Eurovision Winners",
    width=1100,
    height=800,
    object_label=lambda name: name[:3],  # optional label callback
)

fig.show()               # display in Jupyter
fig.write_html("out.html")  # export to HTML
```

`plot_nested` determines which (outer, inner) concept pairs of the two factor
lattices are realized by the combined context, then renders an interactive
navigator with the D3.js force-directed layout.

For publication figures, use the matplotlib backend and save to any format
matplotlib supports:

```python
from conceptflow.visualization import plot_lattice, plot_nested

fig = plot_nested(outer_view, backend="matplotlib", show_object_labels=True)
fig.savefig("nested.pdf")

fig = plot_lattice(lattice, backend="matplotlib")
fig.savefig("lattice.svg")

# Lattices too large for DimFlux: use a quick layered layout instead.
fig = plot_lattice(lattice, backend="matplotlib", layout="none", show_object_labels=False)
```

DimFlux layouts take seconds for small lattices but can take minutes once a
lattice has a few hundred concepts (ConceptFlow warns above 150). `timeout_ms`
(default 1000) limits each optimization phase loosely, not the total run time.

The realized and unrealized combinations (filled and hollow inner nodes) are
also available as data:

```python
from conceptflow.exploration import nested_combinations

table = nested_combinations(outer_view)   # one row per (outer, inner) pair
table[~table.realized]                    # attribute combinations that never occur
``` See
[docs/nested_diagram_tutorial.md](https://github.com/anuragxorma/conceptflow/blob/main/docs/nested_diagram_tutorial.md) for the full
mathematical derivation.

## `.cxt` input/output

```python
from conceptflow.io import read_cxt, write_cxt

ctx = read_cxt("input.cxt")
write_cxt(ctx, "output.cxt")
```

## Metrics

```python
from conceptflow.metrics import (
    attribute_set_support,
    implication_confidence,
    implication_exceptions,
)

support = attribute_set_support(ctx, attributes=[0, 1])

confidence = implication_confidence(
    ctx,
    premise=[0],
    conclusion=[1],
)

# Objects with the premise but not the conclusion (empty iff the rule holds).
exceptions = implication_exceptions(ctx, premise=["a"], conclusion=["b"])
```

## Feature extraction

ConceptFlow can transform FCA structures into machine-learning feature
representations.

The original formal context represents primitive binary attributes:

```text
object -> attributes
```

Feature extraction instead represents objects using formal concepts:

```text
object -> concept memberships
```

Each extracted feature corresponds to a formal concept rather than a
single attribute.

This produces:
- interpretable symbolic features,
- closure-based representations,
- hierarchical concept-derived features.

### Example

```python
import numpy as np

import conceptflow as cf
from conceptflow.feature_extraction import (
    ConceptMembershipEncoder,
)

context = cf.FormalContext.from_array(
    np.array([
        [1, 1, 0],
        [1, 0, 1],
        [0, 1, 1],
    ]),
)

encoder = ConceptMembershipEncoder()

features = encoder.fit_transform(context)

print(features)
```
The encoder supports multiple output representations:

```text
"dataframe"  -> pandas DataFrame
"array"      -> NumPy ndarray
"sparse"     -> scipy.sparse CSR matrix
```

## Decomposition

ConceptFlow includes an initial decomposition module for ordinal
two-factorization.

```python
from conceptflow.decomposition import ExactOrdinalTwoFactorizer

model = ExactOrdinalTwoFactorizer()
model.fit(ctx)

factor_1, factor_2 = model.factors_
coverage = model.coverage_
```

`ExactOrdinalTwoFactorizer` implements *exact* ordinal two-factorization
(Dürrschnabel & Stumme, Algorithm 1) for contexts whose complement concept
lattice has order dimension at most 2. Finding the realizer pair of linear
extensions is a brute-force search, intended for small contexts.

The full `Ord2Factor` algorithm for maximal ordinal two-factorizations
(arbitrary order dimension) is a documented stub — calling `.fit()` raises
`NotImplementedError` — and is reserved for future implementation. It
additionally requires algorithms for:

- maximal induced bipartite subgraph selection,
- incompatibility-graph reduction,
- large-scale approximate ordinal factorizations.

## Implications (stem basis)

ConceptFlow computes the **Duquenne-Guigues (stem) basis**: the smallest set
of exact implications logically equivalent to every implication that holds
in a context.

```python
from conceptflow.rules import ImplicationBasisEstimator

estimator = ImplicationBasisEstimator(output="dataframe")
features = estimator.fit_transform(ctx)  # one bool column per implication

for implication in estimator.get_implications():
    print(
        implication.premise_names(estimator.context_),
        "->",
        implication.conclusion_names(estimator.context_),
    )
```

`fit` computes the basis (`get_implications()` returns it as `Implication`
objects); `transform` re-derives, for any object (not just the ones it was
fit on), which implications' premises that object's own attributes satisfy —
one boolean feature per implication. The underlying algorithm
(`conceptflow.algorithms.compute_canonical_basis`) is Ganter's NextClosure on
integer bitsets, which visits only intents and pseudo-intents; closures under
the implications found so far are vectorised with NumPy and abandoned as soon
as they fail the canonicity test. A 357 x 42 context with 1,376 implications
takes about 0.2 s. `compute_canonical_basis_bruteforce` checks every attribute
subset against the pseudo-intent definition and is
kept as a reference implementation. See
[docs/nested_diagram_tutorial.md](https://github.com/anuragxorma/conceptflow/blob/main/docs/nested_diagram_tutorial.md) for how
implications relate to hollow nodes in a nested diagram, and
[examples/eurovision_nested_diagram.py](https://github.com/anuragxorma/conceptflow/blob/main/examples/eurovision_nested_diagram.py)
for a full worked example.

### Near-implications (Luxenburger basis)

Rules that hold for most but not all objects come from neighbouring concepts:
for an upper neighbour `(A1, B1)` of `(A2, B2)`, the rule `B1 -> B2 \ B1` has
confidence `|A2| / |A1|`. Each rule lists its exception objects.

```python
from conceptflow.algorithms import compute_luxenburger_basis

rules = compute_luxenburger_basis(ctx, min_confidence=0.8, min_support=3)

for rule in rules:
    print(
        rule.premise_names(ctx), "->", rule.conclusion_names(ctx),
        f"conf={rule.confidence:.2f}", f"support={rule.support}/{rule.premise_support}",
        "exceptions:", rule.exception_names(ctx),
    )
```

`min_support` is an object count (int) or a fraction of all objects (float).

## Design philosophy

ConceptFlow separates:

- mathematical FCA objects,
- FCA algorithms,
- preprocessing and conceptual scaling,
- scikit-learn compatible estimators,
- input/output,
- metrics,
- visualization and exploration.

Visualization and nested exploration are intentionally isolated from the core computation layer.

## scikit-learn compatibility

ConceptFlow follows the scikit-learn estimator interface where this is
natural and useful for FCA workflows.

The library supports:

- `fit`, `transform`, and `fit_transform`
- sklearn-style estimators and transformers
- `Pipeline` compatibility
- estimator cloning through `sklearn.base.clone`
- `set_output`
- feature-name propagation

Examples include:

- `ConceptualScaler`
- `ConceptMembershipEncoder`
- `ConceptLatticeEstimator`
- `ImplicationBasisEstimator`
- `ExactOrdinalTwoFactorizer`

ConceptFlow is inspired by the design philosophy of projects such as
scikit-learn and scikit-mine, while remaining focused on Formal Concept
Analysis and symbolic data analysis.

Because FCA works with symbolic and order-theoretic structures rather than
purely numerical arrays, not all components are expected to satisfy the full
`sklearn.utils.estimator_checks.check_estimator` suite. Concretely,
`ConceptualScaler`, `ConceptMembershipEncoder`, `ConceptLatticeEstimator`, and
`ImplicationBasisEstimator` each pass every check except
`check_transformer_preserve_dtypes`, since all four always output `bool`
(concept membership, scaled attributes, and premise-satisfaction indicators
are yes/no predicates, not values to dtype-cast). `ConceptualScaler`
additionally xfails `check_fit_idempotent` for `NominalScale`-based
configurations, since a nominal scale's vocabulary is fixed at fit time by
design (see the note in `tests/test_check_estimator.py`).

Core FCA structures such as:

- `FormalContext`
- `Concept`
- `ConceptLattice`

remain FCA-native mathematical objects rather than purely numerical sklearn
estimators.

## Architecture

ConceptFlow follows a layered architecture that separates mathematical FCA
structures from machine-learning workflows and visualization systems.

```text
core/
    Mathematical FCA structures:
    FormalContext, Concept, ConceptLattice

algorithms/
    Enumeration and FCA algorithms:
    NextClosure, CloseByOne, derivation operators

preprocessing/
    Conceptual scaling and sklearn-style preprocessing

cluster/
    Lattice estimators and sklearn-compatible wrappers

feature_extraction/
    FCA-derived feature representations

metrics/
    FCA metrics and implication measures

rules/
    Duquenne-Guigues (stem) basis computation and its sklearn estimator

decomposition/
    Ordinal factorization and FCA decomposition methods

io/
    Burmeister .cxt input/output support

visualization/
    Backend-neutral graph abstractions, layouts, and rendering
```

A central design principle of ConceptFlow is strict separation between:

- FCA computation,
- graph/layout computation,
- visualization/rendering,
- exploration interfaces.

This separation allows the same FCA structures to be reused across:
- machine learning workflows,
- symbolic analysis,
- visualization systems,
- interactive conceptual exploration.

## Further reading

- [docs/architecture.md](https://github.com/anuragxorma/conceptflow/blob/main/docs/architecture.md) — the full layered-architecture design.
- [docs/nested_diagram_tutorial.md](https://github.com/anuragxorma/conceptflow/blob/main/docs/nested_diagram_tutorial.md) — tutorial on nested line diagrams.
- [docs/nested_diagram_construction.md](https://github.com/anuragxorma/conceptflow/blob/main/docs/nested_diagram_construction.md) — mathematical construction of the join-closure algorithm.
- [Eurovision case study](https://kde.cs.uni-kassel.de/blogs/esc) — worked example and interactive diagram.

## License

BSD-3-Clause. See [LICENSE](LICENSE).

## Status

ConceptFlow is currently experimental.
```