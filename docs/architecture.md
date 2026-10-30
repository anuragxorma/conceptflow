# ConceptFlow Architecture

## Overview

ConceptFlow is designed as a layered FCA framework that combines:

- Formal Concept Analysis,
- symbolic data analysis,
- scikit-learn compatible workflows,
- graph-based visualization systems.

The architecture intentionally separates mathematical FCA computation from
visualization and exploration logic.

---

# Architectural layers

Each subpackage has one job; the arrows of dependence point towards `core/`.

## `core/`

The mathematical foundation: `FormalContext`, `Concept`, `ConceptLattice`
(with navigation: parents, children, join, meet, up- and down-sets),
`ManyValuedContext`, `IncompleteContext` (three-valued contexts with
certain/possible answers), the context operations (`apposition`,
`subposition`, `direct_sum`, `direct_product`, `complement`, `dual`) and
scale-measures (`scale_measure`, `is_scale_measure`, `refines`).
Independent of visualization, machine learning and rendering.

## `algorithms/`

Exact FCA computation on the core structures: derivation and closure,
concept enumeration (NextClosure, Close-by-One with FCbO pruning, brute
force; iceberg lattices), cover relations, the canonical (Duquenne–Guigues)
basis (LinCbO, NextClosure), the proper-premise basis, the Luxenburger basis
of association rules, minimal generators, and clarification and reduction of
contexts.

## `metrics/`

Measures of concepts and rules: support, confidence and exceptions of
implications; exact stability, its bounds and the Δ-measure; entropy and
information gain with respect to labels; linearity and distributivity
indices, lattice width and height, incomparable pairs and concept levels;
and the quality metrics of line diagrams (`drawing_quality.py`), which take
a laid-out `GraphData` or a networkx graph.

## `preprocessing/`

Conceptual scaling of many-valued data into formal contexts:
`ConceptualScaler` (a scikit-learn transformer) with nominal,
contranominal, dichotomic, ordinal, threshold, interordinal and general
scales, missing-value policies and handling of values unseen during fit.

## `cluster/`, `feature_extraction/`, `rules/`, `decomposition/`

scikit-learn estimators on top of the algorithms:

- `cluster.ConceptLatticeEstimator` and
  `feature_extraction.ConceptMembershipEncoder`: concept-membership
  features, optionally from an iceberg lattice or the most stable concepts;
- `rules.ImplicationBasisEstimator` (premise-satisfaction features) and
  `rules.ImplicationOutlierDetector` (objects breaking rules that hold for
  almost all others);
- `decomposition.GreConD` and `decomposition.ASSO` (Boolean matrix
  factorization), `decomposition.Ord2Factor` and
  `ExactOrdinalTwoFactorizer` (ordinal two-factorization, with the bipartite
  subgraph methods in `decomposition/bipartite.py`).

`validation/` holds the input checks they share.

## `exploration/`

Nested conceptual exploration: `ExplorationBuilder` scales a many-valued
context step by step into a tree of `ExplorationView`s;
`nested_view`/`realized_pairs` compute nested line diagrams of two
contexts.

## `visualization/`

Line diagrams and nested line diagrams. Lattices are first turned into
backend-neutral `GraphData`, laid out (DimFlux or a layered layout, both
keeping edges off unrelated concepts) and then rendered: interactive D3
pages (`HTMLFigure`), matplotlib, TikZ or Graphviz.

## `io/`, `datasets/`, `quality`

- `io`: Burmeister `.cxt`, FCART/fcapy JSON and CSV files.
- `datasets`: the bundled FCA-repository contexts, downloads from
  fcarepository.org, the UCI Mushroom data and random contexts.
- `quality`: data-quality reports (duplicate, constant and derivable
  objects and attributes; rule violations; explanations of objects and
  rules).

---

# Design philosophy

ConceptFlow follows several core principles:

## 1. FCA-native foundations

The framework prioritizes mathematically meaningful FCA structures rather than
forcing everything into generic numerical machine-learning abstractions.

---

## 2. sklearn compatibility where useful

ConceptFlow adopts sklearn-compatible APIs for:

- preprocessing,
- feature extraction,
- decomposition,
- estimators,
- pipelines.

Core FCA structures remain FCA-native objects.

---

## 3. Separation of concerns

The framework intentionally separates:

- mathematical structures,
- algorithms,
- feature extraction,
- layouts,
- rendering,
- exploration.

This improves maintainability and allows multiple visualization systems to
reuse the same FCA computation layer.

---

## 4. Backend-neutral visualization

Visualization should operate on abstract graph representations rather than
directly on FCA structures.

This enables:
- multiple renderers,
- reusable layouts,
- independent exploration systems.

---

## 5. Research-oriented extensibility

ConceptFlow is designed as a research framework rather than only a teaching
library.

The architecture therefore prioritizes:
- algorithmic extensibility,
- mathematical clarity,
- reproducibility,
- exploratory analysis of real data.