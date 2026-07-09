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

## `core/`

The mathematical foundation of the framework.

Contains:

- `FormalContext`
- `Concept`
- `ConceptLattice`
- many-valued contexts

Responsibilities:

- closure operators,
- derivation operators,
- lattice representation,
- concept structure.

The `core/` layer is intentionally independent from:
- visualization,
- machine learning,
- rendering backends.

---

## `algorithms/`

Contains FCA algorithms operating on the core structures.

Examples:

- NextClosure
- CloseByOne
- derivation operators
- concept enumeration

Responsibilities:

- exact FCA computation,
- enumeration algorithms,
- order-theoretic algorithms.

This layer should remain computation-focused and renderer-independent.

---

## `preprocessing/`

Contains conceptual scaling and sklearn-compatible preprocessing tools.

Examples:

- `ConceptualScaler`
- nominal scaling
- ordinal scaling
- threshold scaling

Responsibilities:

- transformation from many-valued data into formal contexts,
- sklearn pipeline compatibility,
- sparse feature generation.

---

## `cluster/`

Contains sklearn-style lattice estimators.

Examples:

- `ConceptLattice`

Responsibilities:

- estimator wrappers,
- sklearn-compatible interfaces,
- pipeline integration.

This layer bridges FCA computation with machine learning workflows.

---

## `feature_extraction/`

Transforms FCA structures into machine-learning feature representations.

Examples:

- `ConceptMembershipEncoder`

Responsibilities:

- concept-derived feature spaces,
- sparse symbolic representations,
- interpretable feature extraction.

---

## `metrics/`

Contains FCA metrics and implication measures.

Examples:

- support
- confidence

Responsibilities:

- implication analysis,
- rule quality measures,
- symbolic metric computation.

---

## `decomposition/`

Contains FCA decomposition and factorization methods.

Examples:

- `ExactOrdinalTwoFactorizer`

Responsibilities:

- ordinal factorization,
- Ferrers relations,
- incompatibility-graph methods,
- order-theoretic decomposition.

The full `Ord2Factor` algorithm is reserved for future implementation.

---

## `io/`

Input/output functionality.

Examples:

- Burmeister `.cxt` support

Responsibilities:

- serialization,
- dataset interchange,
- external FCA compatibility.

---

## `visualization/`

Visualization and exploration infrastructure.

Planned responsibilities:

- backend-neutral graph structures,
- layout algorithms,
- DimFlux integration,
- interactive exploration,
- nested diagrams.

Visualization is intentionally separated from FCA computation.

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
- experimental FCA workflows.