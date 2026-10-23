# Changelog

## 0.2.0 (2026-10-23)

### Breaking changes

- **Python 3.12 or newer is required.** The layout engine is now the
  [`dim-flux`](https://pypi.org/project/dim-flux/) package from PyPI, which
  needs Python 3.12. The bundled copy of DimFlux, its Java jar and the Java
  requirement are gone; `sympy` and `z3-solver` are no longer direct
  dependencies.
- **Missing values are no longer silently converted.**
  - `FormalContext` (constructor, `from_dataframe`, `from_array`) reads
    empty cells (None, NaN, `pd.NA`) as "no cross", like a printed cross
    table. Previously NaN became a cross (`bool(NaN)` is True) while None
    became a non-cross. The marker `"?"` (unknown) raises; use
    `IncompleteContext` for unknown entries.
  - The scikit-learn estimators still reject NaN, as scikit-learn requires;
    convert with `FormalContext.from_dataframe` for the cross-table reading.
  - `ConceptualScaler` raises by default (`missing="error"`). Previously NaN
    became its own category (`color=nan`) in nominal scales and "below every
    threshold" in threshold scales. To keep going, pass `missing="false"`
    (none of the scale's attributes) or `missing="unknown"` (see below), or
    fill missing values yourself before scaling.
  - The string `"?"` counts as missing (unknown) in `ConceptualScaler` and
    `IncompleteContext`.
- `plot_lattice(..., title=...)` defaults to `None`; the d3 backend still
  shows "Concept lattice" when no title is given.
- Index handling is stricter: `Concept` accepts only integer indices (NumPy
  integers are converted; floats and NumPy booleans raise), and `True`/`False`
  are no longer accepted as attribute or object indices in derivations.
- `FormalContext` copies the incidence array it is given, so changing the
  original array afterwards no longer changes the context.
- `compute_luxenburger_basis(concepts=...)` rejects concepts with indices
  outside the context and ignores duplicates.

### Added

- `compute_luxenburger_basis`: near-implications (confidence < 1) with
  support, confidence and exception objects, as `AssociationRule` objects.
- `conceptflow.metrics.implication_exceptions`: the objects violating a rule.
- `IncompleteContext`: yes / no / unknown contexts with certain/possible
  derivations, completions, and `implication_status` ("valid", "possible",
  "refuted" over all completions).
- `ConceptualScaler(missing=...)` with `"error"`, `"false"` and `"unknown"`;
  `ExplorationBuilder(missing=...)` with `"error"` and `"false"`.
- Static figures: `plot_lattice(..., backend="matplotlib")` and
  `plot_nested(..., backend="matplotlib")` return matplotlib figures for
  SVG/PDF/PNG export. With `layout="none"` large lattices render in
  well under a second.
- `nested_combinations(view)` and `realized_pairs(...)`: the realized and
  unrealized (outer, inner) concept pairs of a nested diagram as data.
- `timeout_ms` for DimFlux layouts in `plot_lattice`, `plot_nested` and
  `lattice_to_graph_data_dimflux`; a warning for lattices above 150 concepts.
- `compute_canonical_basis_bruteforce`: the definition-based reference
  implementation, kept for testing.

### Fixed

- `compute_canonical_basis` uses NextClosure instead of trying every
  attribute subset: a 240 x 30 context went from not finishing to 0.04 s.
- `ConceptLattice.edges`, `parents()` and `children()` are computed once and
  cached; previously every `parents()`/`children()` call recomputed all
  edges. `compute_hasse_edges` itself is also much faster when given the
  context.
- Concepts holding NumPy integer indices no longer produce wrong Hasse edges.
- Nested diagrams compute filled nodes with a direct test per pair, which is
  much faster on large diagrams (same result).

### Performance

Measured on a laptop; "before" is 0.1.1 or the first fixed version where
0.1.1 did not finish.

| Operation | Context | Before | Now |
|---|---|---|---|
| Canonical basis | 357 x 42 (1,376 implications) | 3.9 s | 0.22 s |
| Canonical basis | random 100 x 24, dense (8,423) | ~150 s | 0.95 s |
| Canonical basis | random 60 x 80 (3,432) | 23 s | 2.1 s |
| Concept enumeration, Close-by-One | random 300 x 40 (55,392 concepts) | 3.3 s | 0.8 s |
| Concept enumeration, NextClosure | random 300 x 40 | 3.0 s | 1.3 s |
| `ConceptLatticeEstimator.fit_transform` | random 300 x 40 | 22 s | 2.3 s |
| `ImplicationBasisEstimator.transform` | random 300 x 40 | 17 s | 0.09 s |
| Luxenburger basis, confidence >= 0.8, support >= 5 | random 300 x 40 | 2.7 s | 0.43 s |

- The canonical basis closes candidate sets under the implications found so
  far with vectorised NumPy passes (packed 64-bit words up to 64 attributes,
  a matrix-vector product beyond), and stops a closure as soon as it fails
  NextClosure's canonicity test. Pure-Python LinClosure was benchmarked as
  well and was slower here.
- NextClosure and Close-by-One run on integer bitsets.
- The transformers compute concept/premise membership for all objects with
  one matrix product instead of per-pair set tests.
- The Luxenburger basis only enumerates concepts that reach the support
  threshold and reads their lower neighbours off the attribute extents.

### Compatibility notes

- `ConceptualScaler` objects pickled with 0.1.x still load and behave like
  `missing="error"`.
- `import conceptflow` no longer imports the layout engine; it is loaded on
  first use.
