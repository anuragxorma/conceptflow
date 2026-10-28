# API stability and deprecations

## What is public

The public API is everything documented in the [API reference](api/core.md):
the classes and functions you can import from `conceptflow` and its
subpackages (`conceptflow.core`, `conceptflow.algorithms`,
`conceptflow.preprocessing`, ...) whose names do not start with an
underscore, with their documented parameters, attributes and behaviour.

Anything whose name starts with `_` — modules such as
`conceptflow.algorithms._bitset`, functions, methods and attributes — is
internal and may change in any release.

## Versions

ConceptFlow uses [semantic versioning](https://semver.org): versions are
`MAJOR.MINOR.PATCH`.

- **Patch releases** (0.3.0 → 0.3.1) only fix bugs.
- **Minor releases** (0.3 → 0.4) add features. Before 1.0 they may also
  change the public API, but only as described below and always listed in
  the [changelog](changelog.md).
- **From 1.0 on**, incompatible changes to the public API happen only in
  major releases.

## Deprecations

When something public has to change:

1. The old name or behaviour keeps working for **at least one minor
   release**, and using it emits a `DeprecationWarning` that names the
   replacement and the version in which the old one will be removed.
2. The changelog lists it under *Deprecated*, and later under *Removed*.

A fix that makes a result correct is not a deprecation: when a result was
wrong, it is fixed in the next release and listed under *Fixed* or
*Changed*.

## Not covered

These may change in any release (changes are noted in the changelog):

- the order of items where no order is documented;
- floating-point results within numerical precision;
- drawings: layouts, coordinates, styling and the exact HTML, SVG, TikZ or
  DOT text;
- the wording of error messages and warnings;
- running time and memory use.

## Currently deprecated

| Deprecated | Use instead | Removed in |
|---|---|---|
| `conceptflow.cluster.ConceptLattice` | `conceptflow.cluster.ConceptLatticeEstimator` | 1.0 |
