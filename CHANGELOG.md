# Changelog

## 0.6.0 (2026-10-30)

New: scale-measures (coarser views of a context, with verification and
refinement pairs), order indices of lattices, and 14 quality metrics for
line diagrams. CI now also tests the built packages and Python 3.13. No
existing behaviour changes, except that the README's first example saves a
matplotlib figure instead of compiling TikZ.

### Added

- **Scale-measures** (Hanika & Hirth, 2022) in `conceptflow.core`:
  `scale_measure(context, attributes)` builds a coarser view on the same
  objects from attribute projections and conjunctions, in a canonical
  attribute order; `is_scale_measure` and `scale_measure_problems` check the
  property attribute by attribute; `refines(coarse, fine)` tells whether every
  extent of one view is an extent of the other (a refinement pair).
- **Order indices** in `conceptflow.metrics`: `lattice_width` (largest
  antichain), `lattice_height` (longest chain), `incomparable_pairs`, and
  `concept_levels` (the longest chain below each concept).
- `conceptflow.visualization.layout_positions(graph_data)`: where each concept
  is drawn, as `{extent: (x, y)}` keyed by object names, in the upright
  coordinates of the figures, so drawings of a refinement pair can be
  compared concept by concept.
- A guide page, *Scale-measures and order indices*.
- **Drawing quality metrics** in `conceptflow.metrics` (contributed by
  marcelnoehre in #2): 14 scores in [0, 1] for line diagrams, in four
  families (crossings, layering, slopes, clearance), with
  `drawing_quality_scores` computing all of them and optional worst cases,
  and `bottleneck_clearance_radius`. They take a laid-out `GraphData` or a
  `networkx.DiGraph`; a `GraphData` is scored as drawn, so DimFlux layouts
  are scored upright. On all 635 drawings of a drawing-algorithm study (eight
  layout algorithms, 12 to 781 concepts) they reproduce the original
  implementation's scores to within 1e-9. Guide page *Drawing quality*.

### Changed

- The README's first example saves a matplotlib figure instead of
  compiling TikZ, so it runs after a plain `pip install` without LaTeX; the
  README's examples are now tested like the documentation's.
- CI also tests what users install: a new job builds the wheel and the
  sdist, checks them with `twine check --strict`, installs each into its own
  clean environment and runs a smoke test from outside the repository
  (`.github/scripts/smoke_test_installed.py`), which also verifies that the
  package, not the source tree, is imported. The same script passed against
  the published 0.5.0 from PyPI.
- CI runs the tests on Python 3.13 as well. On scikit-learn 1.5.2 only the
  estimator checks are left out (they need scikit-learn 1.6); the
  scikit-learn compatibility and documentation tests, previously skipped
  there, pass and now run.
- New workflow tests run whole chains on the bundled contexts too large for
  exhaustive checks (lattice, canonical basis and GreConD, validated through
  their defining properties) and from a raw table to a tuned classifier with
  `GridSearchCV` and to rule-based outlier explanations.

## 0.5.0 (2026-10-30)

Mostly fixes from audits of the whole package: every part was checked
against brute force or the definitions in the literature. The visualization
had most of the bugs; line diagrams and nested line diagrams are now also
tested in a real browser in CI. Two results change on purpose: Ord2Factor's
greedy and annealing methods (now inclusion-minimal, see *Changed*) and the
coordinates of some layouts (edges are kept off unrelated concepts, see
*Fixed*).

ConceptFlow is now **beta** (it was alpha/experimental): results are tested
against independent references, and the API changes only with deprecation
warnings, as described in the API stability policy.

### Deprecated

- `debug_filled_pairs` and `debug_bottom_outer` in
  `conceptflow.visualization`: development tools that only print. They keep
  working with a `DeprecationWarning` and will be removed in 0.6.0; the
  filled pairs are available as data from
  `conceptflow.exploration.realized_pairs`.

### Removed

- `conceptflow.model_selection`: an empty placeholder package, never
  documented. scikit-learn's own `GridSearchCV` and `cross_val_score` work
  with ConceptFlow's estimators in pipelines.

### Changed

- `Ord2Factor` uses `method="annealing"` by default instead of `"exact"`.
  The exact SAT method is exponential: about 20 seconds for a 15 x 15
  context, and it ran for hours a little beyond, without a warning.
  Annealing gave the same results on small contexts and takes seconds on
  larger ones. `"exact"` gets a time limit, `max_time` (default 60 seconds;
  `None` for none), after which `fit` raises `TimeoutError`;
  `minimum_bipartite_deletion` has the same `max_time` (default: none).
- `Ord2Factor(method="greedy")` and `method="annealing"` (and
  `greedy_bipartite_deletion`, `annealing_bipartite_deletion`) now return
  inclusion-minimal results, as the paper states: they end with a pass that
  keeps every cross whose return still allows two ordinal factors. The
  paper's algorithms alone left a droppable cross in about a fifth of small
  random cases. Greedy results improve noticeably (11% fewer dropped crosses
  on 60 random contexts; on `triangles_en` 3 instead of 10), so results
  differ from 0.4.0.
- The interactive pages load D3 in an exact version (7.9.0) instead of the
  newest 7.x, so a D3 release cannot change existing diagrams.
- New browser tests (`tests/test_browser.py`) open the interactive pages in
  headless Firefox: drawing, reduced labels, clicks, nested navigation,
  downloads, narrow pages and notebook embedding. They run in CI and are
  skipped locally unless Playwright is installed.
- The visualization guide has a *Known limitations* section.
- `tests/test_oracles.py` checks the mathematical core against independent
  references written from the definitions (concepts, bases, generators,
  Luxenburger basis, stability, incomplete contexts, GreConD and ASSO as in
  their papers, optimal ordinal factorization, nested diagrams, ordinal
  scales). The browser tests wait for conditions instead of fixed times.
- The architecture page describes the current package (it still listed
  planned parts and an outdated estimator name); `read_csv` and
  `redundancy_report` document how they treat short rows and empty
  attributes.

### Fixed

- A `GraphData` built by hand (without the labels `lattice_to_graph_data`
  stores) could label an object of an iceberg lattice at a concept that is
  not its object concept. `lattice_to_graph_data` now records `is_iceberg`
  and `min_support` in the graph's metadata, and the derived labels leave
  such objects out.
- Integer positions out of range in `object_derivation` and similar
  methods now say when a name with that number exists (names are strings:
  `"5"`, not `5`).
- `write_cxt`, `write_json` and `write_csv` create missing folders, like the
  figure writers already did.
- `make_random_context` rejected NumPy Generators
  (`random_state=np.random.default_rng(0)`), although its documentation
  promised them; ints and `RandomState` give the same contexts as before.
- Line diagrams could draw an edge straight through a concept it does not
  connect, so the drawing showed cover relations that do not exist (layered
  layout: 7 of the bundled and test contexts, e.g. `animals_en`; DimFlux: 4,
  e.g. `missmarple_de`). Both layouts now move such concepts sideways, keeping
  every concept's height and the straight edges of the FCA convention. A
  concept is only moved where it keeps its distance to all others, so some
  edges in crowded layered rows can still pass close to a node; lattices
  above 2000 concepts are left as they are.
- LaTeX in attribute and object names was shown verbatim, e.g. `\sqrt{3}`
  in the bundled triangles data. Line diagrams (interactive, matplotlib,
  Graphviz) and nested diagrams (interactive and matplotlib) now show
  common LaTeX as symbols (`\sqrt{3}` → √3, `\alpha` → α, `\leq` → ≤,
  `x^2` → x²; `$…$` around LaTeX is dropped), and TikZ typesets it in math
  mode. Other names are shown unchanged.
- The interactive nested diagram showed attribute names rewritten: the part
  before `=` was dropped and underscores were title-cased, so `color!=red`
  appeared as `red` and `smoker=yes` and `married=yes` both as `yes`. Names
  are now shown exactly as given.
- Clicking one of the small preview dots inside a bubble of the interactive
  nested diagram did nothing; it now opens that bubble's view.
- An `object_label` function returning something other than a string (e.g.
  `len`) crashed the matplotlib backend; every backend now converts the
  result to text.
- Interactive line diagrams drew every node with an 18 px radius and nested
  diagrams every bubble with 48 px, so on larger lattices nodes overlapped
  and could not be clicked. The radius now shrinks where nodes are close, so
  neighbouring nodes never overlap, and line diagrams can be zoomed in far
  enough to bring small nodes back to full size.
- The interactive line diagram wrote each concept's full intent under its
  node, so labels were long and overlapped. It now uses reduced labelling
  like the other backends (attributes above, objects below) and accepts
  `show_attribute_labels`, `show_object_labels`, `max_label_items` and
  `object_label`. The full extent and intent stay in the tooltip and the
  details panel.
- "Save PDF" in the interactive nested diagram wrote an uncompressed image:
  6.9 MB for a ten-node diagram. It now writes a vector PDF (sharp at any
  zoom, a few KB) through svg2pdf.js; for names outside Latin-1, which
  jsPDF's standard fonts cannot show, it writes a compressed image (about
  200 KB).
- With several interactive figures in one Jupyter notebook only the first
  was drawn (the others failed with "redeclaration of let rootLayer"), and
  the figures' styles leaked into the notebook. Notebooks now show each
  figure in its own frame, sized to the figure. `write_html` and `open`
  still write the plain page.
- In notebooks without ipywidgets, the first DimFlux layout printed
  "TqdmWarning: IProgress not found" (from fcapy, a DimFlux dependency,
  which imports tqdm.auto); it is no longer shown.
- Interactive pages narrower than 560 px put the details panel below the
  drawing instead of leaving the drawing too little (or negative) width.
- Invalid plotting options now raise a clear `ValueError` or `TypeError`
  instead of drawing nonsense: a non-positive interactive `width` or
  `height`; negative or
  non-integer `max_label_items`; TikZ `width_cm` ≤ 0; `layered_layout` with
  `x_spacing` or `y_spacing` ≤ 0 or negative `sweeps`.
- `ExplorationBuilder.root`, `expand` and `expand_all` raise a clear
  `TypeError` when `scales` is not a list of `Scale` objects (e.g. when the
  name is passed in its place).

## 0.4.0 (2026-10-28)

### Added

- **Boolean matrix factorization** in `conceptflow.decomposition`: `GreConD`
  (formal concepts as factors) and `ASSO` (association-based factors), as
  scikit-learn transformers like `NMF`: `components_`, `transform` (factor
  memberships), `inverse_transform`, `coverage_` and `reconstruction_error_`.
- **`Ord2Factor`** is implemented (it used to raise `NotImplementedError`):
  large ordinal two-factorizations of any context, following Algorithm 2 of
  Dürrschnabel & Stumme (2023). The crosses to drop are found exactly with a
  SAT solver (`method="exact"`, MiniSat 2.2 via the new dependency
  `python-sat`) or heuristically (`"greedy"`, `"annealing"`), following
  Dürrschnabel, Hanika & Stumme (2022); the graph functions are in
  `conceptflow.decomposition.bipartite`.
- **`conceptflow.datasets`**: the 20 contexts of the FCA repository
  (fcarepository.org, CC0) bundled for offline use (`load_context`,
  `list_contexts`), `fetch_fca_repository` for the current versions,
  `load_mushroom` (UCI Mushroom, downloaded once and cached; the standard
  8124 × 119 context or the raw many-valued data) and `make_random_context`.
- `ConceptualScaler(handle_unknown="ignore")`: values that `NominalScale` or
  `ContranominalScale` did not see during `fit` no longer have to raise (as
  in scikit-learn's `OneHotEncoder`), e.g. in cross-validation.
- **Benchmarks** against fcapy and concepts (`benchmarks/`, with results in
  the documentation).
- Documentation: API stability and deprecation policy, datasets,
  factorization and benchmark pages; `CITATION.cff` and `CONTRIBUTING.md`;
  continuous integration (tests on Python 3.12 and 3.14 with the oldest and
  newest supported scikit-learn, the documentation build and the linter, on
  every push and weekly).
- Examples use the bundled contexts and run from any folder; a new example
  classifies the Mushroom data with concept features.

### Changed

- `compute_canonical_basis` uses **LinCbO** (Janoštík, Konečný and Krajča,
  2021) by default: the same basis, much faster on contexts with many
  attributes (UCI Mushroom: 134 s → 17 s; random contexts with 70–100
  attributes: 40–85× faster). `algorithm="nextclosure"` selects the previous
  algorithm. `ImplicationBasisEstimator` and `ImplicationOutlierDetector`
  benefit automatically.
- **Faster concept enumeration**: Close-by-One uses FCbO's pruning, and
  bitsets are converted with NumPy (Mushroom: concepts 26 s → 13 s, lattice
  with cover relation 42 s → 29 s).
- **Exact stability** is computed per concept by inclusion–exclusion over
  its lower neighbours instead of over the whole lattice: the same values,
  no longer quadratic (all 238,710 Mushroom concepts in about 40 s), and it
  now also works on iceberg lattices, including
  `ConceptLatticeEstimator(select="stability", min_support=...)`.
- **`ConceptualScaler` is much faster**: values are encoded column by column,
  each distinct value once (Mushroom: 49 s → 0.1 s), with the same output.
- **Concept ids** (`Concept.stable_id()`, the feature names of
  `ConceptLatticeEstimator` and `ConceptMembershipEncoder`, and node ids in
  drawings with `stable_ids=True`) are derived from the intent only, so the
  same concept keeps its id when rows are reordered or resampled (e.g.
  across cross-validation folds), as the documentation promised. Feature
  names differ from earlier versions.
- An integer scale source (`NominalScale(0)`) is a column position, as in
  scikit-learn's `ColumnTransformer`; it used to fail for DataFrames.
- Binary estimators reject `inf` in DataFrames, as they do in arrays.
- `ConceptLatticeEstimator`, `ConceptMembershipEncoder` and
  `ImplicationBasisEstimator` check `output` again at `transform`, after
  `set_params`.
- `ExactOrdinalTwoFactorizer` and `Ord2Factor` set `n_features_in_` and
  `feature_names_in_` and declare sparse input; `Ord2Factor` passes
  scikit-learn's estimator checks.
- The deprecated alias `conceptflow.cluster.ConceptLattice` now says that it
  will be removed in 1.0.
- Python 3.14 is supported.

## 0.3.1 (2026-10-27)

Bug fixes from an audit of 0.3.0.

### Fixed

- **Visualization**
  - Object or attribute names containing `</script>`, and titles containing
    quotes, no longer break the interactive HTML pages; names are shown as
    text, not HTML.
  - Lattices with two concepts were drawn as a single point by the default
    (DimFlux) layout; `backend="d3"` with `layout="none"` drew every node at
    the same spot (it now uses the layered layout).
  - Iceberg lattices drawn with matplotlib or TikZ labelled objects at a
    concept that is not their object concept.
  - matplotlib read `$...$` in names as math (and failed on e.g. `$\foo$`).
  - `.write()` of TikZ and Graphviz figures creates missing folders; TikZ
    refuses suffixes other than `.tex`/`.pdf` instead of writing LaTeX into
    e.g. a `.png`; Graphviz errors show Graphviz's message.
- **Contexts**
  - `FormalContext` and `IncompleteContext` are now truly immutable: their
    arrays are read-only (and copied from the input), so a context cannot be
    changed after creation.
  - `ManyValuedContext` equality returned a DataFrame and hashing failed.
  - `add_objects`/`add_attributes` rearranged wrongly shaped input without
    an error; they now check the shape.
  - A single string where a list of names is expected (e.g.
    `attribute_derivation("ab")`, `implication_confidence(ctx, "a", "b")`)
    is one name, not a set of one-letter names.
  - Lattices with the same concepts are equal whichever algorithm built them.
  - `FormalContext.from_array` accepts scipy sparse matrices.
- **Files**
  - `read_cxt` accepts trailing blank lines and a UTF-8 byte-order mark;
    `write_cxt` refuses names with line breaks, which `.cxt` cannot store.
  - Malformed JSON contexts raise `ValueError` instead of `KeyError`/`TypeError`.
- **Estimators**
  - `NominalScale`/`ContranominalScale` rejected known values when a column's
    type changed between fit and transform (e.g. ints become floats when a
    NaN appears), which broke `missing="false"`/`"unknown"` in
    cross-validation.
  - `ExactOrdinalTwoFactorizer` could run for hours on small inputs (it tried
    all pairs of linear extensions); it now uses a polynomial algorithm.
  - Binary estimators accept string arrays (`"x"`/`"."`, `"1"`/`"0"`) like
    DataFrames do.
  - `ConceptualScaler` fitted on a DataFrame raises a clear feature-name error
    for other columns; `ConceptMembershipEncoder.from_lattice` checks feature
    names; `ImplicationBasisEstimator` feature names are always unique.
- **Smaller fixes:** `min_support` accepts `Fraction`; clear errors for
  `algorithm=None` and a non-numeric `min_confidence`; `concept_entropy`
  returns `0.0` instead of `-0.0`; `IncompleteContext` names unknown
  objects/attributes in its errors.

## 0.3.0 (2026-10-27)

### Added

- **Documentation site** (MkDocs, in `docs/`): getting started, a user guide
  with one page per topic, the theory documents, an API reference generated
  from the docstrings, and this changelog. Every code example in the guide is
  run by the test suite. The README is now a short overview.
- **Lattice navigation** on `ConceptLattice`: `join`, `meet`, `upset`,
  `downset`, `atoms`, `coatoms`, `object_concept` (γ), `attribute_concept`
  (μ), `concept_from_objects`, `concept_from_attributes`.
- **Iceberg lattices:** `min_support=` on `ConceptLattice.from_context`,
  `enumerate_concepts` and `ConceptLatticeEstimator`; only frequent concepts
  are enumerated.
- **Size guard:** a `LargeLatticeWarning` once an enumeration passes 50,000
  concepts (computation continues), and an opt-in `max_concepts=` that stops
  with `ConceptLimitExceeded`.
- **Concept measures** in `conceptflow.metrics`: exact `concept_stability`,
  `stability_bounds`, `delta_measure`, `concept_entropy`,
  `concept_information_gain`, `linearity_index`, `distributivity_index`.
- `ConceptLatticeEstimator(select=..., n_concepts=...)` keeps the most stable
  (or highest-Δ, or largest) concepts as features.
- `compute_minimal_generators` (keys) and `compute_proper_premise_basis`
  (canonical direct basis).
- **Context handling:** `FormalContext.transpose`, `subcontext`, `rename`,
  `add_objects`, `add_attributes`, `remove_objects`, `remove_attributes`;
  `reduce_context` (full reduction); JSON (FCART/fcapy layout) and CSV
  input/output.
- **Context operations** (`conceptflow.core`): `apposition`, `subposition`,
  `direct_sum`, `direct_product`, `complement`, `dual`; `nested_view` and
  `plot_nested(context, inner=other_context)` draw the nested diagram of any
  two contexts over the same objects.
- **Layouts and export:** `layout="layered"` (Sugiyama-style, for large
  lattices), `backend="tikz"` (TikZ/LaTeX, for lattices and nested diagrams)
  and `backend="graphviz"` (DOT).
- **Data quality** (`conceptflow.quality`): `check_implications`,
  `redundancy_report`, `explain_object`, `explain_rule`.
- `ImplicationOutlierDetector`: a scikit-learn outlier detector that flags
  objects breaking strong implications and explains each flag.

### Changed

- Cell values are read by one rule per cell, whatever the type of the table:
  numbers greater than 0 are crosses (previously any non-zero value; 0/1 data
  and counts are unaffected, ±1-encoded data now reads -1 as absent), and
  strings must be a known spelling (`"1"`, `"true"`, `"x"` / `"0"`,
  `"false"`, `"."`, `""`) — anything else, such as `"yes"`/`"no"`, raises
  instead of being guessed. Previously a string's meaning could depend on the
  other cells of the table.
- Integer object/attribute indices are range-checked everywhere (negative or
  too large indices raise `IndexError` instead of being ignored or wrapping
  around).
- Estimators fitted on named columns refuse input with different or
  reordered columns, as scikit-learn's estimators do.
- `plot_lattice(layout="none")` with matplotlib uses the new layered layout.
- `ConceptualScaler` declares string input support to scikit-learn, and all
  transformers declare that they do not preserve the input dtype; only one
  scikit-learn check remains an expected failure (see the scikit-learn
  guide).

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
