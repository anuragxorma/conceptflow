# Drawing quality

How good is a line diagram? `conceptflow.metrics` scores a drawing on 14
aspects of readability, each a number in [0, 1] where 1 is best, following
scikit-learn's `*_score` convention. They make layouts comparable: two
layout algorithms on one lattice, or one algorithm across many lattices.

| Family | Metrics |
|---|---|
| Crossings | `edge_crossing_score`, `crossing_angle_score` |
| Layering | `layer_consistency_score`, `vertical_axis_balance_score` |
| Slopes | `slope_verticality_score`, `slope_economy_score`, `slope_parallelism_score`, `slope_standard_score`, `edge_length_uniformity_score`, `visual_chain_linearity_score` |
| Clearance | `node_node_conflict_score`, `node_edge_conflict_score`, `edge_edge_conflict_score`, `nested_suitability_score` |

What each one measures, and its parameters, is in the
[API reference](../api/metrics.md).

## Scoring a drawing

A drawing is a laid-out `GraphData` (from `plot_lattice(..., backend="graph_data")`,
`layered_layout` or `lattice_to_graph_data_dimflux`) or a `networkx.DiGraph`
whose nodes have `x` and `y` attributes and whose edges point from the lower
to the upper element. `drawing_quality_scores` computes all metrics at once:

```python
import conceptflow as cf
from conceptflow.datasets import load_context
from conceptflow.metrics import drawing_quality_scores, edge_crossing_score

lattice = cf.ConceptLattice.from_context(load_context("livingbeings_en"))
layered = cf.plot_lattice(lattice, backend="graph_data", layout="layered")

scores = drawing_quality_scores(layered)
print(round(scores["edge_crossing"], 2))  # 0.96
print(round(edge_crossing_score(layered), 2))  # 0.96
print(round(scores["layer_consistency"], 2))  # 0.99
```

## Comparing layouts

Scores are comparable between drawings of the same lattice. A `GraphData`
is scored as it is drawn: layouts such as DimFlux's, which the figures turn
upright, are scored upright.

```python
dimflux = cf.plot_lattice(lattice, backend="graph_data", layout="dimflux")
other = drawing_quality_scores(dimflux)
for name in ("edge_crossing", "slope_verticality", "nested_suitability"):
    print(name, round(scores[name], 2), round(other[name], 2))
```

Some metrics compare a drawing with what the lattice allows, so their
values depend on the lattice: `slope_parallelism_score`, for example, can
reach 1 only for distributive lattices. Compare them between drawings of one
lattice rather than between lattices.

## Averages and worst cases

The clearance and crossing-angle metrics average over nodes, vertices or
crossings. `aggregate="min"` gives the worst single term instead, and
`drawing_quality_scores(..., worst_case=True)` adds those as `<name>_min`:

```python
from conceptflow.metrics import node_node_conflict_score

print(round(node_node_conflict_score(layered, aggregate="min"), 2))  # 1.0
worst = drawing_quality_scores(layered, worst_case=True)
print(sorted(name for name in worst if name.endswith("_min"))[0])  # crossing_angle_min
```

Distances such as "too close" are relative to the average length of a cover
edge (`threshold_factor`), and angles have a tolerance (`angle_tolerance`),
so scores do not depend on the size or position of a drawing:

```python
print(round(node_node_conflict_score(layered, threshold_factor=1.0), 2))  # 0.8
```

## Limits

- `vertical_axis_balance_score` needs a unique top and bottom element, so it
  raises `ValueError` for iceberg lattices.
- The metrics score the geometry only: labels are not part of a drawing
  here, so label overlaps are not measured.
- Scores are not invariant to every change: `edge_length_uniformity_score`
  can change slightly when a drawing is mirrored.
