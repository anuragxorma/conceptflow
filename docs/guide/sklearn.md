# scikit-learn

ConceptFlow's estimators follow the scikit-learn API: `fit` returns the
estimator, fitted attributes end in `_`, parameters work with `get_params`,
`set_params`, `clone` and grid search, and they run in pipelines, column
transformers and with `set_output`.

| Estimator | Module | Does |
|---|---|---|
| `ConceptualScaler` | `conceptflow.preprocessing` | many-valued data → binary attributes ([guide](scaling.md)) |
| `ConceptLatticeEstimator` | `conceptflow.cluster` | binary data → one feature per concept |
| `ConceptMembershipEncoder` | `conceptflow.feature_extraction` | the same from a fixed lattice (`from_lattice`) |
| `ImplicationBasisEstimator` | `conceptflow.rules` | binary data → one feature per canonical-basis implication |
| `ImplicationOutlierDetector` | `conceptflow.rules` | outlier detection with implications |

## Concept features in a pipeline

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from conceptflow.cluster import ConceptLatticeEstimator
from conceptflow.preprocessing import ConceptualScaler, NominalScale, ThresholdScale

rng = np.random.default_rng(0)
data = pd.DataFrame(
    {"color": rng.choice(["red", "blue", "green"], 150), "score": rng.uniform(0, 100, 150)}
)
y = ((data.color == "red") & (data.score > 40)).astype(int)

pipe = Pipeline(
    [
        ("scale", ConceptualScaler([NominalScale("color"), ThresholdScale("score", [25, 50, 75])])),
        ("concepts", ConceptLatticeEstimator(select="stability", n_concepts=10)),
        ("classify", LogisticRegression()),
    ]
)
search = GridSearchCV(pipe, {"concepts__n_concepts": [5, 10]}, cv=3).fit(data, y)
print(search.best_params_)
```

Each concept feature is true for the objects having all attributes of the
concept's intent, so it is readable; `get_feature_names_out()` returns
stable concept ids, and `get_lattice()` the fitted lattice.

`ConceptLatticeEstimator` options:

- `algorithm`: `"nextclosure"`, `"closebyone"` or `"bruteforce"`;
- `min_support` / `max_concepts`: iceberg lattice and a hard size limit (see
  [Concept lattices](lattices.md#large-lattices));
- `select` + `n_concepts`: keep the best concepts by `"stability"`,
  `"delta"` or `"support"` (`n_concepts` is ignored when `select=None`, so
  grids can mix both);
- `output`: `"array"`, `"dataframe"` or `"sparse"`.

## Outlier detection with implications

`ImplicationOutlierDetector` learns rules that hold for (almost) all training
objects and flags objects that break them. Unlike most detectors it can say
*why*:

```python
from conceptflow.rules import ImplicationOutlierDetector

X = pd.DataFrame(
    [[1, 1, 0]] * 20 + [[0, 0, 1]] * 10 + [[1, 0, 0]],
    columns=["mammal", "warm-blooded", "scales"],
)
detector = ImplicationOutlierDetector(min_confidence=0.9).fit(X)

print(detector.predict(X)[-1])  # -1
print(detector.explain(X.tail(1))[["premise", "missing", "confidence"]])
```

- `score_samples`: 0 for objects breaking no rule, lower for more unusual
  objects (each broken rule counts `-log2(1 - confidence)`, with confidence
  smoothed so exact rules count more the more evidence they have);
- `contamination=0.05` (default) flags about that share of the training data,
  the most unusual objects. Scores are discrete, so objects tied at the
  threshold are all flagged; objects breaking no rule never are.
  `contamination="auto"` flags every object breaking any rule — easy to
  interpret, but often a large share on real data, since near-implications
  have exceptions by definition;
- `include_exact=True` also uses exact implications (confidence 1);
- a warning suggests higher `min_support` / `min_confidence` when very many
  rules are mined.

## Input conventions

Binary estimators accept boolean/0-1 arrays, sparse matrices, DataFrames and
`FormalContext` objects. For numbers, values greater than 0 are crosses. NaN
is rejected (scikit-learn's convention); if empty cells mean "no cross", pass
`FormalContext.from_dataframe(frame)` instead.

As in scikit-learn, an estimator fitted on named columns (a DataFrame or a
`FormalContext`) refuses named input whose columns differ or come in another
order; plain arrays are not checked.

## Compatibility

All estimators are tested with scikit-learn's `check_estimator` suite
(`parametrize_with_checks`), in several configurations each. They declare
their properties through scikit-learn tags: the transformers output booleans
(`preserves_dtype = []`), `ConceptualScaler` accepts strings, and the binary
estimators accept sparse input.

The only expected failure is `check_estimators_nan_inf` for
`ConceptualScaler` with the default `missing="error"`: NaN is rejected as
scikit-learn expects, but `inf` is a legitimate many-valued value (a
`ThresholdScale` handles it) and is not rejected.

Core FCA objects (`FormalContext`, `Concept`, `ConceptLattice`) are plain
mathematical objects, not estimators.
