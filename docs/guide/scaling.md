# Conceptual scaling and missing values

Real data is many-valued: colours, ratings, measurements. **Conceptual
scaling** turns each many-valued attribute into binary attributes, chosen so
they express exactly the distinctions you care about. The choice of scale is
part of the analysis, so ConceptFlow asks for it explicitly.

## ConceptualScaler

```python
import pandas as pd
from conceptflow.preprocessing import ConceptualScaler, NominalScale, OrdinalScale, ThresholdScale

data = pd.DataFrame(
    {
        "color": ["red", "blue", "red", "green"],
        "risk": ["low", "medium", "high", "low"],
        "score": [12.0, 55.0, 81.0, 40.0],
    },
    index=["g1", "g2", "g3", "g4"],
)

scaler = ConceptualScaler(
    scales=[
        NominalScale("color"),
        OrdinalScale("risk", levels=["low", "medium", "high"], mode="ge"),
        ThresholdScale("score", thresholds=[25, 50, 75]),
    ],
    output="dataframe",
)
binary = scaler.fit_transform(data)
print(list(binary.columns))
```

`output` is `"array"` (default, for pipelines), `"dataframe"`, `"sparse"` or
`"context"` (a `FormalContext`). A scale names its column (`"color"`) or gives
its position (`NominalScale(0)` is the first column), as in scikit-learn's
`ColumnTransformer`; with a plain array, binary attributes are then named
after `x0`, `x1`, ...

Nominal vocabularies are fixed when fitting: by default, transforming a value
not seen during `fit` raises instead of silently producing an all-false row.
In cross-validation a rare value can be missing from a training fold; there,
`handle_unknown="ignore"` (as in scikit-learn's `OneHotEncoder`) gives an
unseen value none of the `x=v` attributes and all of the `x!=v` attributes:

```python
train = pd.DataFrame({"color": ["red", "blue"]})
scaler = ConceptualScaler([NominalScale("color")], handle_unknown="ignore").fit(train)
print(scaler.transform(pd.DataFrame({"color": ["green"]})).tolist())  # [[False, False]]
```

## Scale types

| Scale | Binary attributes for value *v* |
|---|---|
| `NominalScale("color")` | `color=v` (one per observed value) |
| `ContranominalScale("color")` | `color!=w` for every other value *w* |
| `DichotomicScale("minor", true_value="minor", false_value="major")` | the two values explicitly |
| `OrdinalScale("size", levels, mode="ge")` | `size>=l` for every level *l* ≤ *v* (`"le"` and `"exact"` also available) |
| `ThresholdScale("bpm", thresholds=[100, 150])` | `bpm>=t` for every threshold *t* ≤ *v* |
| `InterordinalScale("size", levels)` | both `size<=l` and `size>=l` |
| `GeneralScale("pH", mapping={...})` | any explicit value → set-of-attributes mapping |

```python
from conceptflow.preprocessing import GeneralScale, InterordinalScale

GeneralScale(
    "pH",
    mapping={
        "normal": [],
        "pathological": ["pathological pH"],
        "dangerous": ["pathological pH", "dangerous pH"],
    },
)
InterordinalScale("size", levels=["S", "M", "L"])
```

Choosing scales automatically is an open research question (several scalings
of the same attribute can all be sensible), so ConceptFlow does not guess.

## Missing values

A missing value in many-valued data (None, NaN, NaT, `pd.NA` or the string
`"?"`) is ambiguous: the attribute may not apply to the object, or its value
may be unknown. ConceptFlow never silently turns it into crosses or
non-crosses; `missing=` makes the choice explicit:

| `missing=` | Meaning |
|---|---|
| `"error"` (default) | raise a `ValueError` naming the attribute and the objects |
| `"false"` | the object gets none of that scale's attributes (closed world) |
| `"unknown"` | attributes true for every possible value stay known-yes, those true for none known-no, the rest become unknown |

```python
gaps = data.copy()
gaps.loc["g2", "score"] = None

try:
    ConceptualScaler(scales=[ThresholdScale("score", [50])]).fit(gaps)
except ValueError as error:
    print(error)

closed_world = ConceptualScaler(
    scales=[ThresholdScale("score", [50])], missing="false", output="dataframe"
).fit_transform(gaps)
print(closed_world.loc["g2"].tolist())  # [False]

unknown = ConceptualScaler(
    scales=[ThresholdScale("score", [50])], missing="unknown", output="context"
).fit_transform(gaps)
print(unknown.n_unknown)  # 1
```

With `missing="unknown"`, `"context"` output is an
[`IncompleteContext`](contexts.md#incomplete-contexts); `"array"` and
`"dataframe"` hold 1.0 / 0.0 and NaN for unknown entries (scikit-learn's
convention for missing values). Unknown entries are independent of each
other, so a rule implied by a scale itself (such as `size>=L → size>=M`) may
be reported as "possible" rather than "valid".

!!! note "Binary contexts are different"
    In a binary `FormalContext`, an empty cell simply means "no cross" (see
    [Formal contexts](contexts.md#creating-contexts)). The distinction above
    applies to many-valued data, where a missing value could mean several
    things.
