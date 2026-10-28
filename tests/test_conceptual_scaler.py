import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from sklearn.pipeline import Pipeline

from conceptflow import FormalContext, ManyValuedContext
from conceptflow.cluster import ConceptLatticeEstimator
from conceptflow.preprocessing import (
    ConceptualScaler,
    ContranominalScale,
    InterordinalScale,
    NominalScale,
    OrdinalScale,
    ThresholdScale,
)


def test_conceptual_scaler_nominal_output_context():
    df = pd.DataFrame(
        {"color": ["red", "blue", "red"]},
        index=["g1", "g2", "g3"],
    )

    scaler = ConceptualScaler(
        scales=[NominalScale("color")],
        output="context",
    )

    ctx = scaler.fit_transform(df)

    assert isinstance(ctx, FormalContext)
    assert ctx.objects == ("g1", "g2", "g3")
    assert ctx.attributes == ("color=blue", "color=red")
    assert ctx.incidence.tolist() == [
        [False, True],
        [True, False],
        [False, True],
    ]


def test_conceptual_scaler_output_dataframe():
    df = pd.DataFrame(
        {"color": ["red", "blue"]},
        index=["g1", "g2"],
    )

    scaler = ConceptualScaler(
        scales=[NominalScale("color")],
        output="dataframe",
    )

    out = scaler.fit_transform(df)

    assert list(out.columns) == ["color=blue", "color=red"]
    assert out.loc["g1", "color=red"]


def test_conceptual_scaler_output_array():
    df = pd.DataFrame(
        {"color": ["red", "blue"]},
        index=["g1", "g2"],
    )

    scaler = ConceptualScaler(
        scales=[NominalScale("color")],
        output="array",
    )

    out = scaler.fit_transform(df)

    assert isinstance(out, np.ndarray)
    assert out.dtype == bool
    assert out.shape == (2, 2)


def test_conceptual_scaler_ordinal_scale():
    df = pd.DataFrame(
        {"risk": ["low", "medium", "high"]},
        index=["g1", "g2", "g3"],
    )

    scaler = ConceptualScaler(
        scales=[
            OrdinalScale(
                "risk",
                levels=["low", "medium", "high"],
                mode="ge",
            )
        ],
        output="dataframe",
    )

    out = scaler.fit_transform(df)

    assert list(out.columns) == [
        "risk>=low",
        "risk>=medium",
        "risk>=high",
    ]
    assert out.loc["g1"].tolist() == [True, False, False]
    assert out.loc["g2"].tolist() == [True, True, False]
    assert out.loc["g3"].tolist() == [True, True, True]


def test_conceptual_scaler_threshold_scale():
    df = pd.DataFrame(
        {"score": [20, 45, 80]},
        index=["g1", "g2", "g3"],
    )

    scaler = ConceptualScaler(
        scales=[ThresholdScale("score", thresholds=[30, 60])],
        output="dataframe",
    )

    out = scaler.fit_transform(df)

    assert list(out.columns) == ["score>=30", "score>=60"]
    assert out.loc["g1"].tolist() == [False, False]
    assert out.loc["g2"].tolist() == [True, False]
    assert out.loc["g3"].tolist() == [True, True]


def test_conceptual_scaler_interordinal_scale():
    df = pd.DataFrame(
        {"size": ["S", "M", "L"]},
        index=["g1", "g2", "g3"],
    )

    scaler = ConceptualScaler(
        scales=[InterordinalScale("size", levels=["S", "M", "L"])],
        output="dataframe",
    )

    out = scaler.fit_transform(df)

    assert list(out.columns) == [
        "size<=S",
        "size<=M",
        "size<=L",
        "size>=S",
        "size>=M",
        "size>=L",
    ]
    assert out.loc["g2"].tolist() == [
        False,
        True,
        True,
        True,
        True,
        False,
    ]


def test_conceptual_scaler_accepts_many_valued_context():
    mvc = ManyValuedContext.from_dataframe(
        pd.DataFrame(
            {"color": ["red", "blue"]},
            index=["g1", "g2"],
        )
    )

    scaler = ConceptualScaler(
        scales=[NominalScale("color")],
        output="context",
    )

    ctx = scaler.fit_transform(mvc)

    assert isinstance(ctx, FormalContext)
    assert ctx.n_objects == 2


def test_conceptual_scaler_requires_explicit_scales():
    df = pd.DataFrame({"color": ["red", "blue"]})

    scaler = ConceptualScaler()

    with pytest.raises(ValueError):
        scaler.fit(df)


def test_conceptual_scaler_rejects_unknown_attribute():
    df = pd.DataFrame({"color": ["red", "blue"]})

    scaler = ConceptualScaler(scales=[NominalScale("size")])

    with pytest.raises(KeyError):
        scaler.fit(df)


def test_conceptual_scaler_rejects_invalid_output():
    df = pd.DataFrame({"color": ["red", "blue"]})

    scaler = ConceptualScaler(
        scales=[NominalScale("color")],
        output="wrong",
    )

    with pytest.raises(ValueError):
        scaler.fit(df)


def test_conceptual_scaler_pipeline_with_lattice():
    df = pd.DataFrame(
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
        ("lattice", ConceptLatticeEstimator(algorithm="nextclosure")),
    ])

    pipe.fit(df)

    lattice_step = pipe.named_steps["lattice"]

    assert len(lattice_step.concepts_) > 0
    assert lattice_step.context_.n_objects == 3

def test_conceptual_scaler_get_feature_names_out():
    df = pd.DataFrame(
        {"color": ["red", "blue"]},
        index=["g1", "g2"],
    )

    scaler = ConceptualScaler(
        scales=[NominalScale("color")],
        output="dataframe",
    )

    scaler.fit(df)

    assert scaler.get_feature_names_out().tolist() == [
        "color=blue",
        "color=red",
    ]


def test_conceptual_scaler_get_feature_names_out_validates_input_features():
    df = pd.DataFrame(
        {"color": ["red", "blue"]},
        index=["g1", "g2"],
    )

    scaler = ConceptualScaler(
        scales=[NominalScale("color")],
        output="dataframe",
    )

    scaler.fit(df)

    assert scaler.get_feature_names_out(["color"]).tolist() == [
        "color=blue",
        "color=red",
    ]

    with pytest.raises(ValueError):
        scaler.get_feature_names_out(["wrong"])

def test_conceptual_scaler_set_output_pandas():
    df = pd.DataFrame(
        {"color": ["red", "blue"]},
        index=["g1", "g2"],
    )

    scaler = ConceptualScaler(
        scales=[NominalScale("color")],
        output="array",
    )

    scaler.set_output(transform="pandas")

    out = scaler.fit_transform(df)

    assert isinstance(out, pd.DataFrame)
    assert list(out.columns) == [
        "color=blue",
        "color=red",
    ]



def test_conceptual_scaler_output_sparse():
    df = pd.DataFrame(
        {"color": ["red", "blue"]},
        index=["g1", "g2"],
    )

    scaler = ConceptualScaler(
        scales=[NominalScale("color")],
        output="sparse",
    )

    out = scaler.fit_transform(df)

    assert sparse.issparse(out)
    assert out.shape == (2, 2)
    assert out.dtype == bool

# ---------------------------------------------------------------------------
# Missing values
# ---------------------------------------------------------------------------


def make_missing_df():
    return pd.DataFrame(
        {
            "color": ["red", "blue", None],
            "score": [10.0, np.nan, 60.0],
        },
        index=["g1", "g2", "g3"],
    )


MISSING_SCALES = [NominalScale("color"), ThresholdScale("score", [20, 50])]


def test_missing_values_raise_by_default():
    with pytest.raises(ValueError, match="NaN.*color.*'g3'"):
        ConceptualScaler(scales=MISSING_SCALES).fit(make_missing_df())


def test_missing_values_raise_at_transform_time():
    complete = pd.DataFrame({"color": ["red", "blue"]}, index=["g1", "g2"])
    scaler = ConceptualScaler(scales=[NominalScale("color")]).fit(complete)

    with pytest.raises(ValueError, match="NaN"):
        scaler.transform(pd.DataFrame({"color": [None]}, index=["g9"]))


def test_missing_values_never_become_vocabulary():
    scaler = ConceptualScaler(scales=MISSING_SCALES, missing="false").fit(
        make_missing_df()
    )

    assert list(scaler.get_feature_names_out()) == [
        "color=blue",
        "color=red",
        "score>=20",
        "score>=50",
    ]


def test_missing_false_gives_no_scale_attributes():
    out = ConceptualScaler(
        scales=MISSING_SCALES, output="dataframe", missing="false"
    ).fit_transform(make_missing_df())

    assert out.loc["g3", ["color=blue", "color=red"]].tolist() == [False, False]
    assert out.loc["g2", ["score>=20", "score>=50"]].tolist() == [False, False]
    assert out.loc["g3", ["score>=20", "score>=50"]].tolist() == [True, True]


def test_missing_unknown_dataframe_uses_float_nan():
    out = ConceptualScaler(
        scales=MISSING_SCALES, output="dataframe", missing="unknown"
    ).fit_transform(make_missing_df())

    assert all(dtype == np.float64 for dtype in out.dtypes.tolist())
    assert out.loc["g3", ["color=blue", "color=red"]].isna().all()
    assert out.loc["g2", ["score>=20", "score>=50"]].isna().all()
    assert out.loc["g1", "color=red"] == 1.0


def test_missing_unknown_array_uses_nan():
    out = ConceptualScaler(scales=MISSING_SCALES, missing="unknown").fit_transform(
        make_missing_df()
    )

    assert out.dtype == float
    assert np.isnan(out[2, :2]).all()
    assert out[0].tolist()[:2] == [0.0, 1.0]


def test_missing_unknown_context_output_is_incomplete_context():
    from conceptflow import IncompleteContext

    ctx = ConceptualScaler(
        scales=MISSING_SCALES, output="context", missing="unknown"
    ).fit_transform(make_missing_df())

    assert isinstance(ctx, IncompleteContext)
    assert ctx.n_unknown == 4


def test_missing_policy_validation():
    with pytest.raises(ValueError, match="missing"):
        ConceptualScaler(scales=MISSING_SCALES, missing="drop").fit(make_missing_df())

    with pytest.raises(ValueError, match="sparse"):
        ConceptualScaler(
            scales=MISSING_SCALES, output="sparse", missing="unknown"
        ).fit(make_missing_df())


# ---------------------------------------------------------------------------
# handle_unknown
# ---------------------------------------------------------------------------


def test_handle_unknown_ignore_for_learned_vocabularies():
    train = pd.DataFrame({"color": ["red", "blue"], "size": ["S", "L"]})
    test = pd.DataFrame({"color": ["green", "red"], "size": ["S", "S"]})
    scales = [NominalScale("color"), ContranominalScale("color")]

    with pytest.raises(ValueError, match="handle_unknown"):
        ConceptualScaler(scales).fit(train).transform(test)

    scaler = ConceptualScaler(scales, output="dataframe", handle_unknown="ignore")
    out = scaler.fit(train).transform(test)
    assert out.iloc[0].to_dict() == {
        "color=blue": False, "color=red": False, "color!=blue": True, "color!=red": True,
    }
    assert out.iloc[1]["color=red"] and not out.iloc[1]["color!=red"]


def test_handle_unknown_does_not_hide_errors_in_explicit_scales():
    train = pd.DataFrame({"size": ["S", "L"]})
    scaler = ConceptualScaler([OrdinalScale("size", levels=["S", "L"])], handle_unknown="ignore")
    with pytest.raises(ValueError):
        scaler.fit(train).transform(pd.DataFrame({"size": ["XL"]}))


def test_handle_unknown_is_validated():
    with pytest.raises(ValueError, match="handle_unknown"):
        ConceptualScaler([NominalScale("a")], handle_unknown="skip").fit(pd.DataFrame({"a": ["x"]}))


def test_cross_validation_with_rare_values():
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import make_pipeline

    rng = np.random.default_rng(0)
    color = rng.choice(["red", "blue", "green"], 60)
    color[0] = "purple"  # appears once: missing from four of five training folds
    X = pd.DataFrame({"color": color})
    y = (color == "red").astype(int)
    pipe = make_pipeline(
        ConceptualScaler([NominalScale("color")], handle_unknown="ignore"),
        LogisticRegression(),
    )
    scores = cross_val_score(pipe, X, y, cv=5, error_score="raise")
    assert scores.mean() > 0.9
