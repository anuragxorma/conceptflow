import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from scipy import sparse

from conceptflow import FormalContext, ManyValuedContext
from conceptflow.cluster import ConceptLatticeEstimator
from conceptflow.preprocessing import (
    ConceptualScaler,
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