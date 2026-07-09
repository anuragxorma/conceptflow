import pandas as pd
import pytest

from conceptflow import ManyValuedContext
from conceptflow.preprocessing import (
    ContranominalScale,
    DichotomicScale,
    GeneralScale,
    InterordinalScale,
    NominalScale,
    OrdinalScale,
    ThresholdScale,
)


def make_context():
    return ManyValuedContext.from_dataframe(
        pd.DataFrame(
            {
                "color": ["red", "blue", "red"],
                "risk": ["low", "medium", "high"],
                "score": [20, 45, 80],
                "flag": ["yes", "no", "yes"],
            },
            index=["g1", "g2", "g3"],
        )
    )


def test_nominal_scale_binary_attributes():
    mvc = make_context()
    scale = NominalScale("color")

    assert scale.binary_attributes(mvc) == ["color=blue", "color=red"]


def test_nominal_scale_encode_value():
    mvc = make_context()
    scale = NominalScale("color")

    assert scale.encode_value("red", mvc) == {
        "color=blue": False,
        "color=red": True,
    }


def test_contranominal_scale_encode_value():
    mvc = make_context()
    scale = ContranominalScale("color")

    assert scale.binary_attributes(mvc) == ["color!=blue", "color!=red"]
    assert scale.encode_value("red", mvc) == {
        "color!=blue": True,
        "color!=red": False,
    }


def test_dichotomic_scale_encode_value():
    mvc = make_context()
    scale = DichotomicScale("flag", true_value="yes", false_value="no")

    assert scale.binary_attributes(mvc) == ["flag=yes", "flag=no"]
    assert scale.encode_value("yes", mvc) == {
        "flag=yes": True,
        "flag=no": False,
    }


def test_dichotomic_scale_rejects_invalid_value():
    mvc = make_context()
    scale = DichotomicScale("flag", true_value="yes", false_value="no")

    with pytest.raises(ValueError):
        scale.encode_value("maybe", mvc)


def test_ordinal_scale_ge():
    mvc = make_context()
    scale = OrdinalScale(
        "risk",
        levels=["low", "medium", "high"],
        mode="ge",
    )

    assert scale.binary_attributes(mvc) == [
        "risk>=low",
        "risk>=medium",
        "risk>=high",
    ]
    assert scale.encode_value("medium", mvc) == {
        "risk>=low": True,
        "risk>=medium": True,
        "risk>=high": False,
    }


def test_ordinal_scale_le():
    mvc = make_context()
    scale = OrdinalScale(
        "risk",
        levels=["low", "medium", "high"],
        mode="le",
    )

    assert scale.encode_value("medium", mvc) == {
        "risk<=low": False,
        "risk<=medium": True,
        "risk<=high": True,
    }


def test_ordinal_scale_exact():
    mvc = make_context()
    scale = OrdinalScale(
        "risk",
        levels=["low", "medium", "high"],
        mode="exact",
    )

    assert scale.encode_value("medium", mvc) == {
        "risk=low": False,
        "risk=medium": True,
        "risk=high": False,
    }


def test_ordinal_scale_rejects_duplicate_levels():
    with pytest.raises(ValueError):
        OrdinalScale("risk", levels=["low", "low"])


def test_ordinal_scale_rejects_unknown_value():
    mvc = make_context()
    scale = OrdinalScale("risk", levels=["low", "medium", "high"])

    with pytest.raises(ValueError):
        scale.encode_value("extreme", mvc)


def test_threshold_scale_binary_attributes():
    mvc = make_context()
    scale = ThresholdScale("score", thresholds=[60, 30])

    assert scale.binary_attributes(mvc) == ["score>=30", "score>=60"]


def test_threshold_scale_encode_value():
    mvc = make_context()
    scale = ThresholdScale("score", thresholds=[30, 60])

    assert scale.encode_value(45, mvc) == {
        "score>=30": True,
        "score>=60": False,
    }


def test_threshold_scale_rejects_duplicate_thresholds():
    with pytest.raises(ValueError):
        ThresholdScale("score", thresholds=[30, 30])


def test_interordinal_scale_binary_attributes():
    mvc = make_context()
    scale = InterordinalScale("risk", levels=["low", "medium", "high"])

    assert scale.binary_attributes(mvc) == [
        "risk<=low",
        "risk<=medium",
        "risk<=high",
        "risk>=low",
        "risk>=medium",
        "risk>=high",
    ]


def test_interordinal_scale_encode_value():
    mvc = make_context()
    scale = InterordinalScale("risk", levels=["low", "medium", "high"])

    assert scale.encode_value("medium", mvc) == {
        "risk<=low": False,
        "risk>=low": True,
        "risk<=medium": True,
        "risk>=medium": True,
        "risk<=high": True,
        "risk>=high": False,
    }


def test_general_scale():
    mvc = make_context()
    scale = GeneralScale(
        "risk",
        mapping={
            "low": ["acceptable"],
            "medium": ["acceptable", "needs_review"],
            "high": ["needs_review", "dangerous"],
        },
        attribute_order=["acceptable", "needs_review", "dangerous"],
    )

    assert scale.binary_attributes(mvc) == [
        "acceptable",
        "needs_review",
        "dangerous",
    ]
    assert scale.encode_value("medium", mvc) == {
        "acceptable": True,
        "needs_review": True,
        "dangerous": False,
    }


def test_general_scale_rejects_unknown_value():
    mvc = make_context()
    scale = GeneralScale(
        "risk",
        mapping={"low": ["acceptable"]},
    )

    with pytest.raises(ValueError):
        scale.encode_value("medium", mvc)