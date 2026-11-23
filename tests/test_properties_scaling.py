"""
Property-based tests of conceptual scaling.

Hypothesis draws many-valued tables with missing values (None, NaN,
``pd.NA``, ``"?"``), and each scale is checked against the relation it
stands for, written out here from the scale's documentation: ``color=v``
iff the value is v, ``grade>=k`` iff the value is at least level k, and so
on. Also checked: the three ``missing`` modes, the four outputs agreeing
with each other and with ``get_feature_names_out``, ``handle_unknown`` and
pickling.
"""

import pickle

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("hypothesis")

from hypothesis import given  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

from conceptflow import FormalContext, IncompleteContext  # noqa: E402
from conceptflow.preprocessing import (  # noqa: E402
    ConceptualScaler,
    ContranominalScale,
    DichotomicScale,
    GeneralScale,
    InterordinalScale,
    NominalScale,
    OrdinalScale,
    ThresholdScale,
)

LEVELS = ["low", "mid", "high"]
THRESHOLDS = [33, 66]
MISSING = [None, np.nan, pd.NA, "?"]
GENERAL = {"red": {"warm", "bright"}, "blue": {"cold"}, "green": {"cold", "bright"}}


def is_missing(value):
    return value is None or value is pd.NA or value == "?" or (
        isinstance(value, float) and np.isnan(value)
    )


@st.composite
def tables(draw, missing=True):
    n = draw(st.integers(1, 8), label="objects")

    def column(values):
        options = st.sampled_from(values + MISSING) if missing else st.sampled_from(values)
        return draw(st.lists(options, min_size=n, max_size=n))

    return pd.DataFrame(
        {
            "color": column(["red", "blue", "green"]),
            "grade": column(LEVELS),
            "size": column([0.0, 10.0, 33.0, 50.0, 66.0, 99.5]),
            "flag": column([True, False]),
        },
        index=[f"o{i}" for i in range(n)],
    )


def relations(table):
    """Each scale with its attributes and their relation to a value."""
    colors = sorted({v for v in table["color"] if not is_missing(v)}, key=str)
    return {
        "nominal": (
            NominalScale("color"),
            {f"color={c}": (lambda c: lambda v: v == c)(c) for c in colors},
            colors,
        ),
        "contranominal": (
            ContranominalScale("color"),
            {f"color!={c}": (lambda c: lambda v: v != c)(c) for c in colors},
            colors,
        ),
        "ordinal ge": (
            OrdinalScale("grade", LEVELS, mode="ge"),
            {f"grade>={k}": (lambda k: lambda v: LEVELS.index(v) >= LEVELS.index(k))(k)
             for k in LEVELS},
            LEVELS,
        ),
        "ordinal le": (
            OrdinalScale("grade", LEVELS, mode="le"),
            {f"grade<={k}": (lambda k: lambda v: LEVELS.index(v) <= LEVELS.index(k))(k)
             for k in LEVELS},
            LEVELS,
        ),
        "ordinal exact": (
            OrdinalScale("grade", LEVELS, mode="exact"),
            {f"grade={k}": (lambda k: lambda v: v == k)(k) for k in LEVELS},
            LEVELS,
        ),
        "interordinal": (
            InterordinalScale("grade", LEVELS),
            {**{f"grade<={k}": (lambda k: lambda v: LEVELS.index(v) <= LEVELS.index(k))(k)
                for k in LEVELS},
             **{f"grade>={k}": (lambda k: lambda v: LEVELS.index(v) >= LEVELS.index(k))(k)
                for k in LEVELS}},
            LEVELS,
        ),
        "threshold": (
            ThresholdScale("size", THRESHOLDS),
            {f"size>={t}": (lambda t: lambda v: v >= t)(t) for t in THRESHOLDS},
            None,  # open domain
        ),
        "dichotomic": (
            DichotomicScale("flag"),
            {"flag=True": lambda v: v is True or v == True,  # noqa: E712
             "flag=False": lambda v: v is False or v == False},  # noqa: E712
            [True, False],
        ),
        "general": (
            GeneralScale("color", GENERAL, attribute_order=["warm", "cold", "bright"]),
            {name: (lambda name: lambda v: name in GENERAL[v])(name)
             for name in ["warm", "cold", "bright"]},
            list(GENERAL),
        ),
    }


def source_column(scale):
    return scale.source_attribute


SCALE_NAMES = list(relations(pd.DataFrame({"color": ["red"]})))


@given(tables(missing=False), st.sampled_from(SCALE_NAMES))
def test_every_scale_is_its_relation(table, scale_name):
    scale, attributes, _ = relations(table)[scale_name]
    scaler = ConceptualScaler(scales=[scale], output="dataframe")
    result = scaler.fit_transform(table)
    assert list(result.columns) == list(attributes)
    assert list(scaler.get_feature_names_out()) == list(attributes)
    values = table[source_column(scale)]
    for name, holds in attributes.items():
        assert result[name].tolist() == [bool(holds(v)) for v in values], name


@given(tables(), st.sampled_from(SCALE_NAMES), st.sampled_from(["error", "false", "unknown"]))
def test_missing_values_are_scaled_as_documented(table, scale_name, missing):
    scale, attributes, domain = relations(table)[scale_name]
    values = table[source_column(scale)]
    has_missing = any(is_missing(v) for v in values)
    scaler = ConceptualScaler(scales=[scale], output="dataframe", missing=missing)
    if missing == "error" and has_missing:
        with pytest.raises(ValueError, match="missing"):
            scaler.fit_transform(table)
        return
    if scale_name in ("nominal", "contranominal") and not domain:
        return  # no values at all: nothing to scale (checked elsewhere)
    result = scaler.fit_transform(table)
    assert list(result.columns) == list(attributes)
    for name, holds in attributes.items():
        for value, got in zip(values, result[name]):
            if not is_missing(value):
                assert got == bool(holds(value)), (name, value)
            elif missing == "false":
                assert got == False  # noqa: E712
            elif domain is None:
                assert np.isnan(got), (name, value)
            else:
                possible = {bool(holds(v)) for v in domain}
                expected = possible.pop() if len(possible) == 1 else np.nan
                if np.isnan(expected):
                    assert np.isnan(got), (name, got)
                else:
                    assert got == expected, (name, got)


@given(tables(), st.sampled_from(["false", "unknown"]))
def test_the_outputs_agree(table, missing):
    scales = [ContranominalScale("color"), OrdinalScale("grade", LEVELS),
              ThresholdScale("size", THRESHOLDS), DichotomicScale("flag")]
    if all(is_missing(v) for v in table["color"]):
        scales = scales[1:]
    frame = ConceptualScaler(scales=scales, output="dataframe", missing=missing).fit_transform(
        table
    )
    array = ConceptualScaler(scales=scales, output="array", missing=missing).fit_transform(table)
    context = ConceptualScaler(scales=scales, output="context", missing=missing).fit_transform(
        table
    )
    names = ConceptualScaler(scales=scales, missing=missing).fit(table).get_feature_names_out()
    assert list(frame.columns) == list(names)
    assert list(frame.index) == list(table.index)
    np.testing.assert_array_equal(np.asarray(frame, dtype=float), np.asarray(array, dtype=float))
    if missing == "false":
        sparse = ConceptualScaler(scales=scales, output="sparse", missing=missing).fit_transform(
            table
        )
        np.testing.assert_array_equal(sparse.toarray(), array)
        assert isinstance(context, FormalContext)
        np.testing.assert_array_equal(context.incidence, array)
    else:
        assert isinstance(context, IncompleteContext)
        unknown = np.isnan(np.asarray(array, dtype=float))
        np.testing.assert_array_equal(context.unknown, unknown)
        np.testing.assert_array_equal(context.certain, np.nan_to_num(array) == 1)
        with pytest.raises(ValueError):
            ConceptualScaler(scales=scales, output="sparse", missing="unknown").fit_transform(
                table
            )
    assert tuple(context.attributes) == tuple(names)
    assert tuple(context.objects) == tuple(table.index)


@given(tables(missing=False), tables(missing=False))
def test_unseen_values_are_ignored_or_refused(train, test):
    seen = set(train["color"])
    unseen = [v for v in test["color"] if v not in seen]
    for scale in (NominalScale("color"), ContranominalScale("color")):
        strict = ConceptualScaler(scales=[scale], output="dataframe").fit(train)
        lenient = ConceptualScaler(
            scales=[scale], output="dataframe", handle_unknown="ignore"
        ).fit(train)
        if unseen:
            with pytest.raises(ValueError, match="vocabulary"):
                strict.transform(test)
        else:
            assert strict.transform(test).equals(lenient.transform(test))
        result = lenient.transform(test)
        assert list(result.columns) == list(lenient.get_feature_names_out())
        for value, row in zip(test["color"], result.itertuples(index=False)):
            if value in seen:
                continue
            # An unseen value equals none of the known values.
            expected = isinstance(scale, ContranominalScale)
            assert all(cell == expected for cell in row), (value, row)


@given(tables(missing=False))
def test_fitted_scalers_survive_pickling(table):
    scaler = ConceptualScaler(
        scales=[NominalScale("color"), InterordinalScale("grade", LEVELS)], output="array"
    ).fit(table)
    copied = pickle.loads(pickle.dumps(scaler))
    np.testing.assert_array_equal(copied.transform(table), scaler.transform(table))
    assert list(copied.get_feature_names_out()) == list(scaler.get_feature_names_out())
