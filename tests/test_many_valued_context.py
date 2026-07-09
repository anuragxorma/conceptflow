import numpy as np
import pandas as pd
import pytest

from conceptflow import ManyValuedContext


def test_many_valued_context_from_dataframe():
    df = pd.DataFrame(
        {
            "age": [21, 35],
            "city": ["Berlin", "Paris"],
        },
        index=["g1", "g2"],
    )

    ctx = ManyValuedContext.from_dataframe(df)

    assert ctx.objects == ("g1", "g2")
    assert ctx.attributes == ("age", "city")
    assert ctx.n_objects == 2
    assert ctx.n_attributes == 2
    assert ctx.size == (2, 2)


def test_many_valued_context_from_array():
    data = np.array([
        [1.5, 2.0],
        [3.0, 4.5],
    ])

    ctx = ManyValuedContext.from_array(
        data,
        objects=["g1", "g2"],
        attributes=["x", "y"],
    )

    assert ctx.objects == ("g1", "g2")
    assert ctx.attributes == ("x", "y")
    assert ctx.data.loc["g1", "x"] == 1.5


def test_many_valued_context_from_array_default_names():
    data = [[1, 2], [3, 4]]

    ctx = ManyValuedContext.from_array(data)

    assert ctx.objects == ("g0", "g1")
    assert ctx.attributes == ("m0", "m1")


def test_many_valued_context_rejects_1d_input():
    with pytest.raises(ValueError):
        ManyValuedContext.from_array([1, 2, 3])


def test_many_valued_context_rejects_empty_dataframe():
    with pytest.raises(ValueError):
        ManyValuedContext.from_dataframe(pd.DataFrame())


def test_to_dataframe_returns_copy():
    df = pd.DataFrame({"a": [1, 2]})
    ctx = ManyValuedContext.from_dataframe(df)

    out = ctx.to_dataframe()
    out.loc["0", "a"] = 999

    assert ctx.data.loc["0", "a"] == 1


def test_column_access():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    ctx = ManyValuedContext.from_dataframe(df)

    assert ctx.column("a").tolist() == [1, 2]


def test_get_value():
    df = pd.DataFrame(
        {
            "age": [21, 35],
            "city": ["Berlin", "Paris"],
        },
        index=["g1", "g2"],
    )

    ctx = ManyValuedContext.from_dataframe(df)

    assert ctx.get_value("g1", "age") == 21
    assert ctx.get_value("g2", "city") == "Paris"


def test_row():
    df = pd.DataFrame(
        {
            "age": [21, 35],
            "city": ["Berlin", "Paris"],
        },
        index=["g1", "g2"],
    )

    ctx = ManyValuedContext.from_dataframe(df)

    row = ctx.row("g1")

    assert row["age"] == 21
    assert row["city"] == "Berlin"


def test_unique_values():
    df = pd.DataFrame(
        {
            "color": ["red", "blue", "red"],
        },
        index=["g1", "g2", "g3"],
    )

    ctx = ManyValuedContext.from_dataframe(df)

    assert ctx.unique_values("color") == {"red", "blue"}


def test_restrict_objects():
    df = pd.DataFrame(
        {
            "age": [21, 35, 40],
        },
        index=["g1", "g2", "g3"],
    )

    ctx = ManyValuedContext.from_dataframe(df)

    restricted = ctx.restrict_objects(["g1", "g3"])

    assert restricted.objects == ("g1", "g3")
    assert restricted.n_objects == 2
    assert restricted.get_value("g3", "age") == 40


def test_restrict_objects_rejects_unknown_objects():
    df = pd.DataFrame(
        {
            "age": [21, 35],
        },
        index=["g1", "g2"],
    )

    ctx = ManyValuedContext.from_dataframe(df)

    with pytest.raises(KeyError):
        ctx.restrict_objects(["g1", "g3"])


def test_duplicate_objects_raise_error():
    with pytest.raises(ValueError):
        ManyValuedContext(
            objects=("g1", "g1"),
            attributes=("a",),
            data=[[1], [2]],
        )


def test_duplicate_attributes_raise_error():
    with pytest.raises(ValueError):
        ManyValuedContext(
            objects=("g1", "g2"),
            attributes=("a", "a"),
            data=[[1, 2], [3, 4]],
        )


def test_unknown_object_raises_keyerror():
    df = pd.DataFrame({"a": [1, 2]}, index=["g1", "g2"])
    ctx = ManyValuedContext.from_dataframe(df)

    with pytest.raises(KeyError):
        ctx.get_value("g3", "a")


def test_unknown_attribute_raises_keyerror():
    df = pd.DataFrame({"a": [1, 2]}, index=["g1", "g2"])
    ctx = ManyValuedContext.from_dataframe(df)

    with pytest.raises(KeyError):
        ctx.get_value("g1", "b")