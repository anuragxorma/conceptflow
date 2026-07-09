import numpy as np
import pandas as pd
import pytest

from conceptflow import FormalContext


def test_formal_context_from_array():
    data = np.array([
        [1, 1, 0],
        [1, 0, 1],
        [0, 1, 1],
    ])

    ctx = FormalContext.from_array(
        data,
        objects=["g1", "g2", "g3"],
        attributes=["m1", "m2", "m3"],
    )

    assert ctx.n_objects == 3
    assert ctx.n_attributes == 3
    assert ctx.objects == ("g1", "g2", "g3")
    assert ctx.attributes == ("m1", "m2", "m3")
    assert ctx.incidence.dtype == bool


def test_formal_context_from_dataframe():
    df = pd.DataFrame(
        [
            [1, 0],
            [1, 1],
        ],
        index=["g1", "g2"],
        columns=["m1", "m2"],
    )

    ctx = FormalContext.from_dataframe(df)

    assert ctx.objects == ("g1", "g2")
    assert ctx.attributes == ("m1", "m2")
    assert ctx.n_objects == 2
    assert ctx.n_attributes == 2


def test_invalid_shape_raises_error():
    with pytest.raises(ValueError):
        FormalContext(
            objects=("g1", "g2"),
            attributes=("m1", "m2"),
            incidence=np.array([[1, 0, 1]]),
        )


def test_object_derivation():
    data = np.array([
        [1, 1, 0],
        [1, 0, 1],
        [1, 1, 1],
    ])

    ctx = FormalContext.from_array(data)

    assert ctx.object_derivation([0, 2]) == frozenset({0, 1})
    assert ctx.object_derivation([]) == frozenset({0, 1, 2})


def test_attribute_derivation():
    data = np.array([
        [1, 1, 0],
        [1, 0, 1],
        [1, 1, 1],
    ])

    ctx = FormalContext.from_array(data)

    assert ctx.attribute_derivation([0, 1]) == frozenset({0, 2})
    assert ctx.attribute_derivation([]) == frozenset({0, 1, 2})


def test_closure():
    data = np.array([
        [1, 1, 0],
        [1, 0, 1],
        [1, 1, 1],
    ])

    ctx = FormalContext.from_array(data)

    assert ctx.closure([1]) == frozenset({0, 1})