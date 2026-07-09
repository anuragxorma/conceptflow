import numpy as np

from conceptflow import FormalContext
from conceptflow.algorithms import clarified_context, clarify_context


def test_clarify_context_removes_duplicate_objects():
    ctx = FormalContext.from_array(
        np.array([
            [1, 0],
            [1, 0],
            [0, 1],
        ]),
        objects=["g1", "g2", "g3"],
        attributes=["m1", "m2"],
    )

    result = clarify_context(ctx)

    assert result.context.objects == ("g1", "g3")
    assert result.context.attributes == ("m1", "m2")
    assert result.context.incidence.tolist() == [
        [True, False],
        [False, True],
    ]
    assert result.object_groups == {
        "g1": ("g1", "g2"),
        "g3": ("g3",),
    }


def test_clarify_context_removes_duplicate_attributes():
    ctx = FormalContext.from_array(
        np.array([
            [1, 1, 0],
            [0, 0, 1],
        ]),
        objects=["g1", "g2"],
        attributes=["m1", "m2", "m3"],
    )

    result = clarify_context(ctx)

    assert result.context.objects == ("g1", "g2")
    assert result.context.attributes == ("m1", "m3")
    assert result.context.incidence.tolist() == [
        [True, False],
        [False, True],
    ]
    assert result.attribute_groups == {
        "m1": ("m1", "m2"),
        "m3": ("m3",),
    }


def test_clarify_context_removes_duplicate_objects_and_attributes():
    ctx = FormalContext.from_array(
        np.array([
            [1, 1, 0],
            [1, 1, 0],
            [0, 0, 1],
        ]),
        objects=["g1", "g2", "g3"],
        attributes=["m1", "m2", "m3"],
    )

    result = clarify_context(ctx)

    assert result.context.objects == ("g1", "g3")
    assert result.context.attributes == ("m1", "m3")
    assert result.context.incidence.tolist() == [
        [True, False],
        [False, True],
    ]


def test_clarified_context_returns_only_context():
    ctx = FormalContext.from_array(
        np.array([
            [1, 0],
            [1, 0],
        ]),
        objects=["g1", "g2"],
        attributes=["m1", "m2"],
    )

    clarified = clarified_context(ctx)

    assert isinstance(clarified, FormalContext)
    assert clarified.objects == ("g1",)
    assert clarified.attributes == ("m1", "m2")