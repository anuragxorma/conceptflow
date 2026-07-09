import numpy as np

from conceptflow import FormalContext
from conceptflow.algorithms.derivation import (
    attribute_closure,
    attribute_derivation,
    object_derivation,
)


def test_object_derivation_algorithm():
    data = np.array([
        [1, 1, 0],
        [1, 0, 1],
        [1, 1, 1],
    ])

    ctx = FormalContext.from_array(data)

    assert object_derivation(ctx, [0, 2]) == frozenset({0, 1})


def test_attribute_derivation_algorithm():
    data = np.array([
        [1, 1, 0],
        [1, 0, 1],
        [1, 1, 1],
    ])

    ctx = FormalContext.from_array(data)

    assert attribute_derivation(ctx, [0, 1]) == frozenset({0, 2})


def test_attribute_closure_algorithm():
    data = np.array([
        [1, 1, 0],
        [1, 0, 1],
        [1, 1, 1],
    ])

    ctx = FormalContext.from_array(data)

    assert attribute_closure(ctx, [1]) == frozenset({0, 1})