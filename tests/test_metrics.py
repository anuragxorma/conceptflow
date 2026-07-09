import numpy as np
import pytest

from conceptflow import Concept, FormalContext
from conceptflow.metrics import (
    attribute_set_support,
    concept_support,
    implication_confidence,
)


def make_context():
    return FormalContext.from_array(
        np.array([
            [1, 1, 0],
            [1, 1, 1],
            [1, 0, 1],
            [0, 1, 1],
        ]),
        objects=["g1", "g2", "g3", "g4"],
        attributes=["a", "b", "c"],
    )


def test_concept_support_normalized():
    ctx = make_context()
    concept = Concept(extent={0, 1}, intent={0, 1})

    assert concept_support(concept, ctx) == 0.5


def test_concept_support_absolute():
    ctx = make_context()
    concept = Concept(extent={0, 1}, intent={0, 1})

    assert concept_support(concept, ctx, normalize=False) == 2


def test_attribute_set_support_normalized():
    ctx = make_context()

    assert attribute_set_support(ctx, [0, 1]) == 0.5


def test_attribute_set_support_absolute():
    ctx = make_context()

    assert attribute_set_support(ctx, [0, 1], normalize=False) == 2


def test_implication_confidence():
    ctx = make_context()

    # a -> b
    # objects with a: g1, g2, g3 = 3
    # objects with a and b: g1, g2 = 2
    assert implication_confidence(ctx, premise=[0], conclusion=[1]) == pytest.approx(2 / 3)


def test_implication_confidence_exact_implication():
    ctx = make_context()

    # a,b -> c
    # objects with a and b: g1, g2
    # objects with a,b,c: g2
    assert implication_confidence(ctx, premise=[0, 1], conclusion=[2]) == 0.5


def test_implication_confidence_rejects_zero_premise_support():
    ctx = make_context()

    # No object has all of a, b, and c absent? Actually here a,b,c together
    # occurs for g2, so use a deliberately impossible valid combination
    # by creating a small context below.
    empty_ctx = FormalContext.from_array(
        np.array([
            [1, 0],
            [0, 1],
        ]),
        attributes=["a", "b"],
    )

    with pytest.raises(ValueError):
        implication_confidence(
            empty_ctx,
            premise=[0, 1],
            conclusion=[0],
        )