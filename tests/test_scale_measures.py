"""Scale-measures (Hanika & Hirth 2022) on small hand-checkable contexts."""

import numpy as np
import pytest

from conceptflow import ConceptLattice, FormalContext
from conceptflow.core import is_scale_measure, refines, scale_measure, scale_measure_problems
from conceptflow.core.scale_measures import CONJUNCTION


def make_context():
    # a b c d
    return FormalContext.from_array(
        np.array([[1, 1, 0, 0], [1, 0, 1, 0], [0, 1, 1, 1], [1, 1, 1, 0]]),
        objects=["g1", "g2", "g3", "g4"],
        attributes=["a", "b", "c", "d"],
    )


def test_projection_keeps_columns_and_objects():
    context = make_context()
    scale = scale_measure(context, ["c", "a"])
    assert scale.objects == context.objects
    assert scale.attributes == ("a", "c")  # canonical order: by position in the context
    assert np.array_equal(scale.incidence, context.incidence[:, [0, 2]])
    assert is_scale_measure(context, scale)


def test_conjunction_is_the_and_of_its_parts():
    context = make_context()
    scale = scale_measure(context, [("b", "c"), "d", 0])
    assert scale.attributes == ("a", f"b{CONJUNCTION}c", "d")
    assert scale.incidence[:, 1].tolist() == [False, False, True, True]
    assert is_scale_measure(context, scale) and scale_measure_problems(context, scale) == []


def test_canonical_order_is_by_sorted_positions_of_the_parts():
    context = make_context()
    scale = scale_measure(context, ["d", ("c", "a"), "b", ("b", "d")])
    # Sorted position tuples: (0, 2) "c ∧ a", (1,) "b", (1, 3) "b ∧ d", (3,) "d".
    assert scale.attributes == (f"c{CONJUNCTION}a", "b", f"b{CONJUNCTION}d", "d")


def test_an_attribute_that_is_not_an_extent_fails():
    context = make_context()
    # {g1, g2} is not an extent: its closure is {g1, g2, g4}, the extent of a.
    view = FormalContext(
        objects=context.objects,
        attributes=("a", "g1 or g2"),
        incidence=np.array([[1, 1], [1, 1], [0, 0], [1, 0]], dtype=bool),
    )
    problems = scale_measure_problems(context, view)
    assert not is_scale_measure(context, view)
    assert problems == [
        "attribute 'g1 or g2': its extent has 2 objects but closes to 3 in the original "
        "context, so it is not an extent there"
    ]


def test_other_objects_are_rejected():
    context = make_context()
    reordered = FormalContext(
        objects=context.objects[::-1], attributes=context.attributes,
        incidence=context.incidence[::-1],
    )
    assert scale_measure_problems(context, reordered) == [
        "object sets differ, so the identity is not a map between them"
    ]


@pytest.mark.parametrize(
    "attributes, error",
    [(["x"], KeyError), ([()], ValueError), (["a", "a"], ValueError)],
)
def test_invalid_attributes(attributes, error):
    with pytest.raises(error):
        scale_measure(make_context(), attributes)


def test_none_returns_the_context():
    context = make_context()
    assert scale_measure(context, None) is context


def test_refinement_pairs():
    context = make_context()
    coarse = scale_measure(context, ["a"])
    finer = scale_measure(context, ["a", ("b", "c")])
    assert refines(coarse, finer) and refines(finer, context) and refines(coarse, context)
    assert not refines(context, coarse)
    assert refines(ConceptLattice.from_context(coarse), ConceptLattice.from_context(context))


def test_refines_needs_the_same_objects_and_full_lattices():
    context = make_context()
    other = FormalContext.from_array(np.eye(2, dtype=int), objects=["x", "y"])
    with pytest.raises(ValueError, match="same objects"):
        refines(context, other)
    with pytest.raises(ValueError, match="iceberg"):
        refines(ConceptLattice.from_context(context, min_support=2), context)
