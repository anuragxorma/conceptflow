import numpy as np
import pandas as pd
import pytest

from conceptflow import FormalContext, IncompleteContext


def make_context():
    #       a     b     c
    # g1  yes   yes   no
    # g2  yes   ?     yes
    # g3  ?     no    yes
    return IncompleteContext.from_dataframe(
        pd.DataFrame(
            [[True, True, False], [1, None, 1], [np.nan, 0, 1]],
            index=["g1", "g2", "g3"],
            columns=["a", "b", "c"],
        )
    )


def test_parsing_and_completions():
    ctx = make_context()

    assert ctx.n_unknown == 2
    assert not ctx.is_complete
    assert ctx.certain_context().incidence.tolist() == [
        [True, True, False],
        [True, False, True],
        [False, False, True],
    ]
    assert ctx.possible_context().incidence.tolist() == [
        [True, True, False],
        [True, True, True],
        [True, False, True],
    ]


def test_question_mark_marks_unknown():
    ctx = IncompleteContext.from_array([[1, "?"], [0, 1]])
    assert ctx.unknown.tolist() == [[False, True], [False, False]]


@pytest.mark.parametrize("bad", ["no", "yes", 2, 0.5])
def test_ambiguous_values_are_rejected_not_coerced(bad):
    # bool("no") is True -- exactly the silent coercion this guards against.
    with pytest.raises(ValueError, match="three-valued"):
        IncompleteContext.from_array([[bad]])


def test_certain_and_possible_derivations():
    ctx = make_context()

    assert ctx.certain_extent(["a"]) == {0, 1}
    assert ctx.possible_extent(["a"]) == {0, 1, 2}
    assert ctx.certain_intent(["g2"]) == {0, 2}
    assert ctx.possible_intent(["g2"]) == {0, 1, 2}


def test_implication_status():
    ctx = make_context()

    # Only g1 and g2 can have b; both certainly have a.
    assert ctx.implication_status(["b"], ["a"]) == "valid"
    # g1 certainly has a and certainly lacks c.
    assert ctx.implication_status(["a"], ["c"]) == "refuted"
    # g3 might have a and certainly lacks b, g2 might lack b.
    assert ctx.implication_status(["c"], ["b"]) == "refuted"
    assert ctx.implication_status(["a", "c"], ["b"]) == "possible"

    certain, possible = ctx.implication_counterexamples(["a", "c"], ["b"])
    assert certain == frozenset()
    assert possible == {1, 2}


def test_implication_status_agrees_with_all_completions():
    rng = np.random.default_rng(5)

    for _ in range(40):
        states = rng.integers(0, 3, size=(4, 3))  # 0 no, 1 yes, 2 unknown
        table = np.where(states == 2, None, states).tolist()
        ctx = IncompleteContext.from_array(table)
        unknown_cells = list(zip(*np.nonzero(ctx.unknown)))

        for premise, conclusion in [([0], [1]), ([0, 1], [2]), ([], [0])]:
            holds = []
            for bits in range(2 ** len(unknown_cells)):
                incidence = ctx.certain.copy()
                for k, (g, m) in enumerate(unknown_cells):
                    incidence[g, m] = bool(bits >> k & 1)
                completion = FormalContext.from_array(incidence)
                extent = completion.attribute_derivation(premise)
                holds.append(extent <= completion.attribute_derivation(conclusion))

            expected = (
                "valid" if all(holds) else "possible" if any(holds) else "refuted"
            )
            assert ctx.implication_status(premise, conclusion) == expected


def test_round_trip_and_from_formal_context():
    ctx = make_context()
    frame = ctx.to_dataframe()

    assert frame.isna().sum().sum() == 2
    assert IncompleteContext.from_dataframe(frame) == ctx

    complete = FormalContext.from_array([[1, 0]])
    wrapped = IncompleteContext.from_formal_context(complete)
    assert wrapped.is_complete
    assert wrapped.certain_context() == complete


def test_validation():
    with pytest.raises(ValueError, match="both certain and unknown"):
        IncompleteContext(("g",), ("m",), [[True]], [[True]])
    with pytest.raises(ValueError, match="shape"):
        IncompleteContext(("g",), ("m",), [[True, False]], [[False]])
