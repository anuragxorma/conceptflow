import math

import numpy as np
import pandas as pd
import pytest

from conceptflow import ConceptLattice, FormalContext
from conceptflow.algorithms import compute_canonical_basis
from conceptflow.quality import (
    check_implications,
    explain_object,
    explain_rule,
    redundancy_report,
)
from conceptflow.rules import ImplicationOutlierDetector


def animals():
    #            mammal warm  fly   swim
    data = [[1, 1, 0, 0]] * 8 + [[1, 1, 0, 1], [0, 0, 0, 1], [0, 1, 1, 0], [1, 0, 0, 0]]
    names = [f"m{i}" for i in range(9)] + ["fish", "bird", "odd"]
    return FormalContext.from_array(
        data, objects=names, attributes=["mammal", "warm", "fly", "swim"]
    )


# ---------------------------------------------------------------------------
# check_implications / explain_rule
# ---------------------------------------------------------------------------


def test_check_implications_lists_violations():
    ctx = animals()
    report = check_implications(ctx, [(["mammal"], ["warm"]), (["fly"], ["warm"])])

    assert list(report.columns) == ["rule", "premise", "conclusion", "object", "missing"]
    assert report["object"].tolist() == ["odd"]
    assert report.loc[0, "missing"] == ("warm",)
    assert report.loc[0, "rule"] == "{mammal} -> {warm}"


def test_check_implications_accepts_implication_objects_and_indices():
    ctx = animals()
    basis = compute_canonical_basis(ctx)

    # The canonical basis holds in its own context.
    assert check_implications(ctx, basis).empty
    assert check_implications(ctx, [([0], [1])])["object"].tolist() == ["odd"]

    with pytest.raises(TypeError, match="collections"):
        check_implications(ctx, [("mammal", "warm")])


def test_explain_rule():
    explanation = explain_rule(animals(), (["mammal"], ["warm"]))

    assert (explanation.support, explanation.premise_support) == (9, 10)
    assert math.isclose(explanation.confidence, 0.9)
    assert explanation.exceptions == {"odd": ("warm",)}
    assert "odd" in str(explanation)


# ---------------------------------------------------------------------------
# redundancy_report
# ---------------------------------------------------------------------------


def test_redundancy_report_finds_known_structure():
    ctx = FormalContext.from_array(
        [
            [1, 1, 1, 0, 1, 0],
            [1, 1, 0, 0, 1, 0],
            [1, 1, 0, 0, 1, 0],
            [0, 1, 1, 0, 1, 0],
            [0, 0, 0, 0, 0, 0],
        ],
        objects=["p", "q", "q2", "r", "nothing"],
        attributes=["a", "b", "c", "never", "b_copy", "zz"],
    )
    report = redundancy_report(ctx)

    assert ("q", "q2") in report.duplicate_objects
    assert ("b", "b_copy") in report.duplicate_attributes
    assert set(report.empty_attributes) == {"never", "zz"}
    assert report.empty_objects == ("nothing",)
    assert not report.is_clean
    assert {"kind", "name", "detail"} == set(report.to_frame().columns)


@pytest.mark.parametrize("seed", range(20))
def test_derivable_columns_really_are_intersections(seed):
    rng = np.random.default_rng(seed)
    ctx = FormalContext.from_array(rng.random((10, 6)) < 0.6)
    report = redundancy_report(ctx)
    column = {name: ctx.incidence[:, j] for j, name in enumerate(ctx.attributes)}

    for name, parts in report.derivable_attributes.items():
        assert parts and name not in parts
        assert np.array_equal(np.logical_and.reduce([column[p] for p in parts]), column[name])


def test_clean_context():
    ctx = FormalContext.from_array(np.eye(3, dtype=bool) == 0)  # contranominal scale
    assert redundancy_report(ctx).is_clean


# ---------------------------------------------------------------------------
# explain_object
# ---------------------------------------------------------------------------


def test_explain_object():
    lattice = ConceptLattice.from_context(animals())
    explanation = explain_object(lattice, "m8")

    assert explanation.attributes == ("mammal", "warm", "swim")
    assert explanation.indistinguishable_from == ()
    assert explain_object(lattice, "m0").indistinguishable_from[:2] == ("m1", "m2")
    assert any(extra == ("swim",) for _, extra in explanation.beyond_more_general)
    assert "m8" in str(explanation)


# ---------------------------------------------------------------------------
# ImplicationOutlierDetector
# ---------------------------------------------------------------------------


def test_detector_flags_the_rule_breaker():
    ctx = animals()
    X = pd.DataFrame(ctx.incidence, index=ctx.objects, columns=ctx.attributes)
    detector = ImplicationOutlierDetector(
        min_confidence=0.85, min_support=3, contamination="auto"
    ).fit(X)

    # "odd" is a mammal that is not warm-blooded; the bird is warm-blooded
    # but not a mammal. Each breaks one rule that holds for 9 of 10 objects.
    flagged = {ctx.objects[g] for g in np.flatnonzero(detector.predict(X) == -1)}
    assert flagged == {"odd", "bird"}

    why = detector.explain(X)
    odd_rules = why[why["object"] == ctx.objects.index("odd")]
    assert list(odd_rules["premise"]) == [("mammal",)]
    assert list(odd_rules["missing"]) == [("warm",)]
    bird_rules = why[why["object"] == ctx.objects.index("bird")]
    assert list(bird_rules["missing"]) == [("mammal",)]


@pytest.mark.parametrize("seed", range(10))
def test_scores_follow_the_rule_weight_definition(seed):
    rng = np.random.default_rng(seed)
    X = rng.random((40, 6)) < 0.5
    detector = ImplicationOutlierDetector(min_confidence=0.7, min_support=3).fit(X)

    expected = np.zeros(len(X))
    for (premise, conclusion, support, premise_support), weight in zip(
        detector.rules_, detector.weights_
    ):
        assert math.isclose(weight, -math.log2(1 - (support + 1) / (premise_support + 2)))
        for g, row in enumerate(X):
            attributes = set(np.flatnonzero(row))
            if premise <= attributes and not conclusion <= attributes:
                expected[g] -= weight

    assert np.allclose(detector.score_samples(X), expected)
    assert np.allclose(detector.decision_function(X), expected - detector.offset_)


def test_contamination_flags_the_requested_share():
    rng = np.random.default_rng(3)
    X = rng.random((200, 6)) < 0.4
    X[:, 1] = X[:, 0]  # rule: column 0 -> column 1 ...
    X[rng.choice(np.flatnonzero(X[:, 0]), 12, replace=False), 1] = False  # ... with exceptions
    detector = ImplicationOutlierDetector(min_confidence=0.75, contamination=0.1).fit(X)

    scores = detector.score_samples(X)
    flagged = detector.predict(X) == -1
    cutoff = np.percentile(scores, 10, method="lower")

    # Everything at or below the cutoff is flagged (ties included), nothing above.
    assert flagged.any()
    assert np.array_equal(flagged, (scores <= cutoff) & (scores < 0))


def test_detector_parameter_validation():
    X = np.random.default_rng(0).random((10, 3)) < 0.5
    with pytest.raises(ValueError, match="contamination"):
        ImplicationOutlierDetector(contamination=0.8).fit(X)
    with pytest.raises(TypeError, match="ordered"):
        from conceptflow.quality import check_implications

        check_implications(FormalContext.from_array(X), [{0, 1}])
    with pytest.raises(ValueError, match="min_confidence"):
        ImplicationOutlierDetector(min_confidence=0).fit(X)
