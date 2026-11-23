"""
Property-based tests of the scikit-learn estimators: every output column
means what the documentation says, on the training data and on new data.

- ConceptLatticeEstimator / ConceptMembershipEncoder: a concept column is
  True iff the row has every attribute of the concept's intent.
- ImplicationBasisEstimator: the canonical basis; a column is True iff the
  row has the implication's premise.
- GreConD: factors are formal concepts that never cover a 0, and with
  ``min_coverage=1`` the factorization rebuilds X exactly.
- ASSO: ``reconstruction_error_`` and ``coverage_`` describe the product.
- ImplicationOutlierDetector: scores are minus the weights of the broken
  rules, and ``explain`` lists exactly those rules.
"""

import numpy as np
import pytest

pytest.importorskip("hypothesis")

from hypothesis import given  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

from conceptflow.algorithms import compute_canonical_basis, enumerate_concepts  # noqa: E402
from conceptflow.cluster import ConceptLatticeEstimator  # noqa: E402
from conceptflow.decomposition import ASSO, GreConD  # noqa: E402
from conceptflow.feature_extraction import ConceptMembershipEncoder  # noqa: E402
from conceptflow.metrics import concept_stability, delta_measure  # noqa: E402
from conceptflow.rules import ImplicationBasisEstimator, ImplicationOutlierDetector  # noqa: E402
from tests.property_helpers import context_of, incidences  # noqa: E402

training = incidences(min_objects=1, max_objects=8, min_attributes=1, max_attributes=6)


@st.composite
def train_and_new(draw):
    """Training data and new rows with the same columns."""
    train = draw(training, label="train")
    m = train.shape[1]
    n = draw(st.integers(1, 6), label="new rows")
    cells = draw(st.lists(st.booleans(), min_size=n * m, max_size=n * m), label="new")
    return train, np.array(cells, dtype=bool).reshape(n, m)


def contains(rows, attribute_sets):
    return np.array(
        [[all(row[j] for j in attributes) for attributes in attribute_sets] for row in rows],
        dtype=bool,
    ).reshape(len(rows), len(attribute_sets))


@given(train_and_new(), st.sampled_from([None, 1, 2, 0.5]))
def test_concept_columns_are_intent_containment(data, min_support):
    train, new = data
    estimator = ConceptLatticeEstimator(min_support=min_support).fit(train)
    expected = {
        (c.extent, c.intent)
        for c in enumerate_concepts(context_of(train), min_support=min_support)
    }
    assert {(c.extent, c.intent) for c in estimator.concepts_} == expected
    intents = [c.intent for c in estimator.concepts_]
    for X in (train, new):
        np.testing.assert_array_equal(estimator.transform(X), contains(X, intents))
    # On the training data the columns are the extents.
    for k, concept in enumerate(estimator.concepts_):
        assert set(np.flatnonzero(estimator.transform(train)[:, k])) == concept.extent
    assert list(estimator.get_feature_names_out()) == [c.stable_id() for c in estimator.concepts_]

    encoder = ConceptMembershipEncoder(output="array").fit(train)
    assert set(encoder.concepts_) == set(
        ConceptLatticeEstimator().fit(train).concepts_
    )
    for X in (train, new):
        np.testing.assert_array_equal(
            encoder.transform(X), contains(X, [c.intent for c in encoder.concepts_])
        )


@given(training, st.sampled_from(["stability", "delta", "support"]), st.integers(1, 6))
def test_selected_concepts_are_the_best_scored(train, select, n_concepts):
    full = ConceptLatticeEstimator().fit(train)
    chosen = ConceptLatticeEstimator(select=select, n_concepts=n_concepts).fit(train)
    lattice = full.lattice_
    if select == "stability":
        scores = concept_stability(lattice)
    elif select == "delta":
        scores = delta_measure(lattice)
    else:
        scores = np.array([len(c.extent) for c in lattice.concepts])
    score = dict(zip(lattice.concepts, scores))
    kept = set(chosen.concepts_)
    assert len(kept) == min(n_concepts, lattice.n_concepts)
    rest = [score[c] for c in lattice.concepts if c not in kept]
    if rest:
        assert min(score[c] for c in kept) >= max(rest)
    # Listed in lattice order.
    order = {c: i for i, c in enumerate(lattice.concepts)}
    assert [order[c] for c in chosen.concepts_] == sorted(order[c] for c in chosen.concepts_)


@given(train_and_new())
def test_implication_columns_are_premise_containment(data):
    train, new = data
    estimator = ImplicationBasisEstimator().fit(train)
    basis = compute_canonical_basis(context_of(train))
    assert list(estimator.get_implications()) == basis
    premises = [imp.premise for imp in basis]
    for X in (train, new):
        np.testing.assert_array_equal(estimator.transform(X), contains(X, premises))
    assert len(estimator.get_feature_names_out()) == len(basis)


def boolean_product(usage, components):
    return (usage.astype(int) @ components.astype(int)) > 0


@given(training, st.sampled_from([1.0, 0.9, 0.5]), st.sampled_from([None, 1, 2, 3]))
def test_grecond_factors_are_concepts_that_cover_no_zero(train, min_coverage, n_components):
    model = GreConD(min_coverage=min_coverage, n_components=n_components).fit(train)
    concepts = {(c.extent, c.intent) for c in enumerate_concepts(context_of(train))}
    for factor, component in zip(model.factors_, model.components_):
        assert (factor.extent, factor.intent) in concepts
        assert set(np.flatnonzero(component)) == factor.intent
    usage = model.transform(train)
    rebuilt = boolean_product(usage, model.components_)
    np.testing.assert_array_equal(model.inverse_transform(usage), rebuilt)
    assert not (rebuilt & ~train).any()  # never covers a 0
    ones = int(train.sum())
    coverage = (rebuilt & train).sum() / ones if ones else 1.0
    assert model.coverage_ == pytest.approx(coverage)
    assert model.reconstruction_error_ == int((rebuilt != train).sum())
    if n_components is not None:
        assert model.n_components_ <= n_components
    if n_components is None:
        assert model.coverage_ >= min_coverage - 1e-12
        if min_coverage == 1.0:
            np.testing.assert_array_equal(rebuilt, train)


@given(training, st.sampled_from([0.5, 0.7, 1.0]))
def test_asso_reports_its_reconstruction(train, threshold):
    model = ASSO(threshold=threshold).fit(train)
    usage = model.transform(train)
    rebuilt = boolean_product(usage, model.components_)
    np.testing.assert_array_equal(model.inverse_transform(usage), rebuilt)
    assert model.reconstruction_error_ == int((rebuilt != train).sum())
    ones = int(train.sum())
    assert model.coverage_ == pytest.approx((rebuilt & train).sum() / ones if ones else 1.0)
    assert model.n_components_ <= train.shape[1]


@given(train_and_new(), st.sampled_from([0.6, 0.9, 1.0]), st.booleans(),
       st.sampled_from(["auto", 0.1, 0.5]))
def test_outlier_scores_are_the_weights_of_broken_rules(data, min_confidence, exact,
                                                        contamination):
    train, new = data
    detector = ImplicationOutlierDetector(
        min_confidence=min_confidence, min_support=1, include_exact=exact,
        contamination=contamination,
    ).fit(train)
    for X in (train, new):
        broken = np.array(
            [[all(row[j] for j in premise) and not all(row[j] for j in conclusion)
              for premise, conclusion, _, _ in detector.rules_] for row in X],
            dtype=bool,
        ).reshape(len(X), len(detector.rules_))
        scores = detector.score_samples(X)
        np.testing.assert_allclose(scores, -(broken.astype(float) @ detector.weights_))
        np.testing.assert_array_equal(
            detector.predict(X), np.where(detector.decision_function(X) < 0, -1, 1)
        )
        explanation = detector.explain(X)
        assert len(explanation) == int(broken.sum())
        assert sorted(explanation["object"]) == sorted(np.nonzero(broken)[0].tolist())
        # Objects breaking no rule are never outliers.
        assert (detector.predict(X)[~broken.any(axis=1)] == 1).all()
        if contamination == "auto":
            np.testing.assert_array_equal(detector.predict(X) == -1, broken.any(axis=1))
    for _, _, support, premise_support in detector.rules_:
        assert 1 <= support <= premise_support
