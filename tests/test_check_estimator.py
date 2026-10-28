"""
Formal sklearn check_estimator / parametrize_with_checks tests.

All transformers declare ``tags.transformer_tags.preserves_dtype = []``: they
output boolean membership indicators (or 0/1/NaN floats for
``ConceptualScaler(missing="unknown")``) whatever the input dtype, so sklearn
does not expect dtype preservation.

ConceptualScaler is checked with ``ThresholdScale(0, ...)``, which resolves to
source_attribute="x0" (generic numpy arrays get synthesised column names x0,
x1, ...). A threshold scale accepts any number, which is what sklearn's checks
feed in. Nominal-type scales instead keep the vocabulary seen at fit time and
reject unseen values; sklearn's ``check_fit_idempotent`` transforms freshly
generated random floats, so it cannot pass for them. That design boundary is
tested directly below (``test_nominal_vocabulary_is_fixed_at_fit_time``).

The one remaining expected failure:
  - check_estimators_nan_inf (missing="error" only): NaN is a missing value
    and is rejected exactly as sklearn expects, but inf is a legitimate
    many-valued value (e.g. for ThresholdScale), so it is scaled, not rejected.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.utils.estimator_checks import parametrize_with_checks

from conceptflow.cluster import ConceptLatticeEstimator
from conceptflow.decomposition import ASSO, GreConD, Ord2Factor
from conceptflow.feature_extraction import ConceptMembershipEncoder
from conceptflow.preprocessing import ConceptualScaler, NominalScale, ThresholdScale
from conceptflow.rules import ImplicationBasisEstimator, ImplicationOutlierDetector

_INF_IS_A_VALUE = (
    'NaN is rejected under the default missing="error", but inf is a valid '
    "many-valued value and is scaled like any other value."
)


def _scaler_expected_failures(est):
    if est.missing == "error":
        return {"check_estimators_nan_inf": _INF_IS_A_VALUE}
    return {}


def _scaler(**kwargs):
    return ConceptualScaler(scales=[ThresholdScale(0, [0.0, 0.5])], **kwargs)


@parametrize_with_checks([ConceptMembershipEncoder()])
def test_concept_membership_encoder_sklearn_checks(estimator, check):
    check(estimator)


@parametrize_with_checks(
    [
        ConceptLatticeEstimator(),
        ConceptLatticeEstimator(min_support=2),
        ConceptLatticeEstimator(select="stability", n_concepts=5),
        ConceptLatticeEstimator(select="delta", n_concepts=3, min_support=0.2),
        ConceptLatticeEstimator(select="support", n_concepts=4, algorithm="closebyone"),
    ]
)
def test_concept_lattice_estimator_sklearn_checks(estimator, check):
    check(estimator)


@parametrize_with_checks(
    [
        _scaler(),
        _scaler(missing="false"),
        _scaler(missing="unknown"),
        _scaler(missing="unknown", output="dataframe"),
        _scaler(output="dataframe"),
        _scaler(output="sparse"),
    ],
    expected_failed_checks=_scaler_expected_failures,
)
def test_conceptual_scaler_sklearn_checks(estimator, check):
    check(estimator)


@parametrize_with_checks([ImplicationBasisEstimator()])
def test_implication_basis_estimator_sklearn_checks(estimator, check):
    check(estimator)


@parametrize_with_checks(
    [
        ImplicationOutlierDetector(),
        ImplicationOutlierDetector(contamination=0.1),
        ImplicationOutlierDetector(min_confidence=0.8, min_support=2, include_exact=False),
    ]
)
def test_implication_outlier_detector_sklearn_checks(estimator, check):
    check(estimator)


def test_nominal_vocabulary_is_fixed_at_fit_time():
    # The design boundary that keeps NominalScale out of the generic checks:
    # feature names never drift with transform-time data, and unseen values
    # raise instead of being silently dropped.
    train = pd.DataFrame({"color": ["red", "blue"]})
    scaler = ConceptualScaler(scales=[NominalScale("color")]).fit(train)

    assert list(scaler.get_feature_names_out()) == ["color=blue", "color=red"]
    assert scaler.transform(pd.DataFrame({"color": ["red"]})).tolist() == [[False, True]]

    with pytest.raises(ValueError, match="vocabulary"):
        scaler.transform(pd.DataFrame({"color": ["green"]}))

    # Refitting on the same data is idempotent.
    refit = ConceptualScaler(scales=[NominalScale("color")]).fit(train)
    assert np.array_equal(refit.transform(train), scaler.transform(train))


@parametrize_with_checks(
    [
        GreConD(),
        GreConD(min_coverage=0.8, n_components=3),
        ASSO(),
        ASSO(n_components=2, threshold=0.5),
    ]
)
def test_boolean_factorization_sklearn_checks(estimator, check):
    check(estimator)


# ExactOrdinalTwoFactorizer is not checked here: it only accepts
# two-factorizable contexts and rejects the random data the checks use.
# Ord2Factor factorizes any context.
@parametrize_with_checks(
    [
        Ord2Factor(),
        Ord2Factor(method="greedy", random_state=0),
        Ord2Factor(method="annealing", n_iterations=50, random_state=0),
    ]
)
def test_ord2factor_sklearn_checks(estimator, check):
    check(estimator)
