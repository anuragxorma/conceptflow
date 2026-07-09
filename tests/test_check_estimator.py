"""
Formal sklearn check_estimator / parametrize_with_checks tests.

ConceptMembershipEncoder and ConceptLatticeEstimator pass all checks except
check_transformer_preserve_dtypes — both always output bool by design.

ConceptualScaler uses NominalScale(0), which resolves to source_attribute="x0".
Generic numpy arrays are converted to ManyValuedContexts with synthesised column
names x0, x1, ... so NominalScale(0) works with any 2D array.  The remaining
xfails are:
  - check_transformer_preserve_dtypes: always outputs bool (same as above)
  - check_estimators_nan_inf: NaN is treated as a valid category value; the
    scaler does not reject NaN input by design
"""

import pytest
from sklearn.utils.estimator_checks import parametrize_with_checks

from conceptflow.cluster import ConceptLatticeEstimator
from conceptflow.feature_extraction import ConceptMembershipEncoder
from conceptflow.preprocessing import ConceptualScaler, NominalScale


_BOOL_OUTPUT_XFAIL = (
    "Always outputs bool dtype regardless of input dtype — concept membership "
    "indicators are inherently boolean."
)

_CME_EXPECTED_FAILURES = {
    "check_transformer_preserve_dtypes": _BOOL_OUTPUT_XFAIL,
}

_CLE_EXPECTED_FAILURES = {
    "check_transformer_preserve_dtypes": _BOOL_OUTPUT_XFAIL,
}

_SCALER_NAN = (
    "NaN is treated as a valid category value in FCA; the scaler does not "
    "reject NaN input by design."
)

_SCALER_EXPECTED_FAILURES = {
    "check_transformer_preserve_dtypes": _BOOL_OUTPUT_XFAIL,
    "check_estimators_nan_inf": _SCALER_NAN,
}


@parametrize_with_checks(
    [ConceptMembershipEncoder()],
    expected_failed_checks=lambda est: _CME_EXPECTED_FAILURES,
)
def test_concept_membership_encoder_sklearn_checks(estimator, check):
    check(estimator)


@parametrize_with_checks(
    [ConceptLatticeEstimator()],
    expected_failed_checks=lambda est: _CLE_EXPECTED_FAILURES,
)
def test_concept_lattice_estimator_sklearn_checks(estimator, check):
    check(estimator)


@parametrize_with_checks(
    [ConceptualScaler(scales=[NominalScale(0)])],
    expected_failed_checks=lambda est: _SCALER_EXPECTED_FAILURES,
)
def test_conceptual_scaler_sklearn_checks(estimator, check):
    check(estimator)
