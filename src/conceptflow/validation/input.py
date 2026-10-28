"""
Input validation and conversion utilities.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse

from conceptflow.core import FormalContext, IncompleteContext
from conceptflow.core.context import _cross_table

_UNKNOWN_HINT = (
    "Binary-context estimators need complete data. Choose a completion with "
    "IncompleteContext.certain_context() / .possible_context(), or scale with "
    'ConceptualScaler(missing="false").'
)


def check_binary_context_input(X) -> FormalContext:
    """
    Validate and convert binary context input to FormalContext.

    Supported inputs:
    - FormalContext
    - pandas.DataFrame
    - numpy.ndarray
    - scipy sparse matrices
    - list of lists
    """
    if isinstance(X, FormalContext):
        return X

    if isinstance(X, IncompleteContext):
        if X.is_complete:
            return X.certain_context()
        raise ValueError(f"Input contains NaN: the context has unknown entries. {_UNKNOWN_HINT}")

    if isinstance(X, pd.DataFrame):
        if X.isna().to_numpy().any():
            raise ValueError(
                "Input contains NaN: missing entries. If empty cells mean "
                '"no cross", pass FormalContext.from_dataframe(X) instead. '
                f"For unknown entries: {_UNKNOWN_HINT}"
            )
        if X.isin([np.inf, -np.inf]).to_numpy().any():
            # As for arrays (scikit-learn's convention).
            raise ValueError("Input contains infinity or a value too large.")
        return FormalContext.from_dataframe(X)

    if sparse.issparse(X):
        array = X.toarray()
    else:
        array = np.asarray(X)

    if np.iscomplexobj(array):
        raise ValueError("Complex data not supported")

    if array.dtype.kind in "US":
        array = array.astype(object)

    if array.dtype == object:
        # Try to coerce to numeric — raises TypeError with sklearn-compatible
        # message if elements cannot be converted (e.g. dicts, arbitrary objects).
        try:
            array = array.astype(float)
        except ValueError:
            # Strings such as "x" / ".": read cell by cell, as FormalContext
            # does (unknown spellings raise a ValueError naming the value).
            # Empty cells are still rejected, as for numeric input.
            if pd.isna(array).any():
                raise ValueError(
                    "Input contains NaN: missing entries. If empty cells mean "
                    '"no cross", pass FormalContext.from_array(X) instead.'
                ) from None
            array = _cross_table(array)

    if array.ndim != 2:
        raise ValueError(
            "Reshape your data: expected 2D array, got %dD array instead."
            % array.ndim
        )

    if array.shape[0] == 0:
        raise ValueError(
            "0 sample(s) (shape=%s) while a minimum of 1 is required."
            % str(array.shape)
        )

    if array.ndim == 2 and array.shape[1] == 0:
        raise ValueError(
            "0 feature(s) (shape=%s) while a minimum of 1 is required."
            % str(array.shape)
        )

    if not np.isfinite(array).all():
        raise ValueError(
            "Input contains NaN, infinity or a value too large for "
            "dtype('%s')." % array.dtype
        )

    return FormalContext.from_array(array)


def check_output_param(estimator) -> None:
    """``output`` must be "array", "dataframe" or "sparse" (checked at fit and
    transform time, since set_params can change it after fit)."""
    if estimator.output not in {"array", "dataframe", "sparse"}:
        raise ValueError('output must be "array", "dataframe", or "sparse".')


def input_feature_names(X) -> tuple[str, ...] | None:
    """Column names of named input (DataFrame or FormalContext), else None."""
    if isinstance(X, pd.DataFrame):
        return tuple(map(str, X.columns))
    if isinstance(X, FormalContext):
        return X.attributes
    return None


def remember_feature_names(estimator, X) -> None:
    """At fit time: remember the column names of named input."""
    estimator._fit_feature_names = input_feature_names(X)


def check_feature_names(estimator, X) -> None:
    """
    At transform/predict time: named input must have the same columns, in
    the same order, as the named input seen during fit (as in scikit-learn).
    """
    fitted = getattr(estimator, "_fit_feature_names", None)
    names = input_feature_names(X)
    if fitted is not None and names is not None and names != fitted:
        unexpected = [n for n in names if n not in fitted]
        missing = [n for n in fitted if n not in names]
        detail = (
            f" Unseen: {unexpected[:5]}. Missing: {missing[:5]}."
            if unexpected or missing
            else " The columns are the same but in a different order."
        )
        raise ValueError(
            "The feature names should match those that were passed during fit." + detail
        )
