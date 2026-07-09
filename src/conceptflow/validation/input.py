"""
Input validation and conversion utilities.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse

from conceptflow.core import FormalContext


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

    if isinstance(X, pd.DataFrame):
        return FormalContext.from_dataframe(X)

    if sparse.issparse(X):
        array = X.toarray()
    else:
        array = np.asarray(X)

    if np.iscomplexobj(array):
        raise ValueError("Complex data not supported")

    if array.dtype == object:
        # Try to coerce to numeric — raises TypeError with sklearn-compatible
        # message if elements cannot be converted (e.g. dicts, arbitrary objects).
        array = array.astype(float)

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