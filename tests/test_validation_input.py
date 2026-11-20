import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from conceptflow import FormalContext
from conceptflow.validation import check_binary_context_input


def test_check_binary_context_input_accepts_formal_context():
    ctx = FormalContext.from_array([[1, 0], [0, 1]])

    result = check_binary_context_input(ctx)

    assert result is ctx


def test_check_binary_context_input_accepts_dataframe():
    df = pd.DataFrame(
        [[1, 0], [0, 1]],
        index=["g1", "g2"],
        columns=["m1", "m2"],
    )

    ctx = check_binary_context_input(df)

    assert ctx.objects == ("g1", "g2")
    assert ctx.attributes == ("m1", "m2")


def test_check_binary_context_input_accepts_numpy_array():
    X = np.array([[1, 0], [0, 1]])

    ctx = check_binary_context_input(X)

    assert ctx.n_objects == 2
    assert ctx.n_attributes == 2


def test_check_binary_context_input_accepts_sparse_matrix():
    X = sparse.csr_matrix([[1, 0], [0, 1]])

    ctx = check_binary_context_input(X)

    assert ctx.n_objects == 2
    assert ctx.n_attributes == 2
    assert ctx.incidence.dtype == bool


def test_check_binary_context_input_accepts_list_of_lists():
    X = [[1, 0], [0, 1]]

    ctx = check_binary_context_input(X)

    assert ctx.n_objects == 2
    assert ctx.n_attributes == 2


def test_check_binary_context_input_rejects_1d_input():
    with pytest.raises(ValueError, match="Reshape your data"):
        check_binary_context_input([1, 0, 1])


def test_check_binary_context_input_rejects_complex():
    X = np.array([[1 + 2j, 0], [0, 1 + 0j]])
    with pytest.raises(ValueError, match="Complex data not supported"):
        check_binary_context_input(X)


def test_check_binary_context_input_rejects_nan():
    X = np.array([[1.0, 0.0], [float("nan"), 1.0]])
    with pytest.raises(ValueError, match="NaN"):
        check_binary_context_input(X)


def test_check_binary_context_input_rejects_inf():
    X = np.array([[1.0, 0.0], [float("inf"), 1.0]])
    with pytest.raises(ValueError, match="NaN"):
        check_binary_context_input(X)


def test_check_binary_context_input_rejects_empty_samples():
    X = np.empty((0, 3))
    with pytest.raises(ValueError, match="0 sample"):
        check_binary_context_input(X)


def test_check_binary_context_input_rejects_empty_features():
    X = np.empty((3, 0))
    with pytest.raises(ValueError, match="0 feature"):
        check_binary_context_input(X)


def test_check_binary_context_input_accepts_object_dtype_numerics():
    X = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=object)
    ctx = check_binary_context_input(X)
    assert ctx.n_objects == 2
    assert ctx.n_attributes == 2


def test_check_binary_context_input_rejects_object_dtype_non_numeric():
    X = np.array([[{"a": 1}, 0], [0, 1]], dtype=object)
    with pytest.raises(TypeError):
        check_binary_context_input(X)


# Found by tests/test_properties_input.py: the same cells were read
# differently depending on the container they came in.


@pytest.mark.parametrize(
    "X",
    [
        [["2", "1"]],
        np.array([["2", "1"]]),
        np.array([["1.5", "x"]], dtype=object),
        pd.DataFrame([["-1", "1"]]),
    ],
)
def test_numeric_strings_are_refused_in_every_container(X):
    # Only known spellings ("1", "x", "true", ...) are crosses; arrays read
    # "2" or "1.5" as numbers before, lists and DataFrames refused them.
    with pytest.raises(ValueError, match="Cannot read"):
        check_binary_context_input(X)
    with pytest.raises(ValueError, match="Cannot read"):
        FormalContext.from_array(np.asarray(X, dtype=object))


def test_numbers_in_a_list_with_strings_stay_numbers():
    # NumPy turns [2, "x"] into ["2", "x"]; the number 2 is a cross.
    rows = [[2, "x"], [0, "."]]
    expected = [[True, True], [False, False]]
    assert FormalContext.from_array(rows).incidence.tolist() == expected
    assert check_binary_context_input(rows).incidence.tolist() == expected


@pytest.mark.parametrize("cell", [pd.NA, None, np.nan, np.inf, 1j])
def test_missing_infinite_and_complex_cells_raise_value_error(cell):
    for X in ([[cell, 1]], [[cell, "x"]], np.array([[cell, 1]], dtype=object)):
        with pytest.raises(ValueError):
            check_binary_context_input(X)
