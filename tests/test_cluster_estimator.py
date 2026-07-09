import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from conceptflow.cluster import ConceptLattice, ConceptLatticeEstimator


def test_concept_lattice_estimator_fits_numpy_array():
    X = np.array([
        [1, 1, 0],
        [1, 0, 1],
        [0, 1, 1],
    ])

    model = ConceptLatticeEstimator(algorithm="nextclosure")
    model.fit(X)

    check_is_fitted(model, "lattice_")

    assert model.context_.n_objects == 3
    assert model.context_.n_attributes == 3
    assert len(model.concepts_) == 8
    assert len(model.edges_) > 0


def test_concept_lattice_estimator_fits_dataframe():
    X = pd.DataFrame(
        [
            [1, 1, 0],
            [1, 0, 1],
            [0, 1, 1],
        ],
        index=["g1", "g2", "g3"],
        columns=["m1", "m2", "m3"],
    )

    model = ConceptLatticeEstimator()
    model.fit(X)

    assert model.context_.objects == ("g1", "g2", "g3")
    assert model.context_.attributes == ("m1", "m2", "m3")
    assert len(model.concepts_) == 8


def test_concept_lattice_estimator_works_in_pipeline_as_final_step():
    X = np.array([
        [1, 1, 0],
        [1, 0, 1],
        [0, 1, 1],
    ])

    pipe = Pipeline([
        ("lattice", ConceptLatticeEstimator(algorithm="nextclosure")),
    ])

    pipe.fit(X)

    assert len(pipe.named_steps["lattice"].concepts_) == 8


def test_get_lattice_before_fit_raises_error():
    model = ConceptLatticeEstimator()

    with pytest.raises(Exception):
        model.get_lattice()


def test_invalid_algorithm_raises_error():
    X = np.array([[1, 0], [0, 1]])

    model = ConceptLatticeEstimator(algorithm="unknown")

    with pytest.raises(ValueError):
        model.fit(X)


def test_concept_lattice_estimator_supports_closebyone():
    X = np.array([
        [1, 1, 0],
        [1, 0, 1],
        [0, 1, 1],
    ])

    model = ConceptLatticeEstimator(algorithm="closebyone")
    model.fit(X)

    assert len(model.concepts_) == 8


def test_concept_lattice_deprecated_alias_emits_warning():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = ConceptLattice(algorithm="nextclosure")

    assert any(issubclass(w.category, DeprecationWarning) for w in caught)
    assert "ConceptLatticeEstimator" in str(caught[0].message)


def test_concept_lattice_deprecated_alias_still_works():
    X = np.array([[1, 0], [0, 1]])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        model = ConceptLattice()

    model.fit(X)

    assert hasattr(model, "lattice_")
