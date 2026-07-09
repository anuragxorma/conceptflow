import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline

from conceptflow.cluster import ConceptLatticeEstimator
from conceptflow.feature_extraction import ConceptMembershipEncoder
from conceptflow.preprocessing import ConceptualScaler, NominalScale


def test_concept_membership_encoder_pipeline_compatibility():
    X = np.array([
        [1, 1, 0],
        [1, 0, 1],
        [0, 1, 1],
    ])

    pipeline = Pipeline([
        (
            "encoder",
            ConceptMembershipEncoder(output="array"),
        ),
    ])

    transformed = pipeline.fit_transform(X)

    assert transformed.shape[0] == 3


def test_conceptual_scaler_pipeline_compatibility():
    X = pd.DataFrame(
        {
            "a": ["x", "y", "x"],
            "b": ["u", "u", "v"],
        }
    )

    pipeline = Pipeline([
        (
            "scaler",
            ConceptualScaler(
                scales=[
                    NominalScale("a"),
                    NominalScale("b"),
                ],
                output="array",
            ),
        ),
    ])

    transformed = pipeline.fit_transform(X)

    assert transformed.shape[0] == 3


def test_concept_membership_encoder_sklearn_cloneable():
    from sklearn.base import clone

    encoder = ConceptMembershipEncoder(
        algorithm="nextclosure",
        output="array",
    )

    cloned = clone(encoder)

    assert cloned.algorithm == "nextclosure"
    assert cloned.output == "array"


def test_concept_lattice_basic_estimator_behavior():
    X = np.array([
        [1, 1, 0],
        [1, 0, 1],
        [0, 1, 1],
    ])

    estimator = ConceptLatticeEstimator()

    estimator.fit(X)

    assert hasattr(estimator, "concepts_")
    assert hasattr(estimator, "lattice_")