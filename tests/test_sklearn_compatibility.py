import numpy as np
import pandas as pd
import pytest
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


def _named_categories():
    return pd.DataFrame(
        {"color": ["red", "blue", "red", "green"], "w": [1.0, 2.0, 3.0, 4.0]}
    )


@pytest.mark.parametrize("factorization", ["GreConD", "ASSO"])
@pytest.mark.parametrize("output", ["array", "sparse"])
def test_pipeline_feature_names_after_scaling(factorization, output):
    # The factorization is fitted on the scaler's (unnamed) array output and
    # must accept the names the Pipeline passes on from the scaler.
    from sklearn.pipeline import make_pipeline

    from conceptflow import decomposition
    from conceptflow.preprocessing import ConceptualScaler, NominalScale

    pipeline = make_pipeline(
        ConceptualScaler([NominalScale("color")], output=output),
        getattr(decomposition, factorization)(),
    ).fit(_named_categories()[["color"]])
    names = pipeline.get_feature_names_out()
    assert list(names) == [f"{factorization.lower()}{i}" for i in range(len(names))]


def test_scaler_after_a_step_that_returns_arrays():
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline

    from conceptflow.preprocessing import ConceptualScaler, NominalScale

    pipeline = make_pipeline(
        SimpleImputer(strategy="most_frequent"), ConceptualScaler([NominalScale(0)])
    ).fit(_named_categories()[["color"]])
    assert list(pipeline.get_feature_names_out()) == ["x0=blue", "x0=green", "x0=red"]


@pytest.mark.parametrize(
    "columns, expected",
    [
        (["color", "w"], ["s__color=blue", "s__color=green", "s__color=red", "n__w"]),
        # Integer column names are not feature names in scikit-learn, which
        # passes x0, x1, ... on instead.
        ([0, 1], ["s__0=blue", "s__0=green", "s__0=red", "n__x1"]),
    ],
)
def test_column_transformer_feature_names_with_the_scaler(columns, expected):
    from sklearn.compose import ColumnTransformer

    from conceptflow.preprocessing import ConceptualScaler, NominalScale

    data = _named_categories().set_axis(columns, axis=1)
    transformer = ColumnTransformer(
        [
            ("s", ConceptualScaler([NominalScale(columns[0])]), [columns[0]]),
            ("n", "passthrough", [columns[1]]),
        ]
    ).fit(data)
    assert list(transformer.get_feature_names_out()) == expected


def test_input_features_are_still_checked():
    from conceptflow.decomposition import GreConD
    from conceptflow.preprocessing import ConceptualScaler, NominalScale

    named = GreConD().fit(pd.DataFrame(np.eye(3, dtype=bool), columns=["a", "b", "c"]))
    with pytest.raises(ValueError, match="not equal to feature_names_in_"):
        named.get_feature_names_out(["p", "q", "r"])
    assert len(named.get_feature_names_out(["a", "b", "c"])) == named.n_components_
    unnamed = GreConD().fit(np.eye(3, dtype=bool))
    assert len(unnamed.get_feature_names_out(["p", "q", "r"])) == unnamed.n_components_
    with pytest.raises(ValueError, match="length equal to number of features"):
        unnamed.get_feature_names_out(["p", "q"])
    scaler = ConceptualScaler([NominalScale("color")]).fit(_named_categories()[["color"]])
    with pytest.raises(ValueError, match="not equal to feature_names_in_"):
        scaler.get_feature_names_out(["colour"])


def test_integer_column_names_are_attribute_names_and_checked_in_order():
    # Integer column names (e.g. the divisors 1..6) are FCA attribute names;
    # only input_features follows scikit-learn's rule for them.
    from conceptflow.decomposition import GreConD

    divisors = pd.DataFrame(
        [[n % d == 0 for d in range(1, 7)] for n in range(1, 7)], columns=range(1, 7)
    )
    factorization = GreConD().fit(divisors)
    assert factorization.feature_names_in_ == ("1", "2", "3", "4", "5", "6")
    with pytest.raises(ValueError, match="different order"):
        factorization.transform(divisors[[6, 5, 4, 3, 2, 1]])
    names = [f"x{i}" for i in range(6)]  # what a ColumnTransformer passes on
    assert len(factorization.get_feature_names_out(names)) == factorization.n_components_
