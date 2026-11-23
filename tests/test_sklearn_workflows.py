"""
ConceptFlow's estimators in the scikit-learn workflows users run.

- Searches run in parallel (``n_jobs=2``, estimators pickled to worker
  processes) give the results of a sequential search.
- ``set_output(transform="pandas")`` works for every transformer, alone and
  in a Pipeline, with the input's row index and ``get_feature_names_out``.
- Concept ids are the same in every cross-validation fold, as documented.
- ``missing="unknown"`` scaling works in a ColumnTransformer with an imputer.
- Fitted pipelines survive joblib; fitting never changes the input.
- Results do not depend on the order of the rows.
"""

import pickle

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from sklearn.pipeline import make_pipeline

from conceptflow.cluster import ConceptLatticeEstimator
from conceptflow.decomposition import ASSO, GreConD
from conceptflow.feature_extraction import ConceptMembershipEncoder
from conceptflow.preprocessing import (
    ConceptualScaler,
    InterordinalScale,
    NominalScale,
    OrdinalScale,
    ThresholdScale,
)
from conceptflow.rules import ImplicationBasisEstimator

SCALES = [NominalScale("colour"), ThresholdScale("size", [2.5, 5, 7.5]),
          OrdinalScale("grade", ["low", "mid", "high"])]


def table(n, seed, missing=0.0):
    rng = np.random.default_rng(seed)
    data = pd.DataFrame(
        {
            "colour": rng.choice(["red", "green", "blue", "grey"], n, p=[.4, .3, .2, .1]),
            "size": rng.uniform(0, 10, n).round(1),
            "grade": rng.choice(["low", "mid", "high"], n),
        },
        index=[f"item{i}" for i in range(n)],
    )
    label = ((data["colour"] == "red") & (data["size"] > 5)) | (data["grade"] == "high")
    if missing:
        data = data.astype(object)
        holes = rng.random(data.shape) < missing
        data = data.mask(holes)
    return data, label.astype(int).to_numpy()


def binary(n=40, seed=0):
    data, label = table(n, seed)
    return ConceptualScaler(scales=SCALES, output="dataframe").fit_transform(data), label


TRANSFORMERS = {
    "lattice": lambda: ConceptLatticeEstimator(),
    "stable concepts": lambda: ConceptLatticeEstimator(select="stability", n_concepts=6),
    "encoder": lambda: ConceptMembershipEncoder(output="array"),
    "implications": lambda: ImplicationBasisEstimator(),
    "grecond": lambda: GreConD(),
    "asso": lambda: ASSO(),
}


def test_parallel_search_gives_the_sequential_results():
    data, label = table(90, seed=3)
    pipeline = make_pipeline(
        ConceptualScaler(scales=SCALES, handle_unknown="ignore"),
        ConceptLatticeEstimator(select="stability"),
        LogisticRegression(max_iter=1000),
    )
    grid = {"conceptlatticeestimator__n_concepts": [4, 12],
            "conceptlatticeestimator__select": ["stability", "delta", "support"]}
    results = [
        GridSearchCV(clone(pipeline), grid, cv=3, n_jobs=n_jobs, error_score="raise")
        .fit(data, label).cv_results_["mean_test_score"]
        for n_jobs in (1, 2)
    ]
    np.testing.assert_array_equal(results[0], results[1])


@pytest.mark.parametrize("name", TRANSFORMERS)
def test_pandas_output_alone_and_in_a_pipeline(name):
    X, _ = binary()
    for estimator in (TRANSFORMERS[name](), make_pipeline(TRANSFORMERS[name]())):
        output = estimator.set_output(transform="pandas").fit_transform(X)
        assert isinstance(output, pd.DataFrame)
        assert list(output.columns) == list(estimator.get_feature_names_out())
        assert list(output.index) == list(X.index)
        again = estimator.transform(X.iloc[::-1])
        assert list(again.index) == list(X.index[::-1])


def test_pandas_output_of_the_scaler_in_a_column_transformer():
    data, _ = table(30, seed=4)
    data["extra"] = np.arange(len(data))
    transformer = ColumnTransformer(
        [("scale", ConceptualScaler(scales=SCALES), ["colour", "size", "grade"])],
        remainder="passthrough",
    ).set_output(transform="pandas")
    output = transformer.fit_transform(data)
    assert list(output.columns) == list(transformer.get_feature_names_out())
    assert list(output.index) == list(data.index)
    assert output.columns[-1] == "remainder__extra"


def test_concept_ids_are_the_same_in_every_fold():
    X, _ = binary(60, seed=5)
    fitted = [ConceptLatticeEstimator(output="dataframe").fit(X.iloc[train])
              for train, _ in KFold(4, shuffle=True, random_state=0).split(X)]
    outputs = [model.transform(X) for model in fitted]
    for model, output in zip(fitted, outputs):
        intents = {c.stable_id(): c.intent for c in model.concepts_}
        assert list(output.columns) == list(intents)
    # A concept found in two folds has one id and the same column there.
    for a in range(len(outputs)):
        for b in range(a + 1, len(outputs)):
            common = outputs[a].columns.intersection(outputs[b].columns)
            assert len(common) > 0
            pd.testing.assert_frame_equal(outputs[a][common], outputs[b][common])


def test_unknown_scaling_with_an_imputer_in_cross_validation():
    data, label = table(80, seed=6, missing=0.1)
    pipeline = make_pipeline(
        ColumnTransformer(
            [("scale", ConceptualScaler(scales=SCALES, missing="unknown",
                                        handle_unknown="ignore"),
              ["colour", "size", "grade"])]
        ),
        SimpleImputer(strategy="most_frequent"),
        LogisticRegression(max_iter=1000),
    )
    scores = cross_val_score(pipeline, data, label, cv=4, error_score="raise")
    assert np.isfinite(scores).all()
    output = pipeline[0].fit_transform(data)
    assert np.isnan(output).any() and output.shape[1] == len(pipeline[0].get_feature_names_out())


def test_fitted_pipelines_survive_joblib(tmp_path):
    data, label = table(60, seed=7)
    pipeline = make_pipeline(
        ConceptualScaler(scales=[*SCALES[:2], InterordinalScale("grade", ["low", "mid", "high"])],
                         handle_unknown="ignore"),
        ConceptLatticeEstimator(select="delta", n_concepts=8),
        LogisticRegression(max_iter=1000),
    ).fit(data, label)
    joblib.dump(pipeline, tmp_path / "model.joblib")
    loaded = joblib.load(tmp_path / "model.joblib")
    np.testing.assert_array_equal(loaded.predict_proba(data), pipeline.predict_proba(data))
    copied = pickle.loads(pickle.dumps(pipeline, protocol=5))
    np.testing.assert_array_equal(copied.predict(data), pipeline.predict(data))


@pytest.mark.parametrize("name", TRANSFORMERS)
def test_fitting_does_not_change_the_input(name):
    X, _ = binary()
    for data in (X, X.to_numpy(), X.to_numpy().astype(int)):
        before = data.copy()
        TRANSFORMERS[name]().fit(data).transform(data)
        if isinstance(data, pd.DataFrame):
            pd.testing.assert_frame_equal(data, before)
        else:
            np.testing.assert_array_equal(data, before)


def test_scaling_does_not_change_the_input():
    data, _ = table(30, seed=8, missing=0.1)
    before = data.copy()
    ConceptualScaler(scales=SCALES, missing="unknown").fit_transform(data)
    ConceptualScaler(scales=SCALES, missing="false").fit(data).transform(data)
    pd.testing.assert_frame_equal(data, before)


@pytest.mark.parametrize("name", [*TRANSFORMERS, "scaler"])
def test_results_do_not_depend_on_the_order_of_the_rows(name):
    rng = np.random.default_rng(9)
    if name == "scaler":
        X, _ = table(40, seed=9)
        make = lambda: ConceptualScaler(scales=SCALES)  # noqa: E731
    else:
        X, _ = binary(40, seed=9)
        make = TRANSFORMERS[name]
    order = rng.permutation(len(X))
    shuffled = X.iloc[order]
    model = make().fit(X)
    # Each row is transformed on its own.
    np.testing.assert_array_equal(np.asarray(model.transform(shuffled)),
                                  np.asarray(model.transform(X))[order])
    # Fitting on the rows in another order learns the same features (GreConD
    # and ASSO break ties by attribute, not by row).
    other = make().fit(shuffled)
    assert sorted(other.get_feature_names_out()) == sorted(model.get_feature_names_out())
    columns = list(model.get_feature_names_out())
    reordered = pd.DataFrame(np.asarray(other.transform(X)),
                             columns=other.get_feature_names_out())[columns]
    np.testing.assert_array_equal(reordered.to_numpy(), np.asarray(model.transform(X)))
    if name in ("grecond", "asso"):
        np.testing.assert_array_equal(other.components_, model.components_)
