"""
GreConD and ASSO: checked against plain-loop implementations written from
the papers' pseudocode. scikit-learn's estimator checks are in
test_check_estimator.py.
"""

import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from sklearn.pipeline import make_pipeline

from conceptflow import FormalContext
from conceptflow.decomposition import ASSO, GreConD


def _random_data(seed, max_objects=12, max_attributes=8):
    rng = np.random.default_rng(seed)
    shape = (int(rng.integers(1, max_objects)), int(rng.integers(1, max_attributes)))
    return rng.random(shape) < rng.uniform(0.1, 0.9)


# ---------------------------------------------------------------------------
# Reference implementations (sets and loops, straight from the papers)
# ---------------------------------------------------------------------------


def _reference_grecond(X, min_coverage=1.0, n_components=None):
    n_objects, n_attributes = X.shape
    ones = {(g, m) for g in range(n_objects) for m in range(n_attributes) if X[g, m]}
    uncovered = set(ones)
    allowed = len(ones) - int(np.ceil(min_coverage * len(ones) - 1e-9))

    def down(attributes):
        return frozenset(g for g in range(n_objects) if all(X[g, m] for m in attributes))

    def up(objects):
        return frozenset(m for m in range(n_attributes) if all(X[g, m] for g in objects))

    factors = []
    while len(uncovered) > allowed and (n_components is None or len(factors) < n_components):
        intent, value = frozenset(), 0
        while True:
            best = None
            for j in range(n_attributes):
                if j in intent:
                    continue
                extent = down(intent | {j})
                closed = up(extent)
                score = sum((g, m) in uncovered for g in extent for m in closed)
                if best is None or score > best[0]:
                    best = (score, closed)
            if best is None or best[0] <= value:
                break
            value, intent = best
        extent = down(intent)
        factors.append((extent, intent))
        uncovered -= {(g, m) for g in extent for m in intent}
    return factors


def _reference_asso(X, threshold, n_components=None, w_plus=1.0, w_minus=1.0):
    n_objects, n_attributes = X.shape
    candidates = []
    for i in range(n_attributes):
        support = X[:, i].sum()
        row = [
            support > 0 and (X[:, i] & X[:, j]).sum() / support >= threshold - 1e-12
            for j in range(n_attributes)
        ]
        candidates.append(np.array(row, dtype=bool))

    covered = np.zeros_like(X)
    basis = []
    limit = n_attributes if n_components is None else n_components
    while len(basis) < limit:
        best = None
        for candidate in candidates:
            total, users = 0.0, []
            for g in range(n_objects):
                gain = sum(
                    (w_plus if X[g, m] else -w_minus)
                    for m in range(n_attributes)
                    if candidate[m] and not covered[g, m]
                )
                if gain > 0:
                    total += gain
                    users.append(g)
            if best is None or total > best[0]:
                best = (total, candidate, users)
        if best[0] <= 0:
            break
        basis.append(best[1])
        for g in best[2]:
            covered[g] |= best[1]
    return basis


# ---------------------------------------------------------------------------
# GreConD
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(60))
def test_grecond_matches_reference(seed):
    X = _random_data(seed)
    model = GreConD().fit(X)
    expected = _reference_grecond(X)

    assert [(set(c.extent), set(c.intent)) for c in model.factors_] == [
        (set(e), set(i)) for e, i in expected
    ]


@pytest.mark.parametrize("seed", range(30))
def test_grecond_factors_are_concepts_and_rebuild_exactly(seed):
    X = _random_data(seed, 20, 10)
    context = FormalContext.from_array(X)
    model = GreConD().fit(X)

    for factor in model.factors_:
        assert context.object_derivation(factor.extent) == factor.intent
        assert context.attribute_derivation(factor.intent) == factor.extent
    assert (model.inverse_transform(model.transform(X)) == X).all()
    assert model.coverage_ == 1.0 and model.reconstruction_error_ == 0
    # On training data, memberships are exactly the factors' extents.
    usage = model.transform(X)
    for k, factor in enumerate(model.factors_):
        assert set(np.flatnonzero(usage[:, k])) == set(factor.extent)


@pytest.mark.parametrize("seed", range(20))
def test_grecond_coverage_and_component_limits(seed):
    X = _random_data(seed, 20, 10)
    partial = GreConD(min_coverage=0.5).fit(X)
    assert partial.coverage_ >= 0.5
    assert partial.n_components_ == len(_reference_grecond(X, min_coverage=0.5))

    capped = GreConD(n_components=2).fit(X)
    assert capped.n_components_ <= 2
    assert capped.components_.tolist() == GreConD().fit(X).components_[:2].tolist()


def test_grecond_without_ones():
    model = GreConD().fit(np.zeros((3, 2), dtype=bool))
    assert model.n_components_ == 0
    assert model.transform(np.zeros((2, 2))).shape == (2, 0)
    assert model.coverage_ == 1.0


# ---------------------------------------------------------------------------
# ASSO
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("threshold", [0.4, 0.7, 1.0])
@pytest.mark.parametrize("seed", range(30))
def test_asso_matches_reference(seed, threshold):
    X = _random_data(seed)
    model = ASSO(threshold=threshold).fit(X)
    expected = _reference_asso(X, threshold)

    assert model.components_.tolist() == [b.tolist() for b in expected]


@pytest.mark.parametrize("seed", range(20))
def test_asso_weights_and_component_limit(seed):
    X = _random_data(seed, 20, 10)
    model = ASSO(n_components=2, positive_weight=1.0, negative_weight=3.0).fit(X)
    expected = _reference_asso(X, 0.7, n_components=2, w_plus=1.0, w_minus=3.0)

    assert model.components_.tolist() == [b.tolist() for b in expected]
    rebuilt = model.inverse_transform(model.transform(X))
    assert model.reconstruction_error_ == int((rebuilt != X).sum())


def test_asso_covers_with_fewer_broader_factors():
    # Two blocks, each with one missing cell: ASSO covers each block with one
    # factor (accepting a 0), GreConD needs more concept factors.
    X = np.array(
        [[1, 1, 1, 0, 0, 0], [1, 1, 1, 0, 0, 0], [1, 1, 0, 0, 0, 0],
         [0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 1, 1], [0, 0, 0, 0, 1, 1]],
        dtype=bool,
    )
    asso = ASSO(threshold=0.6).fit(X)
    assert asso.n_components_ == 2
    assert GreConD().fit(X).n_components_ > 2


# ---------------------------------------------------------------------------
# scikit-learn behaviour
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model", [GreConD(), ASSO()])
def test_output_formats_and_feature_names(model):
    frame = pd.DataFrame(_random_data(3, 15, 6).astype(int))
    frame.columns = [f"m{i}" for i in range(frame.shape[1])]
    fitted = model.fit(frame)
    prefix = type(model).__name__.lower()
    names = fitted.get_feature_names_out()
    assert list(names) == [f"{prefix}{k}" for k in range(fitted.n_components_)]

    as_frame = model.set_params(output="dataframe").transform(frame)
    assert list(as_frame.columns) == list(names)
    as_sparse = model.set_params(output="sparse").transform(frame)
    assert sparse.issparse(as_sparse)
    assert (as_sparse.toarray() == as_frame.to_numpy()).all()


@pytest.mark.parametrize("model", [GreConD(), ASSO()])
def test_in_a_pipeline_with_set_output(model):
    from sklearn.preprocessing import FunctionTransformer

    X = pd.DataFrame(_random_data(4, 20, 6).astype(int))
    X.columns = [f"m{i}" for i in range(X.shape[1])]
    binarize = FunctionTransformer(lambda d: d > 0, feature_names_out="one-to-one")
    pipe = make_pipeline(binarize, model)
    out = pipe.set_output(transform="pandas").fit_transform(X)
    assert isinstance(out, pd.DataFrame)


@pytest.mark.parametrize(
    "model, message",
    [
        (GreConD(n_components=0), "n_components"),
        (GreConD(min_coverage=0), "min_coverage"),
        (GreConD(output="list"), "output"),
        (ASSO(threshold=1.5), "threshold"),
        (ASSO(negative_weight=0), "negative_weight"),
    ],
)
def test_invalid_parameters(model, message):
    with pytest.raises(ValueError, match=message):
        model.fit(np.eye(3))


def test_inverse_transform_checks_the_shape():
    model = GreConD().fit(np.eye(3))
    with pytest.raises(ValueError, match="columns"):
        model.inverse_transform(np.ones((2, 5)))
