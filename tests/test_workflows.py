"""
Whole workflows, where the parts are combined as users combine them.

The oracle tests check each operation exhaustively on small contexts; these
run complete chains on data too large for brute force, and check the
results through their defining properties instead.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline

from conceptflow import ConceptLattice
from conceptflow.algorithms import compute_canonical_basis, enumerate_concepts
from conceptflow.cluster import ConceptLatticeEstimator
from conceptflow.datasets import list_contexts, load_context
from conceptflow.decomposition import GreConD
from conceptflow.preprocessing import ConceptualScaler, NominalScale, ThresholdScale
from conceptflow.rules import ImplicationOutlierDetector

# Bundled contexts beyond exhaustive search (more than 10 attributes).
MEDIUM = [name for name, row in list_contexts().iterrows() if row["attributes"] > 10]


def _closure_under(basis, attributes):
    """Close an attribute set under a list of implications."""
    closed, changed = set(attributes), True
    while changed:
        changed = False
        for implication in basis:
            if implication.premise <= closed and not implication.conclusion <= closed:
                closed |= implication.conclusion
                changed = True
    return frozenset(closed)


@pytest.mark.parametrize("name", MEDIUM)
def test_bundled_context_to_lattice_basis_and_factors(name):
    context = load_context(name)
    incidence = np.asarray(context.incidence)

    # Lattice: both algorithms find the same concepts, all of them closed.
    lattice = ConceptLattice.from_context(context)
    other = {c.intent for c in enumerate_concepts(context, method="closebyone")}
    assert {c.intent for c in lattice.concepts} == other
    for concept in lattice.concepts:
        assert context.attribute_derivation(concept.intent) == concept.extent
        assert context.object_derivation(concept.extent) == concept.intent

    # Canonical basis: both algorithms agree, every implication holds, and
    # closing any attribute set with the basis gives its closure X''.
    basis = compute_canonical_basis(context)
    assert {(i.premise, i.conclusion) for i in basis} == {
        (i.premise, i.conclusion) for i in compute_canonical_basis(context, "nextclosure")
    }
    for implication in basis:
        extent = context.attribute_derivation(implication.premise)
        assert implication.conclusion <= context.object_derivation(extent)
    rng = np.random.default_rng(0)
    for _ in range(200):
        size = int(rng.integers(0, 4))
        attributes = frozenset(rng.choice(context.n_attributes, size, replace=False).tolist())
        assert _closure_under(basis, attributes) == context.closure(attributes)

    # Factorization: GreConD's factors are concepts and rebuild the data.
    model = GreConD().fit(incidence.astype(int))
    for factor in model.factors_:
        assert context.object_derivation(factor.extent) == factor.intent
    rebuilt = model.inverse_transform(model.transform(incidence.astype(int)))
    assert np.array_equal(rebuilt, incidence)
    assert model.coverage_ == 1.0 and model.reconstruction_error_ == 0


def _table(n, seed):
    rng = np.random.default_rng(seed)
    colour = rng.choice(["red", "green", "blue"], n)
    size = rng.uniform(0, 10, n).round(1)
    shape = rng.choice(["round", "square"], n)
    label = ((colour == "red") & (size > 5)) | (shape == "square") & (colour == "blue")
    table = pd.DataFrame({"colour": colour, "size": size, "shape": shape},
                         index=[f"item{i}" for i in range(n)])
    return table, label.astype(int)


SCALES = [NominalScale("colour"), ThresholdScale("size", [2.5, 5, 7.5]), NominalScale("shape")]


def test_raw_table_to_tuned_classifier_with_unseen_values():
    train, y_train = _table(120, seed=0)
    test, _ = _table(20, seed=1)
    test.loc["item0", "colour"] = "purple"  # never seen during fit

    pipeline = make_pipeline(
        ConceptualScaler(scales=SCALES, handle_unknown="ignore"),
        ConceptLatticeEstimator(select="stability"),
        LogisticRegression(max_iter=1000),
    )
    search = GridSearchCV(
        pipeline, {"conceptlatticeestimator__n_concepts": [5, 20]}, cv=3
    ).fit(train, y_train)

    assert search.best_params_["conceptlatticeestimator__n_concepts"] in (5, 20)
    assert search.best_score_ > 0.8  # the label is a simple rule of the attributes
    predictions = search.predict(test)
    assert predictions.shape == (20,) and set(predictions) <= {0, 1}


def test_raw_table_to_rules_and_outliers():
    table, _ = _table(80, seed=2)
    context = ConceptualScaler(scales=SCALES, output="context").fit_transform(table)

    # Exact rules found on the scaled table hold on it.
    for implication in compute_canonical_basis(context):
        holders = context.attribute_derivation(implication.premise)
        assert holders <= context.attribute_derivation(implication.conclusion)

    # An object breaking a rule that holds for every training object
    # (size >= 7.5 implies size >= 5) is flagged, and the rule explains it.
    data = context.to_dataframe().astype(int)
    detector = ImplicationOutlierDetector(include_exact=True).fit(data)
    odd = data[data["size>=7.5"] == 1].iloc[:1].copy()
    odd["size>=5"] = 0
    assert detector.predict(odd)[0] == -1
    assert detector.score_samples(odd)[0] < detector.score_samples(data).min()
    explanation = detector.explain(odd)
    assert any("size>=7.5" in premise and "size>=5" in missing
               for premise, missing in zip(explanation["premise"], explanation["missing"]))
