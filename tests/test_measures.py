import itertools
import math

import numpy as np
import pytest

from conceptflow import ConceptLattice, FormalContext
from conceptflow.algorithms import is_cover
from conceptflow.cluster import ConceptLatticeEstimator
from conceptflow.metrics import (
    concept_entropy,
    concept_information_gain,
    concept_stability,
    delta_measure,
    distributivity_index,
    linearity_index,
    stability_bounds,
)


def random_lattice(seed, min_support=None):
    rng = np.random.default_rng(seed)
    g, m = int(rng.integers(1, 9)), int(rng.integers(0, 6))
    ctx = FormalContext.from_array(rng.random((g, m)) < rng.uniform(0.2, 0.8))
    return ConceptLattice.from_context(ctx, min_support=min_support)


def entropy(labels):
    if len(labels) == 0:
        return 0.0
    _, counts = np.unique(labels, return_counts=True)
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum())


@pytest.mark.parametrize("seed", range(30))
def test_stability_matches_definition_and_bounds_contain_it(seed):
    lattice = random_lattice(seed)
    ctx = lattice.context
    stability = concept_stability(lattice)
    lower, upper = stability_bounds(lattice)

    for i, concept in enumerate(lattice.concepts):
        extent = sorted(concept.extent)
        generating = sum(
            1
            for r in range(len(extent) + 1)
            for subset in itertools.combinations(extent, r)
            if ctx.object_derivation(subset) == concept.intent
        )
        assert math.isclose(stability[i], generating / 2 ** len(extent))
        assert lower[i] - 1e-12 <= stability[i] <= upper[i] + 1e-12


@pytest.mark.parametrize("seed", range(30))
def test_delta_measure_matches_definition(seed):
    lattice = random_lattice(seed)
    concepts = lattice.concepts
    delta = delta_measure(lattice)

    for i, concept in enumerate(concepts):
        children = [d for d in concepts if is_cover(d, concept, concepts)]
        largest = max((len(d.extent) for d in children), default=0)
        assert delta[i] == len(concept.extent) - largest


@pytest.mark.parametrize("seed", range(10))
def test_delta_and_bounds_work_on_iceberg_lattices(seed):
    full = random_lattice(seed)
    iceberg = random_lattice(seed, min_support=2)
    full_delta = dict(zip(full.concepts, delta_measure(full)))

    for concept, value in zip(iceberg.concepts, delta_measure(iceberg)):
        assert value == full_delta[concept]

    stability_bounds(iceberg)
    if iceberg.is_iceberg:
        with pytest.raises(ValueError, match="full lattice"):
            concept_stability(iceberg)


@pytest.mark.parametrize("seed", range(20))
def test_entropy_and_information_gain(seed):
    lattice = random_lattice(seed)
    y = np.random.default_rng(seed).integers(0, 3, lattice.context.n_objects)
    entropies = concept_entropy(lattice, y)
    gains = concept_information_gain(lattice, y)

    for i, concept in enumerate(lattice.concepts):
        inside = np.isin(np.arange(len(y)), sorted(concept.extent))
        assert math.isclose(entropies[i], entropy(y[inside]), abs_tol=1e-12)
        share = inside.mean()
        expected = entropy(y) - (
            share * entropy(y[inside]) + (1 - share) * entropy(y[~inside])
        )
        assert math.isclose(gains[i], expected, abs_tol=1e-12)

    with pytest.raises(ValueError, match="one label per object"):
        concept_entropy(lattice, y[:-1] if len(y) > 1 else [0, 0])


@pytest.mark.parametrize("seed", range(40))
def test_lattice_indices_match_pairwise_definitions(seed):
    lattice = random_lattice(seed)
    pairs = list(itertools.combinations(lattice.concepts, 2))
    if not pairs:
        assert linearity_index(lattice) == distributivity_index(lattice) == 1.0
        return

    intents = {c.intent for c in lattice.concepts}
    comparable = sum(a.extent <= b.extent or b.extent <= a.extent for a, b in pairs)
    distributive = sum((a.intent | b.intent) in intents for a, b in pairs)

    assert math.isclose(linearity_index(lattice), comparable / len(pairs))
    assert math.isclose(distributivity_index(lattice), distributive / len(pairs))


def test_chain_is_linear_and_distributive():
    chain = ConceptLattice.from_context(FormalContext.from_array(np.tril(np.ones((5, 5)))))

    assert linearity_index(chain) == 1.0
    assert distributivity_index(chain) == 1.0


# ---------------------------------------------------------------------------
# ConceptLatticeEstimator selection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("select", ["stability", "delta", "support"])
def test_estimator_keeps_the_best_concepts(select):
    X = np.random.default_rng(3).random((30, 6)) < 0.5
    est = ConceptLatticeEstimator(select=select, n_concepts=5).fit(X)
    lattice = est.get_lattice()

    measure = {
        "stability": concept_stability,
        "delta": delta_measure,
        "support": lambda lat: np.array([len(c.extent) for c in lat.concepts]),
    }[select]
    score = dict(zip(lattice.concepts, measure(lattice)))

    assert len(est.concepts_) == 5
    assert est.transform(X).shape == (30, 5)
    kept = min(score[c] for c in est.concepts_)
    assert all(score[c] <= kept for c in lattice.concepts if c not in est.concepts_)


def test_estimator_selection_validation():
    X = np.random.default_rng(0).random((10, 4)) < 0.5

    with pytest.raises(ValueError, match="n_concepts"):
        ConceptLatticeEstimator(select="delta").fit(X)
    # n_concepts is ignored without select, so grids can mix both.
    assert ConceptLatticeEstimator(n_concepts=3).fit(X).transform(X).shape[1] > 3
    with pytest.raises(ValueError, match="select"):
        ConceptLatticeEstimator(select="nope", n_concepts=3).fit(X)
    with pytest.raises(ValueError, match="full lattice"):
        ConceptLatticeEstimator(select="stability", n_concepts=3, min_support=2).fit(X)


def test_estimator_iceberg_features():
    X = np.random.default_rng(1).random((40, 6)) < 0.5
    est = ConceptLatticeEstimator(min_support=10).fit(X)

    assert all(len(c.extent) >= 10 for c in est.concepts_)
    assert est.get_lattice().is_iceberg
