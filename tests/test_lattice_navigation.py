import itertools
import warnings

import numpy as np
import pytest

from conceptflow import ConceptLattice, FormalContext
from conceptflow.algorithms import enumerate_concepts, enumeration, is_cover
from conceptflow.algorithms.enumeration import ConceptLimitExceeded, LargeLatticeWarning


def random_context(seed, max_objects=9, max_attributes=6):
    rng = np.random.default_rng(seed)
    g = int(rng.integers(1, max_objects))
    m = int(rng.integers(0, max_attributes))
    return FormalContext.from_array(
        rng.random((g, m)) < rng.uniform(0.2, 0.8),
        objects=[f"g{i}" for i in range(g)],
        attributes=[f"m{j}" for j in range(m)],
    )


@pytest.mark.parametrize("seed", range(25))
def test_join_and_meet_match_definitions(seed):
    lattice = ConceptLattice.from_context(random_context(seed))
    concepts = lattice.concepts

    for a, b in itertools.product(concepts, repeat=2):
        above = [c for c in concepts if a.extent <= c.extent and b.extent <= c.extent]
        below = [c for c in concepts if c.extent <= a.extent and c.extent <= b.extent]
        assert lattice.join(a, b) == min(above, key=lambda c: len(c.extent))
        assert lattice.meet(a, b) == max(below, key=lambda c: len(c.extent))


@pytest.mark.parametrize("seed", range(25))
def test_object_and_attribute_concepts(seed):
    ctx = random_context(seed)
    lattice = ConceptLattice.from_context(ctx)

    for g, name in enumerate(ctx.objects):
        expected = min(
            (c for c in lattice.concepts if g in c.extent), key=lambda c: len(c.extent)
        )
        assert lattice.object_concept(g) == expected == lattice.object_concept(name)

    for m, name in enumerate(ctx.attributes):
        expected = max(
            (c for c in lattice.concepts if m in c.intent), key=lambda c: len(c.extent)
        )
        assert lattice.attribute_concept(m) == expected == lattice.attribute_concept(name)


@pytest.mark.parametrize("seed", range(15))
def test_atoms_coatoms_and_principal_sets(seed):
    lattice = ConceptLattice.from_context(random_context(seed))
    concepts = lattice.concepts

    assert lattice.atoms() == {c for c in concepts if is_cover(lattice.bottom(), c, concepts)}
    assert lattice.coatoms() == {c for c in concepts if is_cover(c, lattice.top(), concepts)}

    for c in concepts:
        assert lattice.upset(c) == {d for d in concepts if c.extent <= d.extent}
        assert lattice.downset(c) == {d for d in concepts if d.extent <= c.extent}


def test_concept_from_objects_and_attributes():
    ctx = FormalContext.from_array(
        [[1, 1, 0], [1, 0, 1], [1, 1, 1]], attributes=["a", "b", "c"]
    )
    lattice = ConceptLattice.from_context(ctx)

    concept = lattice.concept_from_attributes(["b"])
    assert concept.extent == {0, 2}
    assert concept.intent == {0, 1}
    assert lattice.concept_from_objects(["g0", "g1"]).intent == {0}

    with pytest.raises(ValueError):
        lattice.join()


# ---------------------------------------------------------------------------
# Iceberg lattices and the size guard
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(20))
@pytest.mark.parametrize("method", ["nextclosure", "closebyone", "bruteforce"])
def test_iceberg_is_the_filtered_lattice_in_the_same_order(seed, method):
    ctx = random_context(seed, max_objects=12, max_attributes=7)
    full = enumerate_concepts(ctx, method)

    top = max(full, key=lambda c: len(c.extent))
    for min_support in [1, 2, 3, 0.5, ctx.n_objects, ctx.n_objects + 1]:
        threshold = enumeration._resolve_min_support(min_support, ctx.n_objects)
        # The top concept is always included, even if it is not frequent.
        expected = [c for c in full if len(c.extent) >= threshold or c == top]
        assert enumerate_concepts(ctx, method, min_support=min_support) == expected


@pytest.mark.parametrize("seed", range(15))
def test_iceberg_edges_are_the_cover_relation_of_the_frequent_concepts(seed):
    lattice = ConceptLattice.from_context(random_context(seed, 12, 7), min_support=2)

    assert set(lattice.edges) == {
        (a, b)
        for a in lattice.concepts
        for b in lattice.concepts
        if is_cover(a, b, lattice.concepts)
    }


def test_is_iceberg_only_when_something_was_cut_off():
    all_ones = FormalContext.from_array(np.ones((3, 2)))
    assert not ConceptLattice.from_context(all_ones, min_support=1).is_iceberg

    ctx = FormalContext.from_array([[1, 0], [0, 1], [1, 1]])
    assert ConceptLattice.from_context(ctx, min_support=2).is_iceberg


def test_top_is_kept_when_nothing_is_frequent():
    ctx = FormalContext.from_array([[1, 0], [0, 1]])
    lattice = ConceptLattice.from_context(ctx, min_support=5)

    assert lattice.concepts == (lattice.top(),)
    assert lattice.coatoms() == frozenset()


def test_concepts_from_another_lattice_are_rejected():
    lattice = ConceptLattice.from_context(FormalContext.from_array([[1, 0], [0, 1]]))
    stranger = ConceptLattice.from_context(FormalContext.from_array([[1, 1, 1]])).top()

    for method in [lattice.parents, lattice.children, lattice.upset, lattice.downset]:
        with pytest.raises(ValueError, match="not in this lattice"):
            method(stranger)


def test_iceberg_navigation_errors_are_explicit():
    # a holds for g0..g5, b for g4..g9: both frequent, their meet {g4, g5} not.
    incidence = np.zeros((10, 2), dtype=bool)
    incidence[:6, 0] = True
    incidence[4:, 1] = True
    ctx = FormalContext.from_array(incidence, attributes=["a", "b"])
    iceberg = ConceptLattice.from_context(ctx, min_support=6)

    a, b = iceberg.attribute_concept("a"), iceberg.attribute_concept("b")
    assert iceberg.join(a, b) == iceberg.top()

    with pytest.raises(ValueError, match="iceberg"):
        iceberg.bottom()
    with pytest.raises(ValueError, match="iceberg"):
        iceberg.atoms()
    with pytest.raises(ValueError, match="min_support=6"):
        iceberg.meet(a, b)


def test_large_lattice_warns_once_and_continues(monkeypatch):
    monkeypatch.setattr(enumeration, "LARGE_LATTICE_WARNING_SIZE", 20)
    ctx = FormalContext.from_array(np.random.default_rng(1).random((30, 10)) < 0.5)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        concepts = enumerate_concepts(ctx)
        expected = enumerate_concepts(ctx, "bruteforce")

    ours = [w for w in caught if issubclass(w.category, LargeLatticeWarning)]
    assert len(ours) == 2  # one per enumeration
    # The warning points at the caller's code, not at ConceptFlow internals.
    assert all(w.filename == __file__ for w in ours)
    assert len(concepts) == len(expected)


@pytest.mark.parametrize("method", ["nextclosure", "closebyone", "bruteforce"])
def test_max_concepts_is_an_explicit_hard_limit(method):
    ctx = FormalContext.from_array(np.random.default_rng(1).random((20, 8)) < 0.5)

    with pytest.raises(ConceptLimitExceeded, match="min_support"):
        ConceptLattice.from_context(ctx, algorithm=method, max_concepts=10)

    with pytest.raises(ValueError):
        ConceptLattice.from_context(ctx, max_concepts=0)
