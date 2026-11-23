"""
ConceptFlow against the other Python FCA libraries: fcapy, caspailleur (by
fcapy's author, for keys and implication bases) and concepts.

They are independent implementations, so agreeing with them on many random
contexts catches mistakes that a reference written by the same author might
share. fcapy and caspailleur come with DimFlux; concepts is in the ``dev``
extra (``pip install -e ".[dev]"``).

caspailleur's ``mine_concepts`` and ``mine_implications`` use scikit-mine's
LCM, which fails with current scikit-learn; its lower-level functions (Lindig
intents, keys, pseudo-intents, proper premises) are used instead.

The libraries need at least one object (concepts also one attribute), so
these contexts have; the empty cases are covered by test_properties_core.py.
"""

import warnings

import numpy as np
import pytest

from conceptflow import ConceptLattice, FormalContext
from conceptflow.algorithms import compute_minimal_generators
from conceptflow.io import read_cxt, read_json, write_cxt, write_json
from conceptflow.metrics import concept_stability, stability_bounds

SEEDS = range(40)


def random_context(seed):
    rng = np.random.default_rng(seed)
    g, m = int(rng.integers(1, 10)), int(rng.integers(1, 8))
    incidence = rng.random((g, m)) < rng.uniform(0.1, 0.9)
    if seed % 5 == 0:  # duplicated rows and columns
        incidence = incidence[rng.integers(0, g, g)][:, rng.integers(0, m, m)]
    return FormalContext.from_array(incidence)


def as_names(lattice):
    """Concepts as (object names, attribute names), comparable across libraries."""
    context = lattice.context
    return {
        (frozenset(context.objects[i] for i in c.extent),
         frozenset(context.attributes[j] for j in c.intent))
        for c in lattice.concepts
    }


def named_covers(lattice):
    context = lattice.context

    def names(concept):
        return frozenset(context.objects[i] for i in concept.extent)

    return {(names(lower), names(upper)) for lower, upper in lattice.edges}


# ---------------------------------------------------------------------------
# fcapy
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fcapy():
    from fcapy.context import FormalContext as FcapyContext
    from fcapy.lattice import ConceptLattice as FcapyLattice

    def build(context):
        fc = FcapyContext(
            data=np.asarray(context.incidence).tolist(),
            object_names=list(context.objects),
            attribute_names=list(context.attributes),
        )
        return fc, FcapyLattice.from_context(fc)

    return build


@pytest.mark.parametrize("seed", SEEDS)
def test_concepts_and_covers_agree_with_fcapy(fcapy, seed):
    context = random_context(seed)
    lattice = ConceptLattice.from_context(context)
    fc, theirs = fcapy(context)
    assert {(frozenset(c.extent), frozenset(c.intent)) for c in theirs} == as_names(lattice)
    their_covers = {
        (frozenset(theirs[child].extent), frozenset(theirs[parent].extent))
        for parent, children in theirs.children_dict.items()
        for child in children
    }
    assert their_covers == named_covers(lattice)


@pytest.mark.parametrize("seed", SEEDS)
def test_stability_and_its_bounds_agree_with_fcapy(fcapy, seed):
    context = random_context(seed)
    lattice = ConceptLattice.from_context(context)
    fc, theirs = fcapy(context)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # "stability takes exponential time"
        theirs.calc_concepts_measures("stability", fc)
        theirs.calc_concepts_measures("stability_bounds")
    by_extent = {frozenset(c.extent): c.measures for c in theirs}
    stability = concept_stability(lattice)
    lower, upper = stability_bounds(lattice)
    for i, concept in enumerate(lattice.concepts):
        measures = by_extent[frozenset(context.objects[g] for g in concept.extent)]
        assert stability[i] == pytest.approx(measures["Stab"])
        # fcapy leaves the lower bound unclipped (it can be negative).
        assert lower[i] == pytest.approx(max(0.0, measures["LStab"]))
        assert upper[i] == pytest.approx(measures["UStab"])


@pytest.mark.parametrize("seed", SEEDS)
def test_shortest_keys_are_fcapys_minimum_generators(fcapy, seed):
    context = random_context(seed)
    fc, _ = fcapy(context)
    for concept, keys in compute_minimal_generators(context).items():
        shortest = min(len(k) for k in keys)
        ours = {frozenset(context.attributes[j] for j in k) for k in keys if len(k) == shortest}
        intent = [context.attributes[j] for j in sorted(concept.intent)]
        theirs = {frozenset(g) for g in fc.get_minimal_generators(intent)}
        assert theirs == ours


@pytest.mark.parametrize("seed", range(10))
def test_files_are_exchanged_with_fcapy(fcapy, seed, tmp_path):
    from fcapy.context import FormalContext as FcapyContext

    context = random_context(seed).rename(
        objects=lambda name: f"object {name}", attributes=lambda name: f"attr {name}"
    )
    fc, _ = fcapy(context)

    for suffix, write, read in (("cxt", write_cxt, read_cxt), ("json", write_json, read_json)):
        ours = tmp_path / f"ours.{suffix}"
        write(context, ours)
        loaded = getattr(FcapyContext, f"read_{suffix}")(str(ours))
        assert list(loaded.object_names) == list(context.objects)
        assert list(loaded.attribute_names) == list(context.attributes)
        assert np.array_equal(np.asarray(loaded.data.to_list(), dtype=bool), context.incidence)

        theirs = tmp_path / f"theirs.{suffix}"
        getattr(fc, f"write_{suffix}")(str(theirs))
        assert read(theirs) == context


# ---------------------------------------------------------------------------
# concepts
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def concepts_lattice():
    import concepts

    def build(context):
        theirs = concepts.Context(
            list(context.objects), list(context.attributes),
            np.asarray(context.incidence).tolist(),
        )
        return theirs, theirs.lattice

    return build


@pytest.mark.parametrize("seed", SEEDS)
def test_concepts_covers_meets_and_joins_agree_with_concepts(concepts_lattice, seed):
    context = random_context(seed)
    lattice = ConceptLattice.from_context(context)
    _, theirs = concepts_lattice(context)
    assert {(frozenset(c.extent), frozenset(c.intent)) for c in theirs} == as_names(lattice)
    their_covers = {
        (frozenset(lower.extent), frozenset(c.extent))
        for c in theirs for lower in c.lower_neighbors
    }
    assert their_covers == named_covers(lattice)

    ours = {frozenset(context.objects[g] for g in c.extent): c for c in lattice.concepts}
    their_concepts = list(theirs)
    for a in their_concepts[::3]:
        for b in their_concepts[::2]:
            x, y = ours[frozenset(a.extent)], ours[frozenset(b.extent)]
            join = lattice.join(x, y)
            meet = lattice.meet(x, y)
            assert frozenset(context.objects[g] for g in join.extent) == frozenset(
                (a | b).extent
            )
            assert frozenset(context.objects[g] for g in meet.extent) == frozenset(
                (a & b).extent
            )


@pytest.mark.parametrize("seed", range(10))
def test_cxt_files_are_exchanged_with_concepts(concepts_lattice, seed, tmp_path):
    import concepts

    context = random_context(seed)
    theirs, _ = concepts_lattice(context)
    theirs.tofile(str(tmp_path / "theirs.cxt"), frmat="cxt")
    assert read_cxt(tmp_path / "theirs.cxt") == context

    write_cxt(context, tmp_path / "ours.cxt")
    loaded = concepts.Context.fromfile(str(tmp_path / "ours.cxt"), frmat="cxt")
    assert list(loaded.objects) == list(context.objects)
    assert list(loaded.properties) == list(context.attributes)
    assert np.array_equal(np.asarray(loaded.bools, dtype=bool), context.incidence)


# ---------------------------------------------------------------------------
# caspailleur
# ---------------------------------------------------------------------------


def caspailleur_view(context):
    """caspailleur's intents (Lindig, smallest first), covers, keys and bases."""
    import caspailleur
    from bitarray import frozenbitarray
    from caspailleur.implication_bases import (
        iter_proper_premises_via_keys,
        list_pseudo_intents_via_keys,
    )
    from caspailleur.mine_equivalence_classes import list_intents_via_Lindig
    from caspailleur.order import inverse_order, sort_intents_inclusion

    incidence = np.asarray(context.incidence)
    rows = [frozenbitarray(row.tolist()) for row in incidence]
    columns = [frozenbitarray(column.tolist()) for column in incidence.T]
    intents = sorted(
        (frozenbitarray(i) for i in list_intents_via_Lindig(rows, columns)),
        key=lambda bits: (bits.count(), bits.to01()),
    )
    keys = caspailleur.list_keys(intents)
    parents, transitive = sort_intents_inclusion(intents, return_transitive_order=True)
    return {
        "intents": intents,
        "keys": keys,
        "passkeys": caspailleur.list_passkeys(intents),
        "pseudo_intents": list_pseudo_intents_via_keys(keys.items(), intents),
        "proper_premises": list(iter_proper_premises_via_keys(intents, keys)),
        "parents": parents,  # parents[i]: the smallest intents above intent i
        "children": inverse_order(parents),  # the largest intents below it
        "n_comparable": sum(row.count() for row in transitive),
    }


def as_set(bits):
    return frozenset(bits.search(1))


@pytest.mark.parametrize("seed", SEEDS)
def test_intents_and_covers_agree_with_caspailleur(seed):
    context = random_context(seed)
    lattice = ConceptLattice.from_context(context)
    theirs = caspailleur_view(context)
    intents = [as_set(bits) for bits in theirs["intents"]]
    assert set(intents) == {c.intent for c in lattice.concepts}
    assert len(intents) == lattice.n_concepts
    # parents[i]: the smallest intents above intent i, i.e. its lower covers.
    their_covers = {
        (intents[j], intents[i])
        for i, row in enumerate(theirs["parents"]) for j in row.search(1)
    }
    assert their_covers == {(lower.intent, upper.intent) for lower, upper in lattice.edges}


@pytest.mark.parametrize("seed", SEEDS)
def test_keys_and_passkeys_agree_with_caspailleur(seed):
    context = random_context(seed)
    theirs = caspailleur_view(context)
    intents = [as_set(bits) for bits in theirs["intents"]]
    ours = compute_minimal_generators(context)
    assert {as_set(k): intents[i] for k, i in theirs["keys"].items()} == {
        key: concept.intent for concept, keys in ours.items() for key in keys
    }
    # Passkeys: the keys of the smallest size of their intent.
    assert {as_set(k): intents[i] for k, i in theirs["passkeys"].items()} == {
        key: concept.intent for concept, keys in ours.items()
        for key in keys if len(key) == min(map(len, keys))
    }


# caspailleur 0.2.2 lists {m0, ..., m5} of seed 4 as a pseudo-intent, but the
# smaller pseudo-intent {m0, m2, m3} closes to {m0, m2, m3, m6}, which is not
# inside it, so by definition it is not one (ConceptFlow agrees with the
# brute-force definition in test_properties_core.py). Strict: this test
# says so when caspailleur is fixed.
CASPAILLEUR_PSEUDO_INTENT_BUG = pytest.mark.xfail(
    strict=True, reason="caspailleur lists a set that is not a pseudo-intent"
)


@pytest.mark.parametrize(
    "seed",
    [pytest.param(s, marks=CASPAILLEUR_PSEUDO_INTENT_BUG) if s == 4 else s for s in SEEDS],
)
def test_implication_bases_agree_with_caspailleur(seed):
    from conceptflow.algorithms import compute_canonical_basis, compute_proper_premise_basis

    context = random_context(seed)
    theirs = caspailleur_view(context)
    intents = [as_set(bits) for bits in theirs["intents"]]
    canonical = compute_canonical_basis(context)
    assert {as_set(p): intents[i] for p, i in theirs["pseudo_intents"]} == {
        imp.premise: imp.premise | imp.conclusion for imp in canonical
    }
    proper = compute_proper_premise_basis(context)
    assert {as_set(p): intents[i] for p, i in theirs["proper_premises"]} == {
        imp.premise: context.closure(imp.premise) for imp in proper
    }


@pytest.mark.parametrize("seed", SEEDS)
def test_delta_stability_and_lattice_indices_agree_with_caspailleur(seed):
    import caspailleur
    from caspailleur.indices import delta_stability_by_extents

    from conceptflow.metrics import delta_measure, distributivity_index, linearity_index

    context = random_context(seed)
    lattice = ConceptLattice.from_context(context)
    theirs = caspailleur_view(context)
    n = lattice.n_concepts

    extents = sorted(lattice.concepts, key=lambda c: (len(c.extent), sorted(c.extent)))
    bits = [frozenbitarray_of(c.extent, context.n_objects) for c in extents]
    their_delta = dict(zip([c.extent for c in extents], delta_stability_by_extents(bits)))
    ours = dict(zip([c.extent for c in lattice.concepts], delta_measure(lattice)))
    assert their_delta == ours

    if n >= 2:
        assert linearity_index(lattice) == pytest.approx(
            caspailleur.linearity_index(theirs["n_comparable"], n)
        )
        assert distributivity_index(lattice) == pytest.approx(
            caspailleur.distributivity_index(
                theirs["intents"], theirs["children"], theirs["n_comparable"]
            )
        )


def frozenbitarray_of(indices, size):
    from bitarray import frozenbitarray

    return frozenbitarray([i in indices for i in range(size)])
