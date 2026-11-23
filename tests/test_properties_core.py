"""
Property-based tests of the mathematical core.

Hypothesis draws formal contexts of every shape: empty ones, contexts
without objects or attributes, duplicated, full and empty rows and columns,
and contexts with more than 64 objects or attributes (more than one bit-set
word). Each property is either a law of Formal Concept Analysis or agreement
with ``tests.property_helpers.Reference``, which is written from the
definitions and shares no code with ConceptFlow. A failing example is shrunk
to a smallest context and printed.
"""

import copy
import pickle

import networkx as nx
import numpy as np
import pytest

pytest.importorskip("hypothesis")

from hypothesis import assume, given  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

from conceptflow import ConceptLattice, FormalContext  # noqa: E402
from conceptflow.algorithms import (  # noqa: E402
    attribute_closure,
    attribute_derivation,
    clarify_context,
    compute_canonical_basis,
    compute_canonical_basis_bruteforce,
    compute_hasse_edges,
    compute_luxenburger_basis,
    compute_minimal_generators,
    compute_proper_premise_basis,
    enumerate_concepts,
    object_derivation,
    reduce_context,
)
from conceptflow.algorithms.enumeration import ConceptLimitExceeded  # noqa: E402
from conceptflow.core import (  # noqa: E402
    apposition,
    complement,
    direct_product,
    direct_sum,
    dual,
    is_scale_measure,
    refines,
    scale_measure,
    subposition,
)
from conceptflow.metrics import (  # noqa: E402
    attribute_set_support,
    concept_levels,
    concept_stability,
    concept_support,
    delta_measure,
    distributivity_index,
    implication_confidence,
    implication_exceptions,
    incomparable_pairs,
    lattice_height,
    lattice_width,
    linearity_index,
    stability_bounds,
)
from tests.property_helpers import (  # noqa: E402
    Reference,
    all_subsets,
    as_masks,
    context_of,
    implication_closure,
    incidences,
    sample_masks,
    submasks,
    to_mask,
    to_set,
    wide_incidences,
)

METHODS = ["nextclosure", "closebyone", "bruteforce"]
any_incidences = st.one_of(incidences(), wide_incidences())


def subset_of(data, size, label):
    if size == 0:
        return frozenset()
    return frozenset(data.draw(st.sets(st.integers(0, size - 1)), label=label))


def lattice_of(incidence, **kwargs):
    return ConceptLattice.from_context(context_of(incidence), **kwargs)


# ---------------------------------------------------------------------------
# Derivation operators
# ---------------------------------------------------------------------------


@given(any_incidences, st.data())
def test_derivations_match_the_definition(incidence, data):
    ref, context = Reference(incidence), context_of(incidence)
    objects = subset_of(data, ref.g, "objects")
    attributes = subset_of(data, ref.m, "attributes")

    assert context.object_derivation(objects) == to_set(ref.intent(to_mask(objects)))
    assert context.attribute_derivation(attributes) == to_set(ref.extent(to_mask(attributes)))
    assert context.closure(attributes) == to_set(ref.closure(to_mask(attributes)))
    # The functional API gives the same.
    assert object_derivation(context, objects) == context.object_derivation(objects)
    assert attribute_derivation(context, attributes) == context.attribute_derivation(attributes)
    assert attribute_closure(context, attributes) == context.closure(attributes)
    # Names and positions are interchangeable.
    assert context.object_derivation(
        [context.objects[i] for i in objects]
    ) == context.object_derivation(objects)
    assert context.attribute_derivation(
        [context.attributes[j] for j in attributes]
    ) == context.attribute_derivation(attributes)


@given(any_incidences, st.data())
def test_derivations_form_a_galois_connection(incidence, data):
    context = context_of(incidence)
    g, m = incidence.shape
    a1, a2 = subset_of(data, g, "A1"), subset_of(data, g, "A2")
    b1 = subset_of(data, m, "B1")

    # A ⊆ B'  <=>  B ⊆ A'
    assert (a1 <= context.attribute_derivation(b1)) == (b1 <= context.object_derivation(a1))
    # Antitone: A1 ⊆ A2 => A2' ⊆ A1'
    assert context.object_derivation(a1 | a2) <= context.object_derivation(a1)
    # (A1 ∪ A2)' = A1' ∩ A2'
    assert context.object_derivation(a1 | a2) == (
        context.object_derivation(a1) & context.object_derivation(a2)
    )
    # A' = A'''
    once = context.object_derivation(a1)
    assert context.object_derivation(context.attribute_derivation(once)) == once
    # '' is a closure operator: extensive, idempotent and monotone.
    closed = context.closure(b1)
    assert b1 <= closed
    assert context.closure(closed) == closed
    b2 = subset_of(data, m, "B2")
    assert context.closure(b1) <= context.closure(b1 | b2)


# ---------------------------------------------------------------------------
# Concepts
# ---------------------------------------------------------------------------


@given(any_incidences, st.sampled_from(METHODS))
def test_enumerated_concepts_are_exactly_the_formal_concepts(incidence, method):
    assume(method != "bruteforce" or incidence.shape[1] <= 10)
    ref = Reference(incidence)
    found = enumerate_concepts(context_of(incidence), method=method)
    assert len(found) == len(set(found))
    assert {(c.extent, c.intent) for c in found} == ref.concept_sets()


@given(any_incidences)
def test_concepts_come_in_the_documented_orders(incidence):
    context = context_of(incidence)
    m = context.n_attributes

    # NextClosure: lectic order of the intents, attribute 0 most significant.
    def lectic(concept):
        return sum(1 << (m - 1 - j) for j in concept.intent)

    keys = [lectic(c) for c in enumerate_concepts(context, method="nextclosure")]
    assert keys == sorted(keys) and len(set(keys)) == len(keys)

    def by_size(concept):
        return (len(concept.extent), sorted(concept.extent), sorted(concept.intent))

    found = enumerate_concepts(context, method="closebyone")
    assert found == sorted(found, key=by_size)
    if m <= 10:
        assert enumerate_concepts(context, method="bruteforce") == found


@given(any_incidences, st.data())
def test_iceberg_concepts_are_exactly_the_frequent_concepts(incidence, data):
    ref, context = Reference(incidence), context_of(incidence)
    g = ref.g
    threshold = data.draw(st.integers(0, g + 1), label="min_support")
    # The top concept is always kept, even above |G| objects.
    expected = {(a, b) for a, b in ref.concept_sets() if len(a) >= threshold or len(a) == g}
    for method in ("nextclosure", "closebyone"):
        found = enumerate_concepts(context, method=method, min_support=threshold)
        assert {(c.extent, c.intent) for c in found} == expected
    # A fraction means ceil(fraction * |G|) objects.
    if g:
        fraction = threshold / g
        if fraction <= 1:
            found = enumerate_concepts(context, min_support=fraction)
            assert {(c.extent, c.intent) for c in found} == expected


@given(incidences(), st.data())
def test_lattice_is_an_iceberg_only_if_concepts_were_left_out(incidence, data):
    context = context_of(incidence)
    full = ConceptLattice.from_context(context)
    threshold = data.draw(st.integers(0, context.n_objects + 1), label="min_support")
    lattice = ConceptLattice.from_context(context, min_support=threshold)
    left_out = full.n_concepts - lattice.n_concepts
    if lattice.is_iceberg:
        assert left_out > 0
        assert lattice.min_support == threshold
    else:
        assert left_out == 0 and lattice == full


@given(incidences(), st.data())
def test_max_concepts_stops_exactly_above_the_limit(incidence, data):
    context = context_of(incidence)
    # Also for iceberg lattices, which are enumerated differently.
    min_support = data.draw(st.sampled_from([None, 1, 2]), label="min_support")
    n = len(enumerate_concepts(context, min_support=min_support))
    limit = data.draw(st.integers(1, n + 1), label="max_concepts")
    method = data.draw(st.sampled_from(METHODS), label="method")
    options = {"method": method, "max_concepts": limit, "min_support": min_support}
    if n > limit:
        with pytest.raises(ConceptLimitExceeded):
            enumerate_concepts(context, **options)
    else:
        assert len(enumerate_concepts(context, **options)) == n


@given(incidences(), st.data())
def test_concepts_follow_permutations_of_objects_and_attributes(incidence, data):
    g, m = incidence.shape
    rows = data.draw(st.permutations(range(g)), label="object order")
    columns = data.draw(st.permutations(range(m)), label="attribute order")
    context = context_of(incidence)
    shuffled = context.subcontext(objects=list(rows), attributes=list(columns))
    row_of = {old: new for new, old in enumerate(rows)}
    column_of = {old: new for new, old in enumerate(columns)}
    expected = {
        (frozenset(row_of[i] for i in c.extent), frozenset(column_of[j] for j in c.intent))
        for c in enumerate_concepts(context)
    }
    assert {(c.extent, c.intent) for c in enumerate_concepts(shuffled)} == expected


@given(incidences())
def test_dual_context_has_the_swapped_concepts(incidence):
    context = context_of(incidence)
    concepts = {(c.extent, c.intent) for c in enumerate_concepts(context)}
    for transposed in (dual(context), context.transpose()):
        assert {(c.intent, c.extent) for c in enumerate_concepts(transposed)} == concepts
    assert dual(dual(context)) == context


@given(incidences())
def test_duplicates_do_not_change_the_number_of_concepts(incidence):
    context = context_of(incidence)
    n = len(enumerate_concepts(context))
    if context.n_objects:
        doubled = context.add_objects(["copy"], context.incidence[:1])
        assert len(enumerate_concepts(doubled)) == n
    if context.n_attributes:
        doubled = context.add_attributes(["copy"], context.incidence[:, :1])
        assert len(enumerate_concepts(doubled)) == n


# ---------------------------------------------------------------------------
# The lattice
# ---------------------------------------------------------------------------


@given(any_incidences)
def test_hasse_edges_are_the_cover_relation(incidence):
    lattice = lattice_of(incidence)
    extents = [to_mask(c.extent) for c in lattice.concepts]
    position = {c: i for i, c in enumerate(lattice.concepts)}
    edges = {(position[lower], position[upper]) for lower, upper in lattice.edges}
    assert len(edges) == len(lattice.edges)
    assert edges == Reference.covers(extents)
    # The generic algorithm (no context) gives the same.
    assert set(compute_hasse_edges(lattice.concepts)) == set(lattice.edges)


@given(incidences(), st.data())
def test_hasse_edges_of_any_family_are_the_covers_of_the_induced_order(incidence, data):
    context = context_of(incidence)
    concepts = enumerate_concepts(context)
    family = data.draw(
        st.lists(st.sampled_from(concepts), unique=True).map(tuple), label="family"
    )
    extents = [to_mask(c.extent) for c in family]
    expected = {(family[i], family[j]) for i, j in Reference.covers(extents)}
    assert set(compute_hasse_edges(family)) == expected
    # Passing the context must not make a partial family look complete.
    assert set(compute_hasse_edges(family, context=context)) == expected


@given(incidences(), st.data())
def test_iceberg_edges_are_the_covers_among_frequent_concepts(incidence, data):
    context = context_of(incidence)
    threshold = data.draw(st.integers(1, max(1, context.n_objects)), label="min_support")
    lattice = ConceptLattice.from_context(context, min_support=threshold)
    extents = [to_mask(c.extent) for c in lattice.concepts]
    expected = {
        (lattice.concepts[i], lattice.concepts[j]) for i, j in Reference.covers(extents)
    }
    assert set(lattice.edges) == expected


@given(any_incidences)
def test_top_bottom_atoms_and_coatoms(incidence):
    lattice = lattice_of(incidence)
    g, m = incidence.shape
    top, bottom = lattice.top(), lattice.bottom()
    assert top.extent == frozenset(range(g))
    assert bottom.intent == frozenset(range(m))
    assert all(c.extent <= top.extent and bottom.extent <= c.extent for c in lattice.concepts)
    extents = [to_mask(c.extent) for c in lattice.concepts]
    covers = Reference.covers(extents)
    position = {c: i for i, c in enumerate(lattice.concepts)}
    assert {position[a] for a in lattice.atoms()} == {
        j for i, j in covers if i == position[bottom]
    }
    assert {position[c] for c in lattice.coatoms()} == {
        i for i, j in covers if j == position[top]
    }


@given(incidences(max_objects=6, max_attributes=6), st.data())
def test_meet_and_join_satisfy_the_lattice_laws(incidence, data):
    lattice = lattice_of(incidence)
    ref = Reference(incidence)
    pick = st.sampled_from(lattice.concepts)
    a, b, c = (data.draw(pick, label=name) for name in "abc")

    join, meet = lattice.join(a, b), lattice.meet(a, b)
    assert to_mask(join.extent) == ref.extent(ref.intent(to_mask(a.extent | b.extent)))
    assert meet.extent == a.extent & b.extent
    assert join == lattice.join(b, a) and meet == lattice.meet(b, a)
    assert lattice.join(a, a) == a and lattice.meet(a, a) == a
    assert lattice.join(a, lattice.meet(a, b)) == a
    assert lattice.meet(a, lattice.join(a, b)) == a
    assert lattice.join(a, lattice.join(b, c)) == lattice.join(lattice.join(a, b), c)
    assert lattice.join(a, b, c) == lattice.join(lattice.join(a, b), c)
    assert lattice.meet(a, lattice.meet(b, c)) == lattice.meet(lattice.meet(a, b), c)
    assert lattice.meet(a, b, c) == lattice.meet(lattice.meet(a, b), c)
    # The join is the least upper bound, the meet the greatest lower bound.
    uppers = [x for x in lattice.concepts if a.extent <= x.extent and b.extent <= x.extent]
    lowers = [x for x in lattice.concepts if x.extent <= a.extent and x.extent <= b.extent]
    assert all(join.extent <= x.extent for x in uppers)
    assert all(x.extent <= meet.extent for x in lowers)


@given(incidences(max_objects=6, max_attributes=6), st.data())
def test_up_and_down_sets_and_neighbours(incidence, data):
    lattice = lattice_of(incidence)
    concept = data.draw(st.sampled_from(lattice.concepts), label="concept")
    assert lattice.upset(concept) == {
        x for x in lattice.concepts if concept.extent <= x.extent
    }
    assert lattice.downset(concept) == {
        x for x in lattice.concepts if x.extent <= concept.extent
    }
    assert lattice.parents(concept) == {u for lo, u in lattice.edges if lo == concept}
    assert lattice.children(concept) == {lo for lo, u in lattice.edges if u == concept}


@given(any_incidences)
def test_basic_theorem_objects_and_attributes_generate_every_concept(incidence):
    lattice = lattice_of(incidence)
    context = lattice.context
    for g in range(context.n_objects):
        gamma = lattice.object_concept(g)
        assert gamma.intent == context.object_derivation([g])
        assert lattice.object_concept(context.objects[g]) == gamma
    for m in range(context.n_attributes):
        mu = lattice.attribute_concept(m)
        assert mu.extent == context.attribute_derivation([m])
        assert lattice.attribute_concept(context.attributes[m]) == mu
    for concept in lattice.concepts:
        # Join-dense object concepts, meet-dense attribute concepts.
        if concept.extent:
            assert lattice.join(*(lattice.object_concept(g) for g in concept.extent)) == concept
        if concept.intent:
            assert lattice.meet(
                *(lattice.attribute_concept(m) for m in concept.intent)
            ) == concept
        assert lattice.concept_from_objects(concept.extent) == concept
        assert lattice.concept_from_attributes(concept.intent) == concept


@given(incidences(), st.data())
def test_iceberg_navigation_stays_in_the_iceberg(incidence, data):
    context = context_of(incidence)
    assume(context.n_objects > 0)
    threshold = data.draw(st.integers(1, context.n_objects), label="min_support")
    lattice = ConceptLattice.from_context(context, min_support=threshold)
    assume(lattice.is_iceberg)
    pick = st.sampled_from(lattice.concepts)
    a, b = data.draw(pick, label="a"), data.draw(pick, label="b")
    # Frequent concepts form an up-set: joins stay inside.
    assert lattice.join(a, b) in lattice.concepts
    if len(a.extent & b.extent) >= threshold:
        assert lattice.meet(a, b).extent == a.extent & b.extent
    else:
        with pytest.raises(ValueError, match="iceberg"):
            lattice.meet(a, b)
    with pytest.raises(ValueError):
        lattice.bottom()


@given(any_incidences)
def test_networkx_graph_is_the_transitive_reduction(incidence):
    lattice = lattice_of(incidence)
    graph = lattice.to_networkx()
    assert nx.is_directed_acyclic_graph(graph)
    assert set(graph.nodes) == set(lattice.concepts)
    assert set(nx.transitive_reduction(graph).edges) == set(graph.edges)
    assert [n for n, d in graph.out_degree() if d == 0] == [lattice.top()]
    assert [n for n, d in graph.in_degree() if d == 0] == [lattice.bottom()]


@given(any_incidences)
def test_lattices_are_equal_whatever_algorithm_and_survive_pickling(incidence):
    context = context_of(incidence)
    lattice = ConceptLattice.from_context(context)
    other = ConceptLattice.from_context(context, algorithm="closebyone")
    assert lattice == other and hash(lattice) == hash(other)
    # Each algorithm's documented order is kept.
    for method in METHODS:
        if method != "bruteforce" or context.n_attributes <= 10:
            built = ConceptLattice.from_context(context, algorithm=method)
            assert list(built.concepts) == enumerate_concepts(context, method=method)
    # Equal lattices need the same context and min_support, not only concepts.
    renamed = context.rename(objects=lambda n: f"x{n}", attributes=lambda n: f"y{n}")
    if context.n_objects or context.n_attributes:
        assert ConceptLattice(renamed, lattice.concepts) != lattice
    assert ConceptLattice(context, lattice.concepts, min_support=1) != lattice
    assert ConceptLattice(context, lattice.concepts[1:]) != lattice
    for copied in (pickle.loads(pickle.dumps(lattice)), copy.deepcopy(lattice)):
        assert copied == lattice
        assert not copied.context.incidence.flags.writeable
        assert set(copied.edges) == set(lattice.edges)


@given(any_incidences)
def test_concept_ids_depend_only_on_the_intent(incidence):
    lattice = lattice_of(incidence)
    ids = [c.stable_id() for c in lattice.concepts]
    assert len(set(ids)) == len(ids)
    signatures = [c.signature() for c in lattice.concepts]
    assert len(set(signatures)) == len(signatures)
    # Reordering and dropping objects keeps the ids of the remaining intents.
    context = lattice.context
    reversed_objects = context.subcontext(objects=list(range(context.n_objects))[::-1])
    reordered = ConceptLattice.from_context(reversed_objects)
    assert {c.stable_id() for c in reordered.concepts} == set(ids)


# ---------------------------------------------------------------------------
# Implications
# ---------------------------------------------------------------------------


@given(incidences(max_attributes=7))
def test_canonical_basis_is_the_stem_base(incidence):
    ref, context = Reference(incidence), context_of(incidence)
    expected = {
        to_set(p): to_set(closure & ~p) for p, closure in ref.pseudo_intents().items()
    }
    for algorithm in ("lincbo", "nextclosure"):
        basis = compute_canonical_basis(context, algorithm=algorithm)
        assert {i.premise: i.conclusion for i in basis} == expected
        assert len(basis) == len(expected)
    # The same implications in the same, documented order.
    assert compute_canonical_basis_bruteforce(context) == compute_canonical_basis(context)


@given(any_incidences, st.data())
def test_canonical_basis_is_sound_complete_and_minimal(incidence, data):
    ref, context = Reference(incidence), context_of(incidence)
    basis = compute_canonical_basis(context)
    assert basis == compute_canonical_basis(context, algorithm="nextclosure")
    assert basis == sorted(basis, key=lambda i: (len(i.premise), sorted(i.premise)))
    masks = as_masks(basis)

    for premise, conclusion in masks:
        assert conclusion and not premise & conclusion
        assert ref.holds(premise, conclusion)
        # Pseudo-closed: not an intent, and contains the closure of every
        # smaller premise.
        assert ref.closure(premise) != premise
        for other, other_conclusion in masks:
            if other != premise and other & premise == other:
                assert (other | other_conclusion) & premise == other | other_conclusion
        # Non-redundant: the other implications do not derive this one.
        rest = [(p, c) for p, c in masks if p != premise]
        assert implication_closure(premise, rest) & conclusion != conclusion

    # Complete: closing under the basis is closing in the context.
    for attributes in sample_masks(data, ref.m):
        assert implication_closure(attributes, masks) == ref.closure(attributes)


@given(any_incidences, st.data())
def test_proper_premise_basis_is_direct_and_its_premises_are_proper(incidence, data):
    ref, context = Reference(incidence), context_of(incidence)
    basis = compute_proper_premise_basis(context)
    masks = as_masks(basis)
    assert len({p for p, _ in masks}) == len(masks)
    # Wide contexts have tens of thousands of proper premises; check every
    # implication of small ones and an evenly spread sample of the rest.
    for premise, conclusion in masks[:: max(1, len(masks) // 300)]:
        assert conclusion and not premise & conclusion
        assert ref.holds(premise, conclusion)
        # Every conclusion attribute needs the whole premise.
        for a in to_set(premise):
            assert ref.closure(premise & ~(1 << a)) & conclusion == 0
    # Direct: one pass over the implications closes any set.
    for attributes in sample_masks(data, ref.m):
        one_pass = attributes
        for premise, conclusion in masks:
            if premise & attributes == premise:
                one_pass |= conclusion
        assert one_pass == ref.closure(attributes)


@given(incidences(max_attributes=7))
def test_proper_premise_basis_matches_the_definition(incidence):
    ref, context = Reference(incidence), context_of(incidence)
    expected = {}
    for a in range(1 << ref.m):
        below = 0
        for x in to_set(a):
            below |= ref.closure(a & ~(1 << x))
        star = ref.closure(a) & ~a & ~below
        if star:
            expected[to_set(a)] = to_set(star)
    basis = compute_proper_premise_basis(context)
    assert {i.premise: i.conclusion for i in basis} == expected


@given(any_incidences)
def test_minimal_generators_are_minimal_and_close_to_their_intent(incidence):
    ref, context = Reference(incidence), context_of(incidence)
    generators = compute_minimal_generators(context)
    assert {(c.extent, c.intent) for c in generators} == ref.concept_sets()
    for concept, keys in generators.items():
        assert keys and len(set(keys)) == len(keys)
        assert list(keys) == sorted(keys, key=lambda k: (len(k), sorted(k)))
        for key in keys:
            assert key <= concept.intent
            assert to_set(ref.closure(to_mask(key))) == concept.intent
            for x in key:
                assert to_set(ref.closure(to_mask(key - {x}))) != concept.intent
        if ref.m <= 7:
            expected = {
                k for k in all_subsets(concept.intent)
                if to_set(ref.closure(to_mask(k))) == concept.intent
                and all(to_set(ref.closure(to_mask(k - {x}))) != concept.intent for x in k)
            }
            assert set(keys) == expected


@given(any_incidences, st.data())
def test_luxenburger_basis_is_one_rule_per_hasse_edge(incidence, data):
    context = context_of(incidence)
    lattice = ConceptLattice.from_context(context)
    min_support = data.draw(st.integers(0, context.n_objects + 1), label="min_support")
    min_confidence = data.draw(
        st.sampled_from([0.0, 0.25, 0.5, 0.75, 1.0]), label="min_confidence"
    )
    expected = {
        (upper.intent, lower.intent - upper.intent, len(lower.extent), len(upper.extent),
         upper.extent - lower.extent)
        for lower, upper in lattice.edges
        if len(lower.extent) >= min_support
        and len(lower.extent) / len(upper.extent) >= min_confidence
    }
    rules = compute_luxenburger_basis(
        context, min_confidence=min_confidence, min_support=min_support
    )
    got = [(r.premise, r.conclusion, r.support, r.premise_support, r.exceptions) for r in rules]
    assert len(got) == len(set(got))
    assert set(got) == expected
    for rule in rules:
        assert 0 <= rule.confidence < 1
        assert implication_confidence(context, rule.premise, rule.conclusion) == pytest.approx(
            rule.confidence
        )
        assert implication_exceptions(context, rule.premise, rule.conclusion) == rule.exceptions
    # Decreasing confidence, then decreasing support, then by premise.
    assert rules == sorted(rules, key=lambda r: (
        -r.confidence, -r.support, len(r.premise), sorted(r.premise), sorted(r.conclusion)
    ))

    # Given premise concepts: exactly the rules whose premise concept is among them.
    chosen = data.draw(st.lists(st.sampled_from(lattice.concepts), unique=True), label="concepts")
    intents = {c.intent for c in chosen}
    partial = compute_luxenburger_basis(
        context, min_confidence=min_confidence, min_support=min_support, concepts=chosen
    )
    assert set(partial) == {r for r in rules if r.premise in intents}


@given(any_incidences)
def test_exact_implications_have_confidence_one_and_no_exceptions(incidence):
    context = context_of(incidence)
    for implication in compute_canonical_basis(context):
        assert implication_exceptions(context, implication.premise, implication.conclusion) == (
            frozenset()
        )
        support = attribute_set_support(context, implication.premise, normalize=False)
        if support:
            assert implication_confidence(
                context, implication.premise, implication.conclusion
            ) == 1.0


# ---------------------------------------------------------------------------
# Measures
# ---------------------------------------------------------------------------


@given(any_incidences, st.booleans(), st.sampled_from(["nextclosure", "closebyone"]))
def test_stability_lies_within_its_bounds_and_matches_the_definition(incidence, iceberg, method):
    ref, context = Reference(incidence), context_of(incidence)
    threshold = max(1, ref.g // 2) if iceberg else None
    # The concepts' order differs between the algorithms (bottom first or last).
    lattice = ConceptLattice.from_context(context, algorithm=method, min_support=threshold)
    stability = concept_stability(lattice)
    lower, upper = stability_bounds(lattice)
    delta = delta_measure(lattice)
    all_extents = [a for a, _ in ref.concepts()]
    for i, concept in enumerate(lattice.concepts):
        assert 0.0 <= lower[i] <= stability[i] + 1e-12
        assert stability[i] <= upper[i] + 1e-12 and upper[i] <= 1.0
        a, b = to_mask(concept.extent), to_mask(concept.intent)
        sub = [x for x in all_extents if x != a and x & a == x]
        largest = max(
            (x.bit_count() for x in sub if not any(x != y and x & y == x for y in sub)),
            default=0,
        )
        assert delta[i] == a.bit_count() - largest
        if a.bit_count() <= 12:
            kept = sum(1 for c in submasks(a) if ref.intent(c) == b)
            assert stability[i] == pytest.approx(kept / 2 ** a.bit_count(), abs=1e-12)


@given(incidences(max_objects=5, max_attributes=5), st.sampled_from(["nextclosure", "closebyone"]))
def test_order_measures_match_their_definitions(incidence, method):
    lattice = lattice_of(incidence, algorithm=method)
    extents = [c.extent for c in lattice.concepts]
    n = len(extents)
    comparable = sum(
        1 for i in range(n) for j in range(i + 1, n)
        if extents[i] <= extents[j] or extents[j] <= extents[i]
    )
    pairs = n * (n - 1) // 2
    assert incomparable_pairs(lattice) == pairs - comparable
    assert linearity_index(lattice) == (pytest.approx(comparable / pairs) if pairs else 1.0)

    intents = {c.intent for c in lattice.concepts}
    union_closed = sum(
        1 for i in range(n) for j in range(i + 1, n)
        if lattice.concepts[i].intent | lattice.concepts[j].intent in intents
    )
    assert distributivity_index(lattice) == (
        pytest.approx(union_closed / pairs) if pairs else 1.0
    )

    def longest_chain_to(i):
        return 1 + max(
            (longest_chain_to(j) for j in range(n) if extents[j] < extents[i]), default=0
        )

    levels = [longest_chain_to(i) for i in range(n)]
    assert concept_levels(lattice).tolist() == levels
    assert lattice_height(lattice) == max(levels)

    if n <= 16:
        def antichain(members):
            return all(
                not (extents[i] <= extents[j] or extents[j] <= extents[i])
                for k, i in enumerate(members) for j in members[k + 1:]
            )

        width = max(
            mask.bit_count() for mask in range(1, 1 << n)
            if antichain(sorted(to_set(mask)))
        )
        assert lattice_width(lattice) == width


@given(incidences())
def test_concept_support_is_the_extent_size(incidence):
    lattice = lattice_of(incidence)
    context = lattice.context
    for concept in lattice.concepts:
        assert concept_support(concept, context, normalize=False) == len(concept.extent)
        assert attribute_set_support(context, concept.intent, normalize=False) == len(
            concept.extent
        )


@given(incidences(min_objects=1, max_objects=6, min_attributes=1, max_attributes=5),
       st.integers(1, 9))
def test_results_do_not_depend_on_the_chunk_size(incidence, cells):
    # Large lattices and drawings are processed in chunks of _CHUNK_CELLS
    # cells; tiny chunks run those code paths on small examples.
    from unittest import mock

    from conceptflow.algorithms import _membership
    from conceptflow.cluster import ConceptLatticeEstimator
    from conceptflow.metrics import drawing_quality, drawing_quality_scores, measures
    from conceptflow.visualization import lattice_to_graph_data, layered_layout

    lattice = lattice_of(incidence)
    graph_data = layered_layout(lattice_to_graph_data(lattice))

    def results():
        return (
            incomparable_pairs(lattice), lattice_width(lattice), linearity_index(lattice),
            distributivity_index(lattice), concept_levels(lattice).tolist(),
            ConceptLatticeEstimator().fit(incidence).transform(incidence).tolist(),
            drawing_quality_scores(graph_data, worst_case=True),
        )

    expected = results()
    with mock.patch.object(measures, "_CHUNK_CELLS", cells), \
            mock.patch.object(_membership, "_CHUNK_CELLS", cells), \
            mock.patch.object(drawing_quality, "_CHUNK_CELLS", cells):
        chunked = results()
    assert chunked[:6] == expected[:6]
    assert chunked[6] == pytest.approx(expected[6], nan_ok=True)


def test_lattice_is_full_when_min_support_is_the_bottom_size():
    # The bottom concept has one object, so min_support=1 leaves nothing out.
    context = context_of(np.array([[1, 1], [1, 0]], dtype=bool))
    lattice = ConceptLattice.from_context(context, min_support=1)
    assert not lattice.is_iceberg and lattice == ConceptLattice.from_context(context)
    assert ConceptLattice.from_context(context, min_support=0.5) == lattice
    assert ConceptLattice.from_context(context, min_support=2).is_iceberg


@pytest.mark.parametrize("min_support", [True, "2", None, [1]])
def test_min_support_must_be_a_number(min_support):
    context = context_of(np.eye(3, dtype=bool))
    if min_support is None:  # None means the full lattice
        assert len(enumerate_concepts(context, min_support=min_support)) == 5
        return
    with pytest.raises(TypeError):
        compute_luxenburger_basis(context, min_support=min_support)
    with pytest.raises(TypeError):
        enumerate_concepts(context, min_support=min_support)


def test_min_support_accepts_fractions_and_numpy_numbers():
    from fractions import Fraction

    context = context_of(np.eye(4, dtype=bool))
    expected = enumerate_concepts(context, min_support=2)
    for value in (Fraction(1, 2), np.float32(0.5), np.int64(2), 0.5):
        assert enumerate_concepts(context, min_support=value) == expected
    with pytest.raises(ValueError):
        enumerate_concepts(context, min_support=1.5)
    with pytest.raises(ValueError):
        enumerate_concepts(context, min_support=-1)


# ---------------------------------------------------------------------------
# Context operations, reduction and scale-measures
# ---------------------------------------------------------------------------


def renamed(context, prefix):
    return context.rename(
        objects=lambda name: f"{prefix}{name}", attributes=lambda name: f"{prefix}{name}"
    )


@given(incidences(max_objects=4, max_attributes=4), incidences(max_objects=4, max_attributes=4))
def test_direct_sum_lattice_is_the_product_of_the_lattices(first, second):
    left, right = renamed(context_of(first), "L"), renamed(context_of(second), "R")
    total = len(enumerate_concepts(direct_sum(left, right)))
    assert total == len(enumerate_concepts(left)) * len(enumerate_concepts(right))


@given(incidences(max_objects=4, max_attributes=4), incidences(max_objects=4, max_attributes=4))
def test_direct_product_has_the_defined_incidence(first, second):
    left, right = context_of(first), context_of(second)
    product = direct_product(left, right)
    for (g1, g2), row in zip(
        ((a, b) for a in range(left.n_objects) for b in range(right.n_objects)),
        product.incidence,
    ):
        expected = [
            bool(left.incidence[g1, m1] or right.incidence[g2, m2])
            for m1 in range(left.n_attributes) for m2 in range(right.n_attributes)
        ]
        assert row.tolist() == expected


@given(incidences(min_objects=1), st.data())
def test_apposition_extents_are_the_intersections_of_extents(incidence, data):
    g = incidence.shape[0]
    other = data.draw(incidences(min_objects=g, max_objects=g), label="right")
    left = context_of(incidence)
    right = FormalContext.from_array(other, objects=left.objects).rename(
        attributes=lambda name: f"R{name}"
    )
    together = apposition(left, right)
    extents = {c.extent for c in enumerate_concepts(together)}
    left_extents = {c.extent for c in enumerate_concepts(left)}
    right_extents = {c.extent for c in enumerate_concepts(right)}
    assert extents == {a & b for a in left_extents for b in right_extents}

    stacked = subposition(dual(left), dual(right))
    intents = {c.intent for c in enumerate_concepts(stacked)}
    assert intents == extents


@given(any_incidences)
def test_complement_is_an_involution(incidence):
    context = context_of(incidence)
    assert complement(complement(context)) == context
    assert (complement(context).incidence == ~context.incidence).all()


@given(incidences(max_objects=6, max_attributes=6))
def test_reduction_keeps_the_lattice_and_leaves_nothing_reducible(incidence):
    context = context_of(incidence)
    result = reduce_context(context)
    reduced = result.context

    def hasse(ctx):
        lattice = ConceptLattice.from_context(ctx)
        graph = nx.DiGraph()
        graph.add_nodes_from(lattice.concepts)
        graph.add_edges_from(lattice.edges)
        return graph

    assert nx.is_isomorphic(hasse(context), hasse(reduced))

    rows = [to_mask(np.flatnonzero(r)) for r in reduced.incidence]
    columns = [to_mask(np.flatnonzero(c)) for c in reduced.incidence.T]
    for masks, universe in ((rows, (1 << reduced.n_attributes) - 1),
                            (columns, (1 << reduced.n_objects) - 1)):
        assert len(set(masks)) == len(masks)  # clarified
        for mask in masks:
            above = [x for x in masks if x != mask and x & mask == mask]
            meet = universe
            for x in above:
                meet &= x
            assert meet != mask  # irreducible

    # The groups and the removed names account for every original name.
    groups = result.object_groups
    assert sorted(n for names in groups.values() for n in names) == sorted(context.objects)
    assert set(reduced.objects) | set(result.removed_objects) == set(groups)
    assert not set(reduced.objects) & set(result.removed_objects)
    attribute_groups = result.attribute_groups
    assert sorted(n for names in attribute_groups.values() for n in names) == sorted(
        context.attributes
    )
    assert set(reduced.attributes) | set(result.removed_attributes) == set(attribute_groups)

    clarified = clarify_context(context)
    assert nx.is_isomorphic(hasse(context), hasse(clarified.context))


@given(incidences(min_attributes=1), st.data())
def test_scale_measures_of_conjunctions_refine_into_the_context(incidence, data):
    ref, context = Reference(incidence), context_of(incidence)
    m = context.n_attributes
    entries = data.draw(
        st.lists(
            st.lists(st.integers(0, m - 1), min_size=1, max_size=3, unique=True).map(
                lambda parts: tuple(sorted(parts))
            ),
            min_size=1, max_size=4, unique=True,
        ),
        label="conjunctions",
    )
    scale = scale_measure(context, entries)
    assert scale.objects == context.objects
    assert is_scale_measure(context, scale)
    assert refines(scale, context)
    extents = {a for a, _ in ref.concept_sets()}
    assert {c.extent for c in enumerate_concepts(scale)} <= extents

    column = np.array(data.draw(
        st.lists(st.booleans(), min_size=context.n_objects, max_size=context.n_objects),
        label="column",
    ), dtype=bool)
    other = FormalContext(objects=context.objects, attributes=("x",), incidence=column[:, None])
    assert is_scale_measure(context, other) == (
        frozenset(np.flatnonzero(column).tolist()) in extents
    )


# ---------------------------------------------------------------------------
# Building contexts
# ---------------------------------------------------------------------------


@given(any_incidences, st.data())
def test_adding_and_removing_objects_and_attributes_round_trips(incidence, data):
    context = context_of(incidence)
    g, m = incidence.shape
    rows = np.array(data.draw(st.lists(st.booleans(), min_size=2 * m, max_size=2 * m)),
                    dtype=bool).reshape(2, m)
    bigger = context.add_objects(["new 1", "new 2"], rows)
    assert bigger.remove_objects(["new 1", "new 2"]) == context
    assert (bigger.incidence[g:] == rows).all()
    columns = np.array(data.draw(st.lists(st.booleans(), min_size=g, max_size=g)),
                       dtype=bool).reshape(g, 1)
    wider = context.add_attributes(["new"], columns)
    assert wider.remove_attributes(["new"]) == context
    assert wider.subcontext(attributes=list(range(m))) == context
    assert FormalContext.from_dataframe(context.to_dataframe()) == context


@given(any_incidences)
def test_contexts_are_immutable_and_survive_pickling(incidence):
    context = context_of(incidence)
    assert not context.incidence.flags.writeable
    with pytest.raises(ValueError):
        context.incidence[...] = True
    for copied in (pickle.loads(pickle.dumps(context)), copy.deepcopy(context)):
        assert copied == context and hash(copied) == hash(context)
        assert not copied.incidence.flags.writeable
