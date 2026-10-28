from conceptflow import Concept


def test_concept_normalizes_extent_and_intent_to_frozensets():
    concept = Concept(extent={2, 1}, intent=[3, 4])

    assert concept.extent == frozenset({1, 2})
    assert concept.intent == frozenset({3, 4})


def test_concept_signature_is_deterministic():
    c1 = Concept(extent={2, 1}, intent={4, 3})
    c2 = Concept(extent={1, 2}, intent={3, 4})

    assert c1.signature() == c2.signature()


def test_concept_signature_distinguishes_concepts():
    c1 = Concept(extent={1, 2}, intent={3})
    c2 = Concept(extent={1}, intent={2, 3})

    assert c1.signature() != c2.signature()


def test_concept_stable_id_is_deterministic():
    c1 = Concept(extent={2, 1}, intent={4, 3})
    c2 = Concept(extent={1, 2}, intent={3, 4})

    assert c1.stable_id() == c2.stable_id()


def test_concept_stable_id_uses_prefix():
    concept = Concept(extent={1}, intent={2})

    assert concept.stable_id(prefix="concept").startswith("concept_")


def test_stable_id_depends_only_on_the_intent():
    first = Concept(extent=frozenset({0, 1}), intent=frozenset({2}))
    same_intent = Concept(extent=frozenset({5, 7, 9}), intent=frozenset({2}))
    other = Concept(extent=frozenset({0, 1}), intent=frozenset({1, 2}))

    assert first.stable_id() == same_intent.stable_id()
    assert first.stable_id() != other.stable_id()


def test_concept_feature_names_do_not_depend_on_row_order():
    import numpy as np

    from conceptflow import FormalContext
    from conceptflow.cluster import ConceptLatticeEstimator

    data = np.random.default_rng(0).random((30, 6)) < 0.4
    names = ConceptLatticeEstimator().fit(data).get_feature_names_out()
    reordered = ConceptLatticeEstimator().fit(data[::-1]).get_feature_names_out()
    assert sorted(names) == sorted(reordered)
    assert len(set(names)) == len(names)
    # A concept learned on a subsample keeps its name.
    subsample = ConceptLatticeEstimator().fit(FormalContext.from_array(data[:20]))
    assert set(subsample.get_feature_names_out()) & set(names)
