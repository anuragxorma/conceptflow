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