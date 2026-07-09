import pytest

from conceptflow.algorithms import normalize_enumeration_algorithm


def test_normalize_enumeration_algorithm_canonical_names():
    assert normalize_enumeration_algorithm("nextclosure") == "nextclosure"
    assert normalize_enumeration_algorithm("bruteforce") == "bruteforce"


def test_normalize_enumeration_algorithm_aliases():
    assert normalize_enumeration_algorithm("next_closure") == "nextclosure"
    assert normalize_enumeration_algorithm("next closure") == "nextclosure"
    assert normalize_enumeration_algorithm("brute_force") == "bruteforce"
    assert normalize_enumeration_algorithm("brute-force") == "bruteforce"


def test_normalize_enumeration_algorithm_closebyone_aliases():
    assert normalize_enumeration_algorithm("closebyone") == "closebyone"
    assert normalize_enumeration_algorithm("close_by_one") == "closebyone"
    assert normalize_enumeration_algorithm("close-by-one") == "closebyone"
    assert normalize_enumeration_algorithm("cbo") == "closebyone"


def test_unknown_algorithm_raises_value_error():
    with pytest.raises(ValueError):
        normalize_enumeration_algorithm("unknown")