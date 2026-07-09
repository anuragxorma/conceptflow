"""
Algorithm registry utilities.

This module centralizes algorithm name handling so that public APIs can
support clean canonical names and optional aliases without duplicating
string logic across the codebase.
"""

from __future__ import annotations


ENUMERATION_ALGORITHM_ALIASES = {
    "bruteforce": "bruteforce",
    "brute_force": "bruteforce",
    "brute-force": "bruteforce",
    "nextclosure": "nextclosure",
    "next_closure": "nextclosure",
    "next-closure": "nextclosure",
    "closebyone": "closebyone",
    "close_by_one": "closebyone",
    "close-by-one": "closebyone",
    "cbo": "closebyone",
}

SUPPORTED_ENUMERATION_ALGORITHMS = frozenset(
    ENUMERATION_ALGORITHM_ALIASES.values()
)


def normalize_enumeration_algorithm(name: str) -> str:
    """
    Normalize a concept enumeration algorithm name.

    Parameters
    ----------
    name:
        User-provided algorithm name.

    Returns
    -------
    str
        Canonical algorithm name.

    Raises
    ------
    ValueError
        If the algorithm is unknown.
    """
    normalized = name.lower().strip().replace(" ", "_")

    if normalized not in ENUMERATION_ALGORITHM_ALIASES:
        supported = ", ".join(sorted(SUPPORTED_ENUMERATION_ALGORITHMS))
        raise ValueError(
            f'Unknown concept enumeration algorithm "{name}". '
            f"Supported algorithms are: {supported}."
        )

    return ENUMERATION_ALGORITHM_ALIASES[normalized]