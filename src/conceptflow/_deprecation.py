"""
Deprecation warnings, following the policy in docs/api-stability.md: a
deprecated name keeps working for at least one minor release and says what
replaces it and in which version it will be removed.
"""

from __future__ import annotations

import warnings


def warn_deprecated(old: str, new: str, removal: str, stacklevel: int = 3) -> None:
    """Warn that ``old`` is deprecated in favour of ``new`` until ``removal``."""
    warnings.warn(
        f"{old} is deprecated; use {new} instead. It will be removed in "
        f"ConceptFlow {removal}.",
        DeprecationWarning,
        stacklevel=stacklevel,
    )
