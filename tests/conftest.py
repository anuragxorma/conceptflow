"""
Hypothesis settings for the property-based tests (``test_properties_*.py``).

``HYPOTHESIS_PROFILE=thorough pytest tests/test_properties_*.py`` tries
twenty times as many examples, e.g. before a release.
"""

import os

try:
    from hypothesis import HealthCheck, settings
except ImportError:  # the property-based tests are then skipped
    pass
else:
    # Time per example varies with the size of the lattice, so no deadline.
    common = {
        "deadline": None,
        "suppress_health_check": [HealthCheck.too_slow, HealthCheck.data_too_large],
        "print_blob": True,
    }
    settings.register_profile("default", max_examples=100, **common)
    settings.register_profile("thorough", max_examples=2000, **common)
    settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "default"))
