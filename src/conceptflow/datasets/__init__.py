"""
Datasets: classic formal contexts, the UCI Mushroom data and random contexts.

- ``load_context(name)``: the 20 contexts of the FCA repository
  (fcarepository.org, CC0), shipped with ConceptFlow so they work offline;
  ``list_contexts()`` describes them.
- ``fetch_fca_repository(name)``: download the current version of any
  repository context (cached).
- ``load_mushroom()``: the UCI Mushroom data set (downloaded once, cached),
  as the standard 8124 x 119 formal context or as raw many-valued data.
- ``make_random_context(...)``: random formal contexts.
"""

from conceptflow.datasets._loaders import (
    fetch_fca_repository,
    get_data_home,
    list_contexts,
    load_context,
    load_mushroom,
    make_random_context,
)

__all__ = [
    "fetch_fca_repository",
    "get_data_home",
    "list_contexts",
    "load_context",
    "load_mushroom",
    "make_random_context",
]
