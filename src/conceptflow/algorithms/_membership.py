"""
Vectorised "does this object have every attribute of this set?" tests.

Used by the transformers that turn objects into one boolean feature per
concept intent or implication premise.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy import sparse

# Upper bound on cells per chunk of the (objects x sets) result, which keeps
# each chunk's intermediates around 16 MB however many objects there are.
_CHUNK_CELLS = 1 << 22


def contains_sets(
    incidence: np.ndarray,
    attribute_sets: Sequence[frozenset[int]],
) -> np.ndarray:
    """
    Return ``result[g, k] = attribute_sets[k] ⊆ row g of incidence``.

    An object contains a set iff none of the set's attributes is missing
    from its row, so one product of "missing attributes" (objects x
    attributes) and the sets (attributes x sets) counts the misses for every
    pair at once. The sets are stored sparsely, so memory grows with their
    total size rather than with sets x attributes, and objects are processed
    in chunks.
    """
    incidence = np.asarray(incidence, dtype=bool)
    n_objects, n_attributes = incidence.shape
    n_sets = len(attribute_sets)

    lengths = [len(attributes) for attributes in attribute_sets]
    indptr = np.concatenate([[0], np.cumsum(lengths, dtype=np.int64)])
    indices = np.fromiter(
        (m for attributes in attribute_sets for m in attributes),
        dtype=np.int64,
        count=int(indptr[-1]),
    )
    sets = sparse.csr_matrix(
        (np.ones(len(indices), dtype=np.float32), indices, indptr),
        shape=(n_sets, n_attributes),
    )

    result = np.empty((n_objects, n_sets), dtype=bool)
    chunk = max(1, _CHUNK_CELLS // max(1, n_sets, n_attributes))

    for start in range(0, n_objects, chunk):
        stop = start + chunk
        missing = (~incidence[start:stop]).astype(np.float32)
        # (sets x attributes) @ (attributes x chunk) -> misses per (set, object).
        result[start:stop] = (sets @ missing.T).T == 0

    return result
