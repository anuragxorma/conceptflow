"""
Interestingness measures for formal concepts and indices for whole lattices.

Concept measures return one value per concept, as a NumPy array in the order
of ``lattice.concepts``:

- ``concept_stability``: exact (intensional) stability,
- ``stability_bounds``: cheap lower and upper bounds for stability,
- ``delta_measure``: the Δ-measure (difference to the largest subconcept),
- ``concept_entropy`` / ``concept_information_gain``: label-aware measures,
- ``concept_levels``: the length of the longest chain below each concept.

Lattice indices return one number:

- ``linearity_index``: share of comparable pairs of concepts,
- ``distributivity_index``: share of pairs of intents whose union is an
  intent,
- ``lattice_width``, ``lattice_height``, ``incomparable_pairs``: the largest
  antichain, the longest chain and the number of incomparable pairs.
"""

from __future__ import annotations

import warnings

import numpy as np

from conceptflow.algorithms._bitset import BitsetContext, from_indices, iter_bits
from conceptflow.core.lattice import ConceptLattice

# Cells per chunk of the pairwise (concepts x concepts) computations.
_CHUNK_CELLS = 1 << 22

# Above this many concepts the quadratic measures warn about their cost.
MEASURE_WARNING_SIZE = 20_000


def _warn_if_large(lattice: ConceptLattice, what: str, alternative: str) -> None:
    if lattice.n_concepts > MEASURE_WARNING_SIZE:
        from conceptflow.algorithms.enumeration import _INTERNAL_PREFIXES

        warnings.warn(
            f"{what} grows quadratically with the number of concepts and may take "
            f"minutes for {lattice.n_concepts:,} concepts. {alternative}",
            UserWarning,
            skip_file_prefixes=_INTERNAL_PREFIXES,
        )


def _maximal_lower_extents(lattice: ConceptLattice) -> list[list[int]]:
    """
    Extents (as bitsets) of the lower neighbours of every concept.

    They are the maximal sets among ``A ∩ m'`` for ``m`` not in the intent,
    read off the context, so this also works for iceberg lattices, whose
    lower neighbours may lie below the iceberg.
    """
    bits = BitsetContext(lattice.context)
    result = []

    for concept in lattice.concepts:
        extent = from_indices(concept.extent)
        candidates = {
            extent & bits.columns[m]
            for m in range(bits.n_attributes)
            if m not in concept.intent
        }
        candidates.discard(extent)

        maximal: list[int] = []
        for candidate in sorted(candidates, key=int.bit_count, reverse=True):
            if not any(candidate & accepted == candidate for accepted in maximal):
                maximal.append(candidate)
        result.append(maximal)

    return result


def concept_stability(lattice: ConceptLattice) -> np.ndarray:
    """
    Exact (intensional) stability of every concept.

    The stability of a concept ``(A, B)`` is the share of subsets ``C`` of
    ``A`` with ``C' = B``: how much the intent survives when objects are
    removed. A subset keeps the intent iff it does not fit inside one of the
    maximal proper sub-extents ``A_1, ..., A_r`` of ``A`` (the extents of
    the lower neighbours), so by inclusion-exclusion

        |{C ⊆ A : C' = B}| = Σ_{S ⊆ {1..r}} (-1)^|S| 2^|A ∩ ⋂_{i∈S} A_i|.

    The terms are collected by intersection (each intersection is an extent
    below ``A``), so the cost depends on the concept's neighbourhood, not on
    the size of the lattice; exact integers are used throughout. Only the
    context is needed, so this also works on iceberg lattices.
    """
    bits = BitsetContext(lattice.context)
    columns = bits.columns
    stability = np.empty(lattice.n_concepts, dtype=float)

    for i, concept in enumerate(lattice.concepts):
        extent = from_indices(concept.extent)
        missing = bits.all_attributes & ~from_indices(concept.intent)
        candidates = {extent & columns[m] for m in iter_bits(missing)}
        candidates.discard(extent)

        maximal: list[int] = []
        for candidate in sorted(candidates, key=int.bit_count, reverse=True):
            if not any(candidate & kept == candidate for kept in maximal):
                maximal.append(candidate)

        # terms[I] = sum of (-1)^|S| over the subsets S with intersection I.
        terms = {extent: 1}
        for sub_extent in maximal:
            updated = dict(terms)
            for intersection, coefficient in terms.items():
                key = intersection & sub_extent
                updated[key] = updated.get(key, 0) - coefficient
            terms = {key: c for key, c in updated.items() if c}

        generators = sum(c << key.bit_count() for key, c in terms.items())
        stability[i] = generators / (1 << extent.bit_count())

    return stability


def delta_measure(lattice: ConceptLattice) -> np.ndarray:
    """
    Δ-measure: the number of objects lost to the largest lower neighbour.

    ``Δ(A, B) = min over lower neighbours (A_i, B_i) of |A| - |A_i|``. Large
    values mean the concept does not shrink easily into a more specific one.
    Concepts without lower neighbours get ``|A|``. Works on iceberg lattices.
    """
    values = np.empty(lattice.n_concepts, dtype=int)

    for i, (concept, lowers) in enumerate(
        zip(lattice.concepts, _maximal_lower_extents(lattice))
    ):
        largest = max((lower.bit_count() for lower in lowers), default=0)
        values[i] = len(concept.extent) - largest

    return values


def stability_bounds(lattice: ConceptLattice) -> tuple[np.ndarray, np.ndarray]:
    """
    Lower and upper bounds for stability, from the lower neighbours only.

    With ``Δ_i = |A| - |A_i|`` for the lower neighbours ``(A_i, B_i)``:

        1 - Σ_i 2^(-Δ_i)  <=  stability  <=  1 - 2^(-min_i Δ_i)

    (Buzmakov, Kuznetsov & Napoli, 2014). Much cheaper than the exact value
    and also available on iceberg lattices.

    Returns
    -------
    (lower, upper): two arrays in the order of ``lattice.concepts``.
    """
    lower = np.empty(lattice.n_concepts, dtype=float)
    upper = np.empty(lattice.n_concepts, dtype=float)

    for i, (concept, lowers) in enumerate(
        zip(lattice.concepts, _maximal_lower_extents(lattice))
    ):
        size = len(concept.extent)
        deltas = [size - extent.bit_count() for extent in lowers]

        if not deltas:
            # No subconcept: this is the bottom concept (A, M), and every
            # subset of A -- even the empty one -- derives to M.
            lower[i] = upper[i] = 1.0
            continue

        lower[i] = max(0.0, 1.0 - sum(2.0 ** -d for d in deltas))
        upper[i] = 1.0 - 2.0 ** -min(deltas)

    return lower, upper


def _label_entropy(labels: np.ndarray) -> float:
    if len(labels) == 0:
        return 0.0
    _, counts = np.unique(labels, return_counts=True)
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum()) + 0.0  # + 0.0 turns -0.0 into 0.0


def _check_labels(lattice: ConceptLattice, y) -> np.ndarray:
    y = np.asarray(y)
    if y.ndim != 1 or len(y) != lattice.context.n_objects:
        raise ValueError(
            f"y must be a 1D array with one label per object "
            f"({lattice.context.n_objects}), got shape {y.shape}."
        )
    return y


def concept_entropy(lattice: ConceptLattice, y) -> np.ndarray:
    """
    Shannon entropy (bits) of the labels ``y`` within each concept's extent.

    0 means all objects of the concept share one label. ``y`` has one label
    per object of ``lattice.context``.
    """
    y = _check_labels(lattice, y)
    return np.array(
        [_label_entropy(y[sorted(concept.extent)]) for concept in lattice.concepts]
    )


def concept_information_gain(lattice: ConceptLattice, y) -> np.ndarray:
    """
    Information gain of splitting the objects into "in the extent" and "not".

    ``IG = H(y) - (|A|/n · H(y_A) + (1 - |A|/n) · H(y_rest))``: how much
    knowing that an object belongs to the concept tells about its label.
    """
    y = _check_labels(lattice, y)
    n = len(y)
    total = _label_entropy(y)
    gains = np.empty(lattice.n_concepts, dtype=float)

    for i, concept in enumerate(lattice.concepts):
        inside = np.zeros(n, dtype=bool)
        inside[sorted(concept.extent)] = True
        share = inside.sum() / n if n else 0.0
        gains[i] = total - (
            share * _label_entropy(y[inside]) + (1 - share) * _label_entropy(y[~inside])
        )

    return gains


def _pairwise_counts(left: np.ndarray, right: np.ndarray):
    """Yield (row_slice, left_chunk @ right.T) over chunks of ``left``."""
    chunk = max(1, _CHUNK_CELLS // max(1, right.shape[0]))
    for start in range(0, left.shape[0], chunk):
        yield slice(start, start + chunk), left[start : start + chunk] @ right.T


def _membership_matrix(sets, size: int) -> np.ndarray:
    matrix = np.zeros((len(sets), size), dtype=np.float32)
    for i, members in enumerate(sets):
        matrix[i, list(members)] = 1.0
    return matrix


def _comparable_pairs(lattice: ConceptLattice) -> int:
    """
    Number of unordered pairs of distinct concepts that are comparable.

    Uses intents instead of extents when there are fewer attributes than
    objects (A_i ⊆ A_j exactly when B_j ⊆ B_i), keeping the matrices small.
    """
    n = lattice.n_concepts
    ctx = lattice.context
    if ctx.n_attributes < ctx.n_objects:
        sets, width = [c.intent for c in lattice.concepts], ctx.n_attributes
    else:
        sets, width = [c.extent for c in lattice.concepts], ctx.n_objects
    extents = _membership_matrix(sets, width)
    sizes = extents.sum(axis=1)
    comparable = 0

    for rows, common in _pairwise_counts(extents, extents):
        # A_i ⊆ A_j  <=>  |A_i ∩ A_j| == |A_i|, and A_j ⊆ A_i likewise.
        below = common == sizes[rows][:, None]
        above = common == sizes[None, :]
        comparable += int((below | above).sum())

    # Ordered pairs were counted, including the n pairs (i, i).
    return (comparable - n) // 2


def linearity_index(lattice: ConceptLattice) -> float:
    """
    Share of pairs of distinct concepts that are comparable.

    1 for a chain, small for "wide" lattices. Returns 1.0 for lattices with
    fewer than two concepts.
    """
    n = lattice.n_concepts
    if n < 2:
        return 1.0
    return _comparable_pairs(lattice) / (n * (n - 1) / 2)


def distributivity_index(lattice: ConceptLattice) -> float:
    """
    Share of pairs of distinct intents whose union is again an intent.

    The index depends on the context, not only on the lattice's shape: it is
    1 for the reduced context of a distributive lattice, but attributes such
    as one that no object has can lower it. Needs the full lattice.

    Comparable pairs always qualify (their union is the larger intent). An
    incomparable pair qualifies iff its union is the intent ``B_c`` of the
    concept ``c`` that is their meet; such pairs are found from every ``c``
    by starting with pairs of its upper neighbours whose union is ``B_c``
    and moving upwards while the union stays ``B_c``. This visits only the
    qualifying pairs, which is fast for typical lattices but quadratic for
    (near-)distributive ones, where almost every pair qualifies.
    """
    if lattice.is_iceberg:
        raise ValueError("distributivity_index needs the full lattice.")
    _warn_if_large(lattice, "distributivity_index", "linearity_index() is much cheaper.")

    n = lattice.n_concepts
    if n < 2:
        return 1.0

    position = {concept: i for i, concept in enumerate(lattice.concepts)}
    intents = [from_indices(c.intent) for c in lattice.concepts]
    parents = [[position[p] for p in lattice.parents(c)] for c in lattice.concepts]
    incomparable_distributive = 0

    for c, target in enumerate(intents):
        ups = parents[c]
        # Pairs are stored as (smaller, larger) position and queued only once.
        seen: set[tuple[int, int]] = {
            (min(a, b), max(a, b))
            for k, a in enumerate(ups)
            for b in ups[k + 1 :]
            if intents[a] | intents[b] == target
        }
        queue = list(seen)

        while queue:
            a, b = queue.pop()
            candidates = [(a, up) for up in parents[b]] + [(up, b) for up in parents[a]]
            for x, y in candidates:
                pair = (x, y) if x < y else (y, x)
                if pair not in seen and intents[x] | intents[y] == target:
                    seen.add(pair)
                    queue.append(pair)

        incomparable_distributive += len(seen)

    qualifying = _comparable_pairs(lattice) + incomparable_distributive
    return qualifying / (n * (n - 1) / 2)


def concept_levels(lattice: ConceptLattice) -> np.ndarray:
    """
    The level of every concept: the number of concepts in a longest chain
    from a minimal concept up to it (minimal concepts have level 1).

    In a full lattice the bottom concept is the only one at level 1, and the
    top concept's level is ``lattice_height``. Concepts on one level are
    pairwise incomparable, so a drawing may arrange each level freely from
    left to right. On an iceberg lattice the levels are those of the drawn
    concepts among themselves.
    """
    position = {concept: i for i, concept in enumerate(lattice.concepts)}
    levels = np.zeros(lattice.n_concepts, dtype=int)
    # Lower neighbours have smaller extents, so they get their level first.
    for i in sorted(range(lattice.n_concepts), key=lambda k: len(lattice.concepts[k].extent)):
        below = [levels[position[child]] for child in lattice.children(lattice.concepts[i])]
        levels[i] = max(below, default=0) + 1
    return levels


def lattice_height(lattice: ConceptLattice) -> int:
    """
    The number of concepts in a longest chain: the largest ``concept_levels``.

    A chain of k concepts has height k; a lattice with one concept has
    height 1.
    """
    return int(concept_levels(lattice).max())


def incomparable_pairs(lattice: ConceptLattice) -> int:
    """
    The number of unordered pairs of concepts neither of which lies below
    the other.

    It is what a two-dimensional drawing has to decide (which of the two
    goes left), and it grows much faster than the number of concepts in wide
    lattices; ``linearity_index`` is the share of the other pairs.
    """
    n = lattice.n_concepts
    return n * (n - 1) // 2 - _comparable_pairs(lattice)


def lattice_width(lattice: ConceptLattice) -> int:
    """
    The size of a largest antichain: the most concepts that are pairwise
    incomparable.

    By Dilworth's theorem this is the number of concepts minus a maximum
    matching between "is strictly below" pairs (all comparable pairs, not
    only covers), found with SciPy's Hopcroft-Karp implementation. Quadratic
    in the number of concepts in time and memory.
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import maximum_bipartite_matching

    _warn_if_large(lattice, "lattice_width", "incomparable_pairs() is cheaper.")
    n = lattice.n_concepts
    if n < 2:
        return n
    extents = _membership_matrix([c.extent for c in lattice.concepts], lattice.context.n_objects)
    sizes = extents.sum(axis=1)
    rows, columns = [], []
    for chunk, common in _pairwise_counts(extents, extents):
        # i strictly below j: A_i is contained in A_j and smaller (extents differ).
        below = (common == sizes[chunk][:, None]) & (sizes[chunk][:, None] < sizes[None, :])
        r, c = np.nonzero(below)
        rows.append(r + chunk.start)
        columns.append(c)
    rows, columns = np.concatenate(rows), np.concatenate(columns)
    graph = csr_matrix((np.ones(len(rows), dtype=np.int8), (rows, columns)), shape=(n, n))
    matching = maximum_bipartite_matching(graph, perm_type="column")
    return n - int((matching >= 0).sum())
