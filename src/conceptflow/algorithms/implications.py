"""
Duquenne-Guigues (stem) basis computation for ConceptFlow.

The canonical basis is the smallest set of implications (attribute-set
"premise forces conclusion" rules) that is logically equivalent to every
implication that holds in a formal context. It is built from the context's
*pseudo-intents*.

Definition (Ganter & Wille, "Formal Concept Analysis", Prop. 2.15)
-------------------------------------------------------------------
For a formal context K = (G, M, I) with derivation/closure operator
``''`` (``attribute_closure``), a set P subseteq M is a **pseudo-intent** if:

1. P'' != P              (P is not itself a concept intent), and
2. for every pseudo-intent Q that is a *proper subset* of P: Q'' subseteq P.

Condition 2 is recursive, but well-founded: since Q must be strictly
smaller than P, processing candidate sets in increasing size order (ties
broken by any fixed rule, here lectic/lexicographic) guarantees every
pseudo-intent smaller than the current candidate has already been decided
before it is needed.

The canonical basis is then exactly:

    { P -> (P'' \\ P) : P is a pseudo-intent }

Implementations:

- ``compute_canonical_basis`` uses LinCbO (Janoštík, Konečný and Krajča,
  "LinCbO: fast algorithm for computation of the Duquenne-Guigues basis",
  Information Sciences 572, 2021) by default, or Ganter's NextClosure with
  ``algorithm="nextclosure"``. Both work on integer bitsets and visit only
  the closed sets of the closure operator ``A |-> A^{L*}`` (concept intents
  plus pseudo-intents), so their cost grows with the size of the lattice and
  the basis, not with ``2^|M|``.
- ``compute_canonical_basis_bruteforce`` tries every subset of attributes and
  checks the definition above directly. It is exponential in ``|M|`` and is
  kept as a readable correctness baseline, in the same spirit as
  ``enumerate_concepts_bruteforce``.

The module also provides ``compute_luxenburger_basis`` for *near*
(partial) implications: rules that hold with confidence below 1, each with
its list of exception objects.
"""

from __future__ import annotations

import math
import numbers
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import chain, combinations

import numpy as np

from conceptflow.algorithms._bitset import BitsetContext, iter_bits, to_indices
from conceptflow.algorithms.derivation import attribute_closure
from conceptflow.core.concept import Concept
from conceptflow.core.context import FormalContext


@dataclass(frozen=True)
class Implication:
    """
    A single exact implication premise -> conclusion.

    Parameters
    ----------
    premise:
        Attribute indices forming the "if" side.

    conclusion:
        Attribute indices forced by the premise, *beyond* the premise
        itself (i.e. ``conclusion`` and ``premise`` are disjoint).
    """

    premise: frozenset[int]
    conclusion: frozenset[int]

    def __post_init__(self) -> None:
        object.__setattr__(self, "premise", frozenset(self.premise))
        object.__setattr__(self, "conclusion", frozenset(self.conclusion))

    def premise_names(self, context: FormalContext) -> tuple[str, ...]:
        """Return the premise as attribute names, in a fixed order."""
        return tuple(context.attributes[i] for i in sorted(self.premise))

    def conclusion_names(self, context: FormalContext) -> tuple[str, ...]:
        """Return the conclusion as attribute names, in a fixed order."""
        return tuple(context.attributes[i] for i in sorted(self.conclusion))

    def __repr__(self) -> str:
        return (
            f"Implication(premise={sorted(self.premise)}, "
            f"conclusion={sorted(self.conclusion)})"
        )


def _powerset(n_attributes: int):
    """Yield every subset of range(n_attributes), smallest first."""
    attributes = range(n_attributes)
    return chain.from_iterable(
        combinations(attributes, size) for size in range(n_attributes + 1)
    )


def compute_canonical_basis_bruteforce(context: FormalContext) -> list[Implication]:
    """
    Compute the Duquenne-Guigues (stem) basis by checking every subset.

    This is exponential in the number of attributes and only practical for
    small contexts (roughly ``|M| <= 15``). Use ``compute_canonical_basis``
    for real data; this function is a reference implementation for tests.

    Step by step
    ------------
    1. Generate every candidate attribute subset, sorted by
       ``(size, sorted(subset))`` -- smallest first. This order is what
       makes step 3 well-defined: any pseudo-intent that could matter for
       a minimality check on the current candidate is strictly smaller,
       so it was already visited earlier in this same loop.
    2. Close the candidate with the context's own derivation operator
       (``attribute_closure``, i.e. B''). If the candidate is already
       equal to its own closure, it is a genuine concept intent, not a
       pseudo-intent -- skip it.
    3. Otherwise, check the minimality condition against every
       already-found pseudo-intent Q that is a proper subset of this
       candidate: all of their closures must already fit inside the
       candidate. If so, the candidate is itself a new pseudo-intent;
       record it together with its closure.
    4. Once every candidate has been classified, turn each pseudo-intent
       P (with closure P'') into one implication P -> (P'' \\ P) -- the
       attributes P forces *beyond itself*.

    Parameters
    ----------
    context:
        The formal context to compute the basis for.

    Returns
    -------
    list[Implication]
        The canonical basis, ordered by ``(premise size, premise)``.
    """
    pseudo_intents: dict[frozenset[int], frozenset[int]] = {}

    for subset in _powerset(context.n_attributes):
        candidate = frozenset(subset)
        closure = attribute_closure(context, candidate)

        if closure == candidate:
            # Already a concept intent -- nothing new to force.
            continue

        is_minimal = all(
            known_closure <= candidate
            for known_premise, known_closure in pseudo_intents.items()
            if known_premise < candidate
        )

        if is_minimal:
            pseudo_intents[candidate] = closure

    ordered_premises = sorted(
        pseudo_intents,
        key=lambda premise: (len(premise), sorted(premise)),
    )

    return [
        Implication(premise=premise, conclusion=pseudo_intents[premise] - premise)
        for premise in ordered_premises
    ]


class _ImplicationStore:
    """
    The basis found so far, stored for fast ``A^{L*}`` closures.

    ``A^{L*}`` closes ``A`` under every implication whose premise is a
    *proper* subset of the (growing) set. Each closure pass tests all
    implications at once with NumPy instead of looping over them in Python:

    - up to 64 attributes, premises and closures are packed into one
      ``uint64`` each and tested with bitwise operations;
    - beyond that, premises are a 0/1 matrix and one matrix-vector product
      counts, per premise, the attributes still missing from ``A``.

    ``close`` also stops as soon as the closure picks up a *forbidden*
    attribute: NextClosure rejects such candidates anyway, and most
    candidates are rejected. (Pure-Python LinClosure was measured too; its
    per-implication bookkeeping made it slower than both layouts here.)
    """

    _WORD = 64

    def __init__(self, n_attributes: int) -> None:
        self.n = n_attributes
        self.packed = n_attributes <= self._WORD
        self.k = 0
        capacity = 64

        if self.packed:
            self.premises = np.zeros(capacity, dtype=np.uint64)
            self.closures = np.zeros(capacity, dtype=np.uint64)
        else:
            self.nbytes = (n_attributes + 7) // 8
            self.premises = np.zeros((capacity, n_attributes), dtype=np.float32)
            self.closures = np.zeros((capacity, n_attributes), dtype=bool)
            self.sizes = np.zeros(capacity, dtype=np.float32)

    def _to_bools(self, mask: int) -> np.ndarray:
        raw = np.frombuffer(mask.to_bytes(self.nbytes, "little"), dtype=np.uint8)
        return np.unpackbits(raw, bitorder="little")[: self.n].astype(bool)

    def _from_bools(self, bools: np.ndarray) -> int:
        return int.from_bytes(np.packbits(bools, bitorder="little").tobytes(), "little")

    def add(self, premise: int, closure: int) -> None:
        if self.k == len(self.premises):
            self.premises = np.concatenate([self.premises, np.zeros_like(self.premises)])
            self.closures = np.concatenate([self.closures, np.zeros_like(self.closures)])
            if not self.packed:
                self.sizes = np.concatenate([self.sizes, np.zeros_like(self.sizes)])

        if self.packed:
            self.premises[self.k] = premise
            self.closures[self.k] = closure
        else:
            self.premises[self.k] = self._to_bools(premise)
            self.closures[self.k] = self._to_bools(closure)
            self.sizes[self.k] = premise.bit_count()

        self.k += 1

    def close(self, attributes: int, forbidden: int) -> int | None:
        """
        Return ``attributes^{L*}``, or None if it meets ``forbidden``.
        """
        if self.k == 0:
            return None if attributes & forbidden else attributes

        if self.packed:
            return self._close_packed(attributes, forbidden)

        return self._close_matrix(attributes, forbidden)

    def _close_packed(self, attributes: int, forbidden: int) -> int | None:
        premises = self.premises[: self.k]
        closures = self.closures[: self.k]
        current = np.uint64(attributes)
        blocked = np.uint64(forbidden)

        while True:
            fire = ((premises & ~current) == 0) & (premises != current)
            if not fire.any():
                break
            grown = current | np.bitwise_or.reduce(closures[fire])
            if grown == current:
                break
            current = grown
            if current & blocked:
                return None

        return None if current & blocked else int(current)

    def _close_matrix(self, attributes: int, forbidden: int) -> int | None:
        premises = self.premises[: self.k]
        closures = self.closures[: self.k]
        sizes = self.sizes[: self.k]
        current = self._to_bools(attributes)
        blocked = self._to_bools(forbidden)

        while True:
            missing = premises @ (~current).astype(np.float32)
            # P ⊊ A  <=>  no premise attribute is missing and |P| < |A|.
            fire = (missing == 0) & (sizes < current.sum())
            if not fire.any():
                break
            grown = current | closures[fire].any(axis=0)
            if (grown == current).all():
                break
            current = grown
            if (current & blocked).any():
                return None

        return None if (current & blocked).any() else self._from_bools(current)


CANONICAL_BASIS_ALGORITHMS = ("lincbo", "nextclosure")


def compute_canonical_basis(
    context: FormalContext, algorithm: str = "lincbo"
) -> list[Implication]:
    """
    Compute the Duquenne-Guigues (stem) basis of a formal context.

    Both algorithms enumerate the closed sets of ``A^{L*}``, where ``L`` is
    the part of the basis found so far, in lectic order. These closed sets
    are exactly the concept intents and the pseudo-intents; every one that
    is not a concept intent is a pseudo-intent and contributes the
    implication ``P -> P'' \\ P``.

    Parameters
    ----------
    context:
        The formal context to compute the basis for.

    algorithm:
        ``"lincbo"`` (default): LinCbO, a Close-by-One search in which each
        closure continues from its parent's (LinClosure counters), so it
        only processes the newly added attribute. Much faster on contexts
        with many attributes. ``"nextclosure"``: Ganter's NextClosure, which
        closes every candidate from scratch. Both give the same basis.

    Returns
    -------
    list[Implication]
        The canonical basis, ordered by ``(premise size, premise)``.
    """
    if algorithm == "lincbo":
        basis = _lincbo(context)
    elif algorithm == "nextclosure":
        basis = _nextclosure_basis(context)
    else:
        raise ValueError(
            f"Unknown canonical basis algorithm {algorithm!r}. "
            f"Use one of {CANONICAL_BASIS_ALGORITHMS}."
        )

    implications = [
        Implication(premise=to_indices(premise), conclusion=to_indices(closure & ~premise))
        for premise, closure in basis
    ]

    return sorted(
        implications,
        key=lambda imp: (len(imp.premise), sorted(imp.premise)),
    )


def _lincbo(context: FormalContext) -> list[tuple[int, int]]:
    """
    LinCbO: (pseudo-intent, closure) bitset pairs in lectic order.

    A depth-first Close-by-One search that tries children with the largest
    attribute first, which visits the ``L*``-closed sets in lectic order (so
    every pseudo-intent below a set is known before the set is closed).

    - A child ``(B ∪ {i})^{L*}`` is computed by LinClosure *continuing from
      B's counters*: ``counts[k]`` is the number of premise attributes of
      implication ``k`` still missing from B, so only implications whose
      premise contains a newly added attribute are touched.
    - The child is canonical iff its closure adds no attribute smaller than
      ``i`` (checked as attributes are added, to stop early).
    - A pseudo-intent ``P`` adds ``P -> P''``; the search then continues
      from the intent ``P''`` (if canonical) instead of from ``P``.
    """
    bits = BitsetContext(context)
    n = bits.n_attributes
    columns = bits.columns
    rows = bits.rows
    all_attributes = bits.all_attributes

    basis: list[tuple[int, int]] = []
    by_attribute: list[list[int]] = [[] for _ in range(n)]
    by_attribute_array: list[np.ndarray | None] = [None] * n

    def implications_with(a: int) -> np.ndarray:
        array = by_attribute_array[a]
        if array is None or len(array) != len(by_attribute[a]):
            array = np.array(by_attribute[a], dtype=np.intp)
            by_attribute_array[a] = array
        return array

    def intent_of(extent: int, known: int) -> int:
        # Intersect the rows of the extent's objects when there are few of
        # them; otherwise test each remaining attribute's column.
        rest = all_attributes & ~known
        if extent.bit_count() < 2 * rest.bit_count():
            result = all_attributes
            for g in iter_bits(extent):
                result &= rows[g]
            return result | known
        result = known
        for m in iter_bits(rest):
            if not extent & ~columns[m]:
                result |= 1 << m
        return result

    def close(base, base_extent, base_counts, new, below):
        """
        ``(base ∪ new)^{L*}`` with its extent and counters, continuing from
        base's counters; None as soon as an attribute in ``below`` appears.
        """
        counts = base_counts.copy()
        current = base | new
        extent = base_extent
        todo = new & ~base
        while todo:
            fired: list[int] = []
            for a in iter_bits(todo):
                extent &= columns[a]
                indices = implications_with(a)
                if len(indices):
                    counts[indices] -= 1
                    hit = indices[counts[indices] == 0]
                    if len(hit):
                        fired.extend(hit.tolist())
            added = 0
            for k in fired:
                added |= basis[k][1]
            added &= ~current
            if added & below:
                return None
            current |= added
            todo = added
        return current, extent, counts

    def with_new_implications(counts: np.ndarray, current: int) -> np.ndarray:
        """Counters extended by the implications found since they were made."""
        k = len(counts)
        if k == len(basis):
            return counts
        extra = np.fromiter(
            ((basis[j][0] & ~current).bit_count() for j in range(k, len(basis))),
            dtype=np.int32,
            count=len(basis) - k,
        )
        return np.concatenate([counts, extra])

    def add_implication(premise: int, closure: int) -> None:
        for a in iter_bits(premise):
            by_attribute[a].append(len(basis))
        basis.append((premise, closure))

    root = intent_of(bits.all_objects, 0)
    start = (0, bits.all_objects, np.zeros(0, dtype=np.int32))
    if root:
        add_implication(0, root)
        start = close(0, bits.all_objects, with_new_implications(start[2], 0), root, 0)

    # Frames: [set, extent, counters, generating attribute, next attribute to try].
    stack = [[start[0], start[1], start[2], -1, n - 1]]
    while stack:
        frame = stack[-1]
        current, extent, counts, y, i = frame
        while i > y and current >> i & 1:
            i -= 1
        if i <= y:
            stack.pop()
            continue
        frame[4] = i - 1
        counts = frame[2] = with_new_implications(counts, current)

        bit = 1 << i
        below = (bit - 1) & ~current
        found = close(current, extent, counts, bit, below)
        if found is None:
            continue
        candidate, candidate_extent, candidate_counts = found
        intent = intent_of(candidate_extent, candidate)
        if intent == candidate:
            stack.append([candidate, candidate_extent, candidate_counts, i, n - 1])
            continue

        add_implication(candidate, intent)
        if intent & below:
            continue  # the intent is reached from another branch
        candidate_counts = with_new_implications(candidate_counts, candidate)
        jumped = close(candidate, candidate_extent, candidate_counts, intent & ~candidate, below)
        stack.append([jumped[0], jumped[1], jumped[2], i, n - 1])

    return basis


def _nextclosure_basis(context: FormalContext) -> list[tuple[int, int]]:
    """
    NextClosure: (pseudo-intent, closure) bitset pairs in lectic order.

    1. Start with ``A = {}`` (closed under the empty basis).
    2. If ``A'' != A``, ``A`` is a pseudo-intent: record ``A -> A''``.
    3. Find the lectically next ``L*``-closed set: for attributes ``i`` from
       the last to the first, try ``B = ((A ∩ {0..i-1}) ∪ {i})^{L*}`` and
       accept the first ``B`` that adds no attribute smaller than ``i``.
    4. Stop once ``A`` is the full attribute set.
    """
    bits = BitsetContext(context)
    basis: list[tuple[int, int]] = []
    store = _ImplicationStore(bits.n_attributes)

    current: int | None = 0

    while current is not None:
        closure = bits.closure(current)

        if closure != current:
            basis.append((current, closure))
            store.add(current, closure)

        following = None
        prefix = current

        for i in range(bits.n_attributes - 1, -1, -1):
            bit = 1 << i

            if prefix & bit:
                prefix &= ~bit
                continue

            # Canonicity: the closure may not add an attribute below i.
            candidate = store.close(prefix | bit, forbidden=(bit - 1) & ~prefix)

            if candidate is not None:
                following = candidate
                break

        current = following

    return basis


@dataclass(frozen=True)
class AssociationRule:
    """
    A near (partial) implication premise -> conclusion with confidence < 1.

    Parameters
    ----------
    premise:
        Attribute indices forming the "if" side.

    conclusion:
        Attribute indices that hold for most, but not all, objects having
        the premise. Disjoint from ``premise``.

    support:
        Number of objects having ``premise | conclusion``.

    premise_support:
        Number of objects having ``premise``.

    exceptions:
        Object indices that have the premise but lack part of the
        conclusion -- the counterexamples that keep the rule from being an
        exact implication.
    """

    premise: frozenset[int]
    conclusion: frozenset[int]
    support: int
    premise_support: int
    exceptions: frozenset[int]

    def __post_init__(self) -> None:
        object.__setattr__(self, "premise", frozenset(self.premise))
        object.__setattr__(self, "conclusion", frozenset(self.conclusion))
        object.__setattr__(self, "exceptions", frozenset(self.exceptions))

    @property
    def confidence(self) -> float:
        """``support / premise_support``."""
        return self.support / self.premise_support

    def premise_names(self, context: FormalContext) -> tuple[str, ...]:
        """Return the premise as attribute names, in a fixed order."""
        return tuple(context.attributes[i] for i in sorted(self.premise))

    def conclusion_names(self, context: FormalContext) -> tuple[str, ...]:
        """Return the conclusion as attribute names, in a fixed order."""
        return tuple(context.attributes[i] for i in sorted(self.conclusion))

    def exception_names(self, context: FormalContext) -> tuple[str, ...]:
        """Return the exception objects as object names, in a fixed order."""
        return tuple(context.objects[i] for i in sorted(self.exceptions))

    def __repr__(self) -> str:
        return (
            f"AssociationRule(premise={sorted(self.premise)}, "
            f"conclusion={sorted(self.conclusion)}, "
            f"confidence={self.confidence:.3f}, "
            f"support={self.support}/{self.premise_support}, "
            f"n_exceptions={len(self.exceptions)})"
        )


def _resolve_min_support(min_support: int | float, n_objects: int) -> int:
    """Turn an absolute (int) or relative (float) support threshold into a count."""
    if isinstance(min_support, bool) or not isinstance(
        min_support, (numbers.Integral, numbers.Real)
    ):
        raise TypeError("min_support must be an int (count) or a float (fraction).")

    if not isinstance(min_support, numbers.Integral):
        # str() gives the shortest decimal for NumPy floats too, so
        # np.float32(0.1) counts as 0.1 rather than 0.10000000149.
        try:
            min_support = float(str(min_support))
        except ValueError:  # e.g. Fraction(1, 2), whose str() is "1/2"
            min_support = float(min_support)
        if not 0.0 <= min_support <= 1.0:
            raise ValueError("A fractional min_support must lie in [0, 1].")
        # Rounding keeps e.g. 0.1 * 30 == 3.0000000000000004 from rounding up to 4.
        return math.ceil(round(min_support * n_objects, 6))

    if min_support < 0:
        raise ValueError("min_support must be non-negative.")

    return int(min_support)


def compute_luxenburger_basis(
    context: FormalContext,
    min_confidence: float = 0.0,
    min_support: int | float = 1,
    concepts: Sequence[Concept] | None = None,
) -> list[AssociationRule]:
    """
    Compute the Luxenburger basis of near (partial) implications.

    For every pair of neighbouring concepts ``(A1, B1)`` above ``(A2, B2)``
    in the concept lattice, the basis contains the rule

        B1 -> B2 \\ B1    with confidence |A2| / |A1| < 1.

    Together with the canonical basis (exact implications), these rules
    determine the confidence of every association rule of the context
    (Luxenburger, 1991). Each rule carries its exception objects
    ``A1 \\ A2``.

    Parameters
    ----------
    context:
        The formal context to mine.

    min_confidence:
        Keep only rules with ``confidence >= min_confidence``.

    min_support:
        Keep only rules whose support ``|A2|`` reaches this threshold. An
        ``int`` is an object count; a ``float`` is a fraction of all objects.

    concepts:
        Concepts of ``context`` to use as rule premises, e.g. the already
        computed lattice. Their lower neighbours always come from the full
        context, so passing a subset yields exactly the rules whose premise
        concept is in it. Enumerated with Close-by-One when omitted.

    Returns
    -------
    list[AssociationRule]
        Rules sorted by decreasing confidence, then decreasing support.
    """
    from conceptflow.algorithms._bitset import from_indices
    from conceptflow.algorithms.enumeration import _closebyone_bitsets, _intent_of

    if (
        isinstance(min_confidence, bool)
        or not isinstance(min_confidence, numbers.Real)
        or not 0.0 <= min_confidence <= 1.0
    ):
        raise ValueError("min_confidence must be a number in [0, 1].")

    support_threshold = _resolve_min_support(min_support, context.n_objects)
    bits = BitsetContext(context)
    columns = bits.columns

    # A rule's premise concept has at least as many objects as its lower
    # neighbour, so only concepts reaching the support threshold matter.
    if concepts is None:
        uppers = _closebyone_bitsets(bits, min_extent_size=max(support_threshold, 1))
    else:
        uppers_by_extent: dict[int, tuple[int, int]] = {}
        for concept in concepts:
            if (
                any(g >= context.n_objects for g in concept.extent)
                or any(m >= context.n_attributes for m in concept.intent)
            ):
                raise ValueError(
                    f"Concept {concept!r} has indices outside the context "
                    f"({context.n_objects} objects, {context.n_attributes} attributes)."
                )
            if len(concept.extent) >= max(support_threshold, 1):
                extent = from_indices(concept.extent)
                uppers_by_extent[extent] = (extent, from_indices(concept.intent))
        uppers = list(uppers_by_extent.values())

    rules: list[AssociationRule] = []

    for extent, intent in uppers:
        premise_support = extent.bit_count()

        # Lower neighbours: the maximal extents among A ∩ m' for m not in B
        # (see hasse._lower_cover_indices_from_context).
        candidates = {
            extent & columns[m]
            for m in range(bits.n_attributes)
            if not intent >> m & 1
        }
        candidates.discard(extent)

        maximal: list[int] = []
        for candidate in sorted(candidates, key=int.bit_count, reverse=True):
            if not any(candidate & accepted == candidate for accepted in maximal):
                maximal.append(candidate)

        for lower_extent in maximal:
            support = lower_extent.bit_count()

            if support < support_threshold or support / premise_support < min_confidence:
                continue

            rules.append(
                AssociationRule(
                    premise=to_indices(intent),
                    conclusion=to_indices(_intent_of(lower_extent, columns) & ~intent),
                    support=support,
                    premise_support=premise_support,
                    exceptions=to_indices(extent & ~lower_extent),
                )
            )

    return sorted(
        rules,
        key=lambda rule: (
            -rule.confidence,
            -rule.support,
            len(rule.premise),
            sorted(rule.premise),
            sorted(rule.conclusion),
        ),
    )


def compute_proper_premise_basis(context: FormalContext) -> list[Implication]:
    """
    Compute the proper-premise basis (canonical direct basis).

    A *proper premise* for an attribute ``m`` is a minimal set ``A`` (not
    containing ``m``) with ``m`` in ``A''``. The basis consists of the
    implications ``A -> A•`` with

        A• = A'' \\ (A ∪ ⋃_{a in A} (A \\ {a})''),

    the attributes for which ``A`` is a proper premise (Ganter). Unlike the
    canonical basis it is *direct*: applying each implication once to any
    attribute set already gives its closure.

    Every proper premise is a minimal generator (key) -- a proper subset with
    the same extent would imply ``m`` as well -- so the premises are found
    among the keys, whose subsets' extents are already known.

    Returns
    -------
    list[Implication]
        Ordered by ``(premise size, premise)``.
    """
    from conceptflow.algorithms.enumeration import _intent_of
    from conceptflow.algorithms.generators import _keys_with_extents

    bits = BitsetContext(context)
    key_extent = _keys_with_extents(bits)
    intent_of: dict[int, int] = {}

    def closure_of_key(key: int) -> int:
        extent = key_extent[key]
        if extent not in intent_of:
            intent_of[extent] = _intent_of(extent, bits.columns)
        return intent_of[extent]

    implications = []
    for key in key_extent:
        conclusion = closure_of_key(key) & ~key
        rest = key
        while rest and conclusion:
            low = rest & -rest
            conclusion &= ~closure_of_key(key ^ low)
            rest ^= low

        if conclusion:
            implications.append(
                Implication(premise=to_indices(key), conclusion=to_indices(conclusion))
            )

    return sorted(implications, key=lambda imp: (len(imp.premise), sorted(imp.premise)))
