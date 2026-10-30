"""
Results checked against independent oracles: each reference below is written
from the mathematical definition (or the paper's pseudocode) in plain Python,
shares no code with ConceptFlow, and is run on small seeded random contexts
where exhaustive search is feasible.

These tests answer "would a wrong result be noticed?" for the mathematical
core. They are deliberately slow-ish but small (a few seconds in total).
"""

import itertools

import numpy as np
import pytest

from conceptflow import ConceptLattice, FormalContext, IncompleteContext
from conceptflow.algorithms import (
    compute_canonical_basis,
    compute_luxenburger_basis,
    compute_minimal_generators,
    compute_proper_premise_basis,
    enumerate_concepts,
)
from conceptflow.core import apposition
from conceptflow.decomposition import ASSO, GreConD, Ord2Factor
from conceptflow.exploration import realized_pairs
from conceptflow.metrics import concept_stability, delta_measure

SEEDS = range(25)


# ---------------------------------------------------------------------------
# Reference implementations
# ---------------------------------------------------------------------------


def random_incidence(seed, max_objects=8, max_attributes=6):
    rng = np.random.default_rng(seed)
    shape = (int(rng.integers(1, max_objects + 1)), int(rng.integers(1, max_attributes + 1)))
    return rng.random(shape) < rng.uniform(0.15, 0.85)


class Ref:
    """Derivation operators and exhaustive constructions on a boolean matrix."""

    def __init__(self, incidence):
        self.I = np.asarray(incidence, dtype=bool)
        self.g, self.m = self.I.shape
        self.rows = [frozenset(np.flatnonzero(row).tolist()) for row in self.I]

    def extent(self, attributes):
        return frozenset(i for i in range(self.g) if frozenset(attributes) <= self.rows[i])

    def intent(self, objects):
        return frozenset(j for j in range(self.m) if all(self.I[i, j] for i in objects))

    def closure(self, attributes):
        return self.intent(self.extent(attributes))

    def subsets(self, items):
        items = sorted(items)
        for size in range(len(items) + 1):
            yield from (frozenset(s) for s in itertools.combinations(items, size))

    def concepts(self):
        intents = {self.closure(b) for b in self.subsets(range(self.m))}
        return {(self.extent(b), b) for b in intents}

    def pseudo_intents(self):
        found = []
        for p in sorted(self.subsets(range(self.m)), key=len):
            if self.closure(p) != p and all(self.closure(q) <= p for q in found if q < p):
                found.append(p)
        return {p: self.closure(p) - p for p in found}


def context_of(incidence):
    return FormalContext.from_array(np.asarray(incidence, dtype=int))


# ---------------------------------------------------------------------------
# Concepts, bases, generators, measures
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["nextclosure", "closebyone", "bruteforce"])
@pytest.mark.parametrize("seed", SEEDS)
def test_concepts_match_the_definition(seed, method):
    ref = Ref(random_incidence(seed))
    found = enumerate_concepts(context_of(ref.I), method=method)
    assert len(found) == len(ref.concepts())
    assert {(c.extent, c.intent) for c in found} == ref.concepts()


@pytest.mark.parametrize("seed", SEEDS)
def test_iceberg_concepts_match_the_definition(seed):
    ref = Ref(random_incidence(seed))
    for threshold in range(1, ref.g + 1):
        found = {c.intent for c in enumerate_concepts(context_of(ref.I), min_support=threshold)}
        assert found == {b for a, b in ref.concepts() if len(a) >= threshold}


@pytest.mark.parametrize("algorithm", ["lincbo", "nextclosure"])
@pytest.mark.parametrize("seed", SEEDS)
def test_canonical_basis_is_the_stem_base(seed, algorithm):
    ref = Ref(random_incidence(seed))
    basis = compute_canonical_basis(context_of(ref.I), algorithm=algorithm)
    assert {i.premise: i.conclusion for i in basis} == ref.pseudo_intents()


@pytest.mark.parametrize("seed", SEEDS)
def test_proper_premise_basis_matches_the_definition(seed):
    ref = Ref(random_incidence(seed))
    expected = {}
    for a in ref.subsets(range(ref.m)):
        below = set().union(*(ref.closure(a - {x}) for x in a)) if a else set()
        star = ref.closure(a) - a - below
        if star:
            expected[a] = star
    basis = compute_proper_premise_basis(context_of(ref.I))
    assert {i.premise: i.conclusion for i in basis} == expected


@pytest.mark.parametrize("seed", SEEDS)
def test_minimal_generators_match_the_definition(seed):
    ref = Ref(random_incidence(seed))
    generators = compute_minimal_generators(context_of(ref.I))
    for concept, keys in generators.items():
        b = concept.intent
        expected = {
            k for k in ref.subsets(b)
            if ref.closure(k) == b and all(ref.closure(k - {x}) != b for x in k)
        }
        assert set(keys) == expected


@pytest.mark.parametrize("seed", SEEDS)
def test_luxenburger_basis_matches_the_definition(seed):
    ref = Ref(random_incidence(seed))
    concepts = ref.concepts()
    expected = set()
    for a1, b1 in concepts:
        for a2, b2 in concepts:
            # min_support=1 (the default): the rule must hold for some object.
            if a2 and a2 < a1 and not any(a2 < x < a1 for x, _ in concepts):
                expected.add((b1, b2 - b1, len(a2), len(a1), a1 - a2))
    rules = compute_luxenburger_basis(context_of(ref.I))
    assert {
        (r.premise, r.conclusion, r.support, r.premise_support, r.exceptions) for r in rules
    } == expected


@pytest.mark.parametrize("iceberg", [False, True])
@pytest.mark.parametrize("seed", SEEDS)
def test_stability_and_delta_match_the_definitions(seed, iceberg):
    ref = Ref(random_incidence(seed))
    lattice = ConceptLattice.from_context(
        context_of(ref.I), min_support=max(1, ref.g // 2) if iceberg else None
    )
    stability, delta = concept_stability(lattice), delta_measure(lattice)
    all_extents = [a for a, _ in ref.concepts()]
    for i, concept in enumerate(lattice.concepts):
        a, b = concept.extent, concept.intent
        kept = sum(1 for c in ref.subsets(a) if ref.intent(c) == b)
        assert stability[i] == pytest.approx(kept / 2 ** len(a), abs=1e-12)
        lower = [len(x) for x in all_extents
                 if x < a and not any(x < y < a for y in all_extents)]
        assert delta[i] == len(a) - max(lower, default=0)


# ---------------------------------------------------------------------------
# Incomplete contexts: certain/possible = in every/some completion
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(40))
def test_incomplete_context_answers_hold_for_all_completions(seed):
    rng = np.random.default_rng(seed)
    g, m = int(rng.integers(1, 5)), int(rng.integers(1, 4))
    values = rng.choice([0, 1, 2], (g, m), p=[0.4, 0.4, 0.2])  # 2: unknown
    unknown = list(zip(*np.nonzero(values == 2)))
    if len(unknown) > 8:
        pytest.skip("too many completions")
    context = IncompleteContext.from_array(
        [[None if v == 2 else int(v) for v in row] for row in values]
    )
    completions = []
    for bits in itertools.product([False, True], repeat=len(unknown)):
        completion = values == 1
        for (i, j), bit in zip(unknown, bits):
            completion[i, j] = bit
        completions.append(Ref(completion))
    attribute_sets = list(Ref(values == 1).subsets(range(m)))
    for b in attribute_sets:
        extents = [c.extent(b) for c in completions]
        assert context.certain_extent(b) == frozenset.intersection(*extents)
        assert context.possible_extent(b) == frozenset.union(*extents)
    for premise, conclusion in itertools.product(attribute_sets, repeat=2):
        broken = [c.extent(premise) - c.extent(premise | conclusion) for c in completions]
        certain, possible = context.implication_counterexamples(premise, conclusion)
        assert certain == frozenset.intersection(*broken)
        assert possible == frozenset.union(*broken)
        holds = [not b for b in broken]
        expected = "valid" if all(holds) else "possible" if any(holds) else "refuted"
        assert context.implication_status(premise, conclusion) == expected


# ---------------------------------------------------------------------------
# Boolean factorization: the papers' pseudocode
# ---------------------------------------------------------------------------


def reference_grecond(incidence):
    """Belohlavek & Vychodil (2010), Algorithm 2, until all 1s are covered."""
    ref = Ref(incidence)
    uncovered = {(i, j) for i in range(ref.g) for j in range(ref.m) if ref.I[i, j]}
    factors = []
    while uncovered:
        d, value = frozenset(), 0
        while True:
            best = None
            for j in range(ref.m):
                if j in d:
                    continue
                a = ref.extent(d | {j})
                b = ref.intent(a)
                covered = sum((i, k) in uncovered for i in a for k in b)
                if covered > value:
                    best, value = b, covered
            if best is None:
                break
            d = best
        a = ref.extent(d)
        factors.append((a, d))
        uncovered -= {(i, k) for i in a for k in d}
    return factors


def reference_asso(incidence, threshold):
    """Miettinen et al. (2008): association matrix, greedy weighted cover."""
    data = np.asarray(incidence, dtype=bool)
    g, m = data.shape
    support = data.sum(axis=0)
    candidates = np.array([
        [support[i] > 0 and (data[:, i] & data[:, j]).sum() / support[i] >= threshold
         for j in range(m)]
        for i in range(m)
    ])
    covered = np.zeros_like(data)
    components = []
    while len(components) < m:
        best, best_total, best_users = None, 0, None
        for c in range(m):
            gains = np.array([
                (data[r] & ~covered[r] & candidates[c]).sum()
                - (~data[r] & ~covered[r] & candidates[c]).sum()
                for r in range(g)
            ])
            total = gains[gains > 0].sum()
            if total > best_total:
                best, best_total, best_users = c, total, gains > 0
        if best is None:
            break
        components.append(tuple(candidates[best]))
        covered |= best_users[:, None] & candidates[best][None, :]
    return components


@pytest.mark.parametrize("seed", SEEDS)
def test_grecond_follows_the_paper(seed):
    incidence = random_incidence(seed, max_objects=10, max_attributes=8)
    model = GreConD().fit(incidence.astype(int))
    factors = [(f.extent, f.intent) for f in model.factors_]
    assert factors == reference_grecond(incidence)
    assert model.reconstruction_error_ == 0  # factors are concepts: exact


@pytest.mark.parametrize("threshold", [0.5, 0.7, 1.0])
@pytest.mark.parametrize("seed", SEEDS)
def test_asso_follows_the_paper(seed, threshold):
    incidence = random_incidence(seed, max_objects=10, max_attributes=8)
    model = ASSO(threshold=threshold).fit(incidence.astype(int))
    assert [tuple(c) for c in model.components_] == reference_asso(incidence, threshold)


# ---------------------------------------------------------------------------
# Ordinal two-factorization: the largest two-factorizable sub-relation
# ---------------------------------------------------------------------------


def is_ferrers(relation):
    return all((g, n) in relation or (h, m) in relation for g, m in relation for h, n in relation)


def two_factorizable(relation):
    """Bipartite incompatibility graph (Dürrschnabel & Stumme, 2023)."""
    cells = sorted(relation)
    neighbours = {c: set() for c in cells}
    for (g, m), (h, n) in itertools.combinations(cells, 2):
        if (g, n) not in relation and (h, m) not in relation:
            neighbours[(g, m)].add((h, n))
            neighbours[(h, n)].add((g, m))
    colour = {}
    for start in cells:
        if start in colour:
            continue
        colour[start], stack = 0, [start]
        while stack:
            v = stack.pop()
            for w in neighbours[v]:
                if w not in colour:
                    colour[w] = 1 - colour[v]
                    stack.append(w)
                elif colour[w] == colour[v]:
                    return False
    return True


@pytest.mark.parametrize("seed", range(30))
def test_exact_ordinal_factorization_is_optimal(seed):
    rng = np.random.default_rng(seed)
    incidence = rng.random((int(rng.integers(2, 6)), int(rng.integers(2, 6)))) < 0.5
    cells = {tuple(map(int, c)) for c in np.argwhere(incidence)}
    if len(cells) > 13:
        pytest.skip("too many sub-relations")
    best = next(
        len(cells) - k
        for k in range(len(cells) + 1)
        if any(two_factorizable(cells - set(d)) for d in itertools.combinations(sorted(cells), k))
    )
    for method in ("exact", "annealing", "greedy"):
        model = Ord2Factor(method=method, random_state=0).fit(incidence.astype(int))
        first, second = (set(f.relation) for f in model.factors_)
        assert is_ferrers(first) and is_ferrers(second)
        assert first | second == set(model.covered_incidence_) <= cells
        if method == "exact":
            assert len(model.covered_incidence_) == best


# ---------------------------------------------------------------------------
# Nested line diagrams: filled pairs are the concepts of the apposition
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", SEEDS)
def test_realized_pairs_are_the_apposition_concepts(seed):
    rng = np.random.default_rng(seed)
    g = int(rng.integers(1, 7))
    outer = FormalContext.from_array((rng.random((g, int(rng.integers(1, 4)))) < 0.5).astype(int))
    inner = FormalContext.from_array(
        (rng.random((g, int(rng.integers(1, 4)))) < 0.5).astype(int),
        objects=outer.objects,
    ).rename(attributes=lambda name: f"inner:{name}")
    whole = Ref(np.asarray(apposition(outer, inner).incidence))
    split = outer.n_attributes
    expected = {
        (frozenset(j for j in b if j < split), frozenset(j - split for j in b if j >= split))
        for _, b in whole.concepts()
    }
    outer_lattice = ConceptLattice.from_context(outer)
    inner_lattice = ConceptLattice.from_context(inner)
    got = {
        (c.intent, d.intent)
        for c, d in realized_pairs(outer_lattice, inner_lattice)
    }
    assert got == expected


# ---------------------------------------------------------------------------
# Ordinal scales: the relation each mode stands for
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(10))
def test_ordinal_and_interordinal_scales_match_their_relations(seed):
    import pandas as pd

    from conceptflow.preprocessing import ConceptualScaler, InterordinalScale, OrdinalScale

    levels = [1, 2, 3, 4, 5]
    values = np.random.default_rng(seed).choice(levels, 12).tolist()
    data = pd.DataFrame({"v": values}, index=[f"o{i}" for i in range(12)])
    relations = {"ge": (">=", lambda v, k: v >= k), "le": ("<=", lambda v, k: v <= k),
                 "exact": ("=", lambda v, k: v == k)}
    for mode, (symbol, holds) in relations.items():
        scaled = ConceptualScaler(
            scales=[OrdinalScale("v", levels, mode=mode)], output="dataframe"
        ).fit_transform(data)
        for k in levels:
            assert scaled[f"v{symbol}{k}"].tolist() == [holds(v, k) for v in values]
    scaled = ConceptualScaler(
        scales=[InterordinalScale("v", levels)], output="dataframe"
    ).fit_transform(data)
    for k in levels:
        assert scaled[f"v<={k}"].tolist() == [v <= k for v in values]
        assert scaled[f"v>={k}"].tolist() == [v >= k for v in values]


# ---------------------------------------------------------------------------
# Order indices and scale-measures, from their definitions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", SEEDS)
def test_order_indices_match_their_definitions(seed):
    from conceptflow.metrics import (
        concept_levels,
        incomparable_pairs,
        lattice_height,
        lattice_width,
    )

    ref = Ref(random_incidence(seed, max_objects=6, max_attributes=5))
    lattice = ConceptLattice.from_context(context_of(ref.I))
    extents = [c.extent for c in lattice.concepts]
    comparable = {(i, j) for i, a in enumerate(extents) for j, b in enumerate(extents)
                  if i != j and (a <= b or b <= a)}
    pairs = len(extents) * (len(extents) - 1) // 2
    assert incomparable_pairs(lattice) == pairs - len(comparable) // 2

    # Width: the largest set of pairwise incomparable concepts, by trying sets
    # from the largest size down.
    def antichain(indices):
        return all((i, j) not in comparable for i, j in itertools.combinations(indices, 2))

    width = next(k for k in range(len(extents), 0, -1)
                 if any(antichain(s) for s in itertools.combinations(range(len(extents)), k)))
    assert lattice_width(lattice) == width

    # Levels: the most concepts on a chain ending in each concept.
    def longest_chain_to(i):
        return 1 + max((longest_chain_to(j) for j in range(len(extents))
                        if extents[j] < extents[i]), default=0)

    levels = [longest_chain_to(i) for i in range(len(extents))]
    assert concept_levels(lattice).tolist() == levels
    assert lattice_height(lattice) == max(levels)


@pytest.mark.parametrize("seed", SEEDS)
def test_scale_measures_match_the_definition(seed):
    from conceptflow.core import is_scale_measure, refines, scale_measure

    ref = Ref(random_incidence(seed))
    context = context_of(ref.I)
    extents = {a for a, _ in ref.concepts()}
    rng = np.random.default_rng(seed)

    # Projections and conjunctions: every extent of the scale is one of the
    # context's, so the scale refines into the context.
    entries = [tuple(rng.choice(ref.m, size=int(rng.integers(1, min(3, ref.m) + 1)),
                                replace=False).tolist()) for _ in range(3)]
    entries = list(dict.fromkeys(tuple(sorted(e)) for e in entries))
    scale = scale_measure(context, entries)
    scale_extents = {a for a, _ in Ref(np.asarray(scale.incidence)).concepts()}
    assert scale_extents <= extents
    assert is_scale_measure(context, scale) and refines(scale, context)

    # An arbitrary column is a scale-measure iff its extent is an extent.
    column = rng.random(ref.g) < 0.5
    other = FormalContext(objects=context.objects, attributes=("x",), incidence=column[:, None])
    assert is_scale_measure(context, other) == (
        frozenset(np.flatnonzero(column).tolist()) in extents
    )
