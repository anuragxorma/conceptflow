# Nested Line Diagram: Mathematical Construction

This document describes both the abstract theory from Ganter & Wille and the
actual implementation.  The two do not always coincide at the code level — the
implementation reaches the same result via a different but equivalent path.
Each section is marked:

- **Theory** — what Ganter & Wille §4.1 says about the mathematical object.
- **Implementation** — what the code actually computes, and how it relates to
  the theory.

---

## 1. The Two Factor Contexts

### Theory

Let G be the set of objects (Eurovision winners 1975–2025, excluding 2020).
Suppose the attribute set M is partitioned into two disjoint parts:

```
M_outer = { regional_support, cultural_support, historical_support, political_support }
M_inner = { bpm ≥ 100, bpm ≥ 150, key_mode_minor }
```

The **outer formal context** K_outer = (G, M_outer, I_outer) and the
**inner formal context** K_inner = (G, M_inner, I_inner) are the two
factors.  Their **apposition** is

```
K = K_outer | K_inner  =  (G, M_outer ∪ M_inner, I)
```

where (g, m) ∈ I iff (g, m) ∈ I_outer for m ∈ M_outer, or (g, m) ∈ I_inner
for m ∈ M_inner.  K is the combined context whose concept lattice B(K) the
nested diagram encodes.

### Implementation

`builder.root(scales=outer_scales)` builds K_outer by applying a
`ConceptualScaler` to the full `ManyValuedContext`, producing a `FormalContext`
with objects in DataFrame row order and attributes M_outer.

`builder.expand_all(parent=root, scales=inner_scales)` builds K_inner (for the
top outer concept, which has all objects) by restricting the MVC to the top
concept's extent and applying a second `ConceptualScaler` with the inner scales.

**K itself is never constructed.**  The implementation reaches image(φ) — the
set of filled coordinate pairs — without ever building B(K).

**Enforced invariant:** `exploration_view_to_nested_data` checks that
M_outer ∩ M_inner = ∅ and raises `ValueError` if not.  This is required for K
to be the apposition; disjointness is what makes image(φ) equal to the
join-closure of the atomic pairs.

---

## 2. The Subdirect Embedding (Theory)

**Theory only.** This section describes the abstract object.  The
implementation never computes φ directly.

Given the apposition K = K_outer | K_inner, there is an embedding

```
φ : B(K) ↪ B(K_outer) × B(K_inner)
φ(C) = (π_outer(C), π_inner(C))
```

where π_outer(C) is the concept of K_outer whose extent is the closure of
ext(C) under K_outer:

```
ext(π_outer(C)) = { g ∈ G : for all m ∈ int(C) ∩ M_outer, (g,m) ∈ I_outer }
```

and analogously for π_inner(C).

The image of φ is the **subdirect product**: a sublattice of
B(K_outer) × B(K_inner) onto which both projections are surjective.  Ganter &
Wille (Theorem 4.3) prove that φ is an isomorphism onto this image.

---

## 3. What a Filled Inner Node Means

### Theory

An inner concept c_inner is drawn **filled** at outer concept c_outer iff

```
(c_outer, c_inner) ∈ image(φ)
```

This means there exists a concept C ∈ B(K) such that π_outer(C) = c_outer and
π_inner(C) = c_inner.

### Implementation

The implementation computes the same set without building B(K) or evaluating φ.
Instead it uses the join-closure of object-level coordinate pairs, described in
Sections 4–5.  Correctness of this shortcut depends on the bridging lemma in
Section 4.

---

## 4. Atomic Coordinate Pairs

### Theory

For each g ∈ G, the object concept μ(g) ∈ B(K) is the smallest concept
containing g.  Its image under φ is the **atomic coordinate pair**:

```
φ(μ(g)) = (π_outer(μ(g)), π_inner(μ(g)))
```

**Bridging lemma.** In the apposition K = K_outer | K_inner:

```
π_outer(μ_K(g)) = μ_{K_outer}(g)
π_inner(μ_K(g)) = μ_{K_inner}(g)
```

*Proof.* In K, ext(μ_K(g)) = {g}''_K = objects sharing all of g's outer *and*
inner attributes.  This set is a subset of {g}''_{K_outer} (which only requires
sharing outer attributes).  Now:

- {g}''_{K_outer} is already K_outer-closed.
- g ∈ ext(μ_K(g)) ⊆ {g}''_{K_outer}, so ext(μ_K(g)) lies inside the closed
  set {g}''_{K_outer}.
- The closure of any set A with g ∈ A ⊆ {g}''_{K_outer} in K_outer equals
  {g}''_{K_outer} (it is sandwiched: {g}''_{K_outer} ⊆ A''_{K_outer} ⊆
  ({g}''_{K_outer})''_{K_outer} = {g}''_{K_outer}).

Therefore ext(π_outer(μ_K(g))) = {g}''_{K_outer} = ext(μ_{K_outer}(g)).  The
argument for the inner projection is symmetric. ∎

### Implementation

Because of the bridging lemma, the implementation can compute atomic pairs
directly from the factor lattices without touching B(K):

```
μ_outer(g)  =  smallest concept in B(K_outer) whose extent contains g
μ_inner(g)  =  smallest concept in B(K_inner) whose extent contains g
```

Both are computed by `_mu_local` applied to the outer and template (full-object
inner) lattices respectively.  The `obj_idx` argument is an integer index into
the shared object list; the code asserts that this list is identical in both
contexts before pairing.

The seed set is

```
P_0 = { (μ_outer(g), μ_inner(g)) : g ∈ G }  ∪  { (⊥_outer, ⊥_inner) }
```

The bottom pair is added explicitly because ⊥_K has an empty extent; no object
maps to it, yet it is always a concept of B(K) and must be represented.

---

## 5. Join-Closure

### Theory

φ is **join-preserving**: for any C_1, C_2 ∈ B(K),

```
φ(C_1 ∨ C_2) = (π_outer(C_1) ∨ π_outer(C_2),  π_inner(C_1) ∨ π_inner(C_2))
             = φ(C_1) ∨ φ(C_2)
```

This holds because in the apposition, the outer-scale closure of a union of
extents equals the join of the individual outer closures, and analogously for
the inner scale.

Since every concept C ∈ B(K) satisfies

```
C = ∨{ μ(g) : g ∈ ext(C) }
```

(provable because for a closed set A, A = ∪{ext(μ(g)) : g ∈ A}), the image
of φ equals the join-closure of {φ(μ(g)) : g ∈ G} ∪ {φ(⊥_K)}.

### Implementation

The code applies the join-closure directly in the factor lattices using the
atomic pairs from Section 4 in place of {φ(μ(g))}.  This is valid by the
bridging lemma:

```
if (c_o, c_i), (c_o′, c_i′) ∈ P  then add  (c_o ∨ c_o′,  c_i ∨ c_i′)  to P
```

The join c ∨ c′ in each factor lattice is the smallest concept whose extent
contains ext(c) ∪ ext(c′) — computed by scanning all concepts for the minimum
extent size containing the union.  This is unique in any closure system.
Iteration continues until fixpoint:

```
P* = ⋃_{n ≥ 0} P_n  =  image(φ)
```

---

## 6. Correctness: P* = image(φ)

Two containments establish the equality.

**P* ⊆ image(φ).** Every element of P* is a finite join of pairs from P_0.
Each pair in P_0 is φ(μ(g)) or φ(⊥_K) by the bridging lemma.  Because φ
preserves joins, any join of such pairs is the image of the corresponding join
in B(K), hence lies in image(φ).

**image(φ) ⊆ P*.** Every C ∈ B(K) satisfies C = ∨{μ(g) : g ∈ ext(C)}.
Applying φ and join-preservation:

```
φ(C) = ∨{ φ(μ(g)) : g ∈ ext(C) }
     = ∨{ (μ_outer(g), μ_inner(g)) : g ∈ ext(C) }   [by the bridging lemma]
     ∈ P*
```

---

## 7. Relation to Ganter & Wille §4.1

Theorem 4.3 proves that B(K) is isomorphic to a subdirect product of B(K_outer)
and B(K_inner) when K is the apposition K_outer | K_inner.  The nested line
diagram (Figure 4.3) makes this visible: at each outer node c_outer, the inner
lattice is rendered with

```
filled nodes = { c_inner : (c_outer, c_inner) ∈ P* }
```

This set is the **fiber** of the subdirect product over c_outer: all concepts
of B(K) whose outer projection equals c_outer.  Because φ is injective, the
number of filled inner nodes at c_outer equals the number of concepts in B(K)
projecting to that outer coordinate.

**What the implementation does not do.** It never enumerates B(K), never
evaluates φ on a concept, and never forms the combined context K.  It computes
P* = image(φ) purely from the two factor lattices via the join-closure algorithm,
relying on the bridging lemma (Section 4) to connect factor-level quantities to
the abstract embedding.  The result is mathematically identical to the G&W
diagram; the path to reach it is different.

---

*Ganter, B. & Wille, R. (1999). Formal Concept Analysis: Mathematical
Foundations. Springer-Verlag. §4.1, Theorem 4.3, Figure 4.3.*
