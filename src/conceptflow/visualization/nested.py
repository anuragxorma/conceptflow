"""
nested.py

Nested diagram visualization support.

This module connects the exploration layer to a browser-rendered nested
diagram implementing Toscana-style subdirect decomposition:

- Reduced labels: attributes only at γ(m) (concept with largest extent
  introducing m), objects only at μ(g) (concept with smallest extent
  containing g).
- Attribute labels ABOVE node, object labels BELOW node.
- Subdirect decomposition preview: every outer node shows the same inner
  template (from the ⊤ outer concept). Inner nodes are filled (dark) if
  their extent intersects the new objects introduced at this outer node,
  hollow otherwise.

Big picture
-----------
The full pipeline is:

    ManyValuedContext
        -> ExplorationBuilder
        -> ExplorationView tree
        -> exploration_view_to_nested_data
        -> render_nested_html
        -> HTMLFigure
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from conceptflow.exploration.view import ExplorationView
from conceptflow.visualization.dimflux_layout import lattice_to_graph_data_dimflux
from conceptflow.visualization.html_figure import HTMLFigure


def _gamma_local(lattice) -> dict[int, str]:
    """γ(m): stable_id of the concept with the LARGEST extent having m in its intent."""
    result = {}
    for attr_idx in range(lattice.context.n_attributes):
        candidates = [c for c in lattice.concepts if attr_idx in c.intent]
        if candidates:
            result[attr_idx] = max(candidates, key=lambda c: len(c.extent)).stable_id()
    return result


def _mu_local(lattice) -> dict[int, str]:
    """μ(g): stable_id of the concept with the SMALLEST extent still containing g."""
    result = {}
    for obj_idx in range(lattice.context.n_objects):
        candidates = [c for c in lattice.concepts if obj_idx in c.extent]
        if candidates:
            result[obj_idx] = min(candidates, key=lambda c: len(c.extent)).stable_id()
    return result


def _inner_template_data(template_view: ExplorationView) -> dict[str, Any]:
    """
    Build the shared inner template from the ⊤ outer concept's child view.

    Returns a dict with 'nodes', 'edges', and 'mu_by_name'.
    'mu_by_name' is a helper popped by the caller before embedding in JSON.

    new_objects are NOT included in the template nodes: they depend on which
    outer concept we are currently at (μ_outer) and are injected per outer
    concept when building the synthetic click-through child view.
    """
    inner_gd = lattice_to_graph_data_dimflux(template_view.lattice, stable_ids=True)

    gamma_inner = _gamma_local(template_view.lattice)
    mu_inner_idx = _mu_local(template_view.lattice)  # obj_idx → stable_id

    inner_by_attrs: dict[str, list[str]] = {
        c.stable_id(): [] for c in template_view.lattice.concepts
    }

    for attr_idx, cid in gamma_inner.items():
        inner_by_attrs[cid].append(template_view.context.attributes[attr_idx])

    # μ_inner by object name: used to place object labels in the click-through.
    mu_by_name: dict[str, str] = {
        template_view.context.objects[obj_idx]: cid
        for obj_idx, cid in mu_inner_idx.items()
    }

    return {
        "nodes": [
            {
                "id": node.node_id,
                "x": node.x,
                "y": node.y,
                "extent": sorted(node.metadata["extent"]),
                "extent_size": node.metadata["extent_size"],
                "intent": sorted(node.metadata["intent"]),
                "intent_size": node.metadata["intent_size"],
                "new_attrs": sorted(inner_by_attrs[node.node_id]),
            }
            for node in inner_gd.nodes
        ],
        "edges": [
            {"source": e.source, "target": e.target}
            for e in inner_gd.edges
        ],
        "mu_by_name": mu_by_name,
    }


def _compute_filled_pairs(
    outer_lattice,
    inner_lattice,
    mu_outer: dict[int, str],
    mu_inner: dict[int, str],
) -> dict[str, set[str]]:
    """
    Compute filled (outer_id, inner_id) coordinate pairs of the subdirect product.

    A pair is filled iff it is the projection of some concept in B(K) — the full
    combined lattice — onto the two factor lattices.  By the join-generation
    theorem this equals the join-closure (in the direct product B(K_outer) ×
    B(K_inner)) of the atomic pairs {(μ_outer(g), μ_inner(g)) : g ∈ G} together
    with the bottom pair (⊥_outer, ⊥_inner).

    Returns: outer concept stable_id → set of filled inner concept stable_ids.
    """
    outer_concepts = list(outer_lattice.concepts)
    inner_concepts = list(inner_lattice.concepts)

    outer_ext: dict[str, frozenset] = {c.stable_id(): c.extent for c in outer_concepts}
    inner_ext: dict[str, frozenset] = {c.stable_id(): c.extent for c in inner_concepts}

    def _lat_join(ext_by_id: dict, concept_list: list, id1: str, id2: str) -> str:
        # Join = smallest concept whose extent ⊇ extent(id1) ∪ extent(id2).
        combined = ext_by_id[id1] | ext_by_id[id2]
        best = None
        for c in concept_list:
            if combined <= c.extent and (best is None or len(c.extent) < len(best.extent)):
                best = c
        return best.stable_id()

    def outer_join(id1: str, id2: str) -> str:
        return _lat_join(outer_ext, outer_concepts, id1, id2)

    def inner_join(id1: str, id2: str) -> str:
        return _lat_join(inner_ext, inner_concepts, id1, id2)

    # Atomic coordinate pairs — one per object.
    pairs: set[tuple[str, str]] = set()
    for obj_idx, oid in mu_outer.items():
        iid = mu_inner.get(obj_idx)
        if iid is not None:
            pairs.add((oid, iid))

    # ⊥_K projects to (⊥_outer, ⊥_inner).
    bot_outer = min(outer_concepts, key=lambda c: len(c.extent)).stable_id()
    bot_inner = min(inner_concepts, key=lambda c: len(c.extent)).stable_id()
    pairs.add((bot_outer, bot_inner))

    # Close under componentwise joins until fixpoint.
    changed = True
    while changed:
        changed = False
        pairs_list = list(pairs)
        for i in range(len(pairs_list)):
            for j in range(i + 1, len(pairs_list)):
                o1, i1 = pairs_list[i]
                o2, i2 = pairs_list[j]
                new_pair = (outer_join(o1, o2), inner_join(i1, i2))
                if new_pair not in pairs:
                    pairs.add(new_pair)
                    changed = True

    result: dict[str, set[str]] = {}
    for oid, iid in pairs:
        result.setdefault(oid, set()).add(iid)
    return result


def debug_filled_pairs(view: ExplorationView) -> None:
    """
    Print a mathematical verification report of the subdirect product filling.

    For every outer concept, prints:
      - outer concept stable_id, extent, intent
      - the filled inner concept stable_ids
      - for each filled inner concept: inner extent, inner intent
      - the generating reason: 'bottom pair', 'atomic(obj=<name>)', or
        'join(<pair1>, <pair2>)' (the earliest derivable reason in BFS order)

    Ends with a focused block for the bottom-most outer concept.
    """
    top_concept = view.lattice.top()
    top_id = top_concept.stable_id()
    template_view = view.children.get(top_id)

    if template_view is None:
        print("No inner template found (no child view at top outer concept).")
        return

    assert list(view.context.objects) == list(template_view.context.objects), (
        "Object order mismatch between outer and inner contexts."
    )

    mu_outer = _mu_local(view.lattice)
    mu_inner = _mu_local(template_view.lattice)

    outer_concepts = list(view.lattice.concepts)
    inner_concepts = list(template_view.lattice.concepts)

    outer_ext: dict[str, frozenset] = {c.stable_id(): c.extent for c in outer_concepts}
    inner_ext: dict[str, frozenset] = {c.stable_id(): c.extent for c in inner_concepts}
    outer_int: dict[str, frozenset] = {c.stable_id(): c.intent for c in outer_concepts}
    inner_int: dict[str, frozenset] = {c.stable_id(): c.intent for c in inner_concepts}

    outer_obj_names = view.context.objects
    inner_obj_names = template_view.context.objects
    outer_attr_names = view.context.attributes
    inner_attr_names = template_view.context.attributes

    def fmt_ext(ids, obj_names):
        return "{" + ", ".join(obj_names[i] for i in sorted(ids)) + "}" if ids else "∅"

    def fmt_int(ids, attr_names):
        return "{" + ", ".join(attr_names[i] for i in sorted(ids)) + "}" if ids else "∅"

    # --- Reproduce _compute_filled_pairs with provenance tracking ---
    def _lat_join(ext_by_id, concept_list, id1, id2):
        combined = ext_by_id[id1] | ext_by_id[id2]
        best = None
        for c in concept_list:
            if combined <= c.extent and (best is None or len(c.extent) < len(best.extent)):
                best = c
        return best.stable_id()

    def outer_join(id1, id2):
        return _lat_join(outer_ext, outer_concepts, id1, id2)

    def inner_join(id1, id2):
        return _lat_join(inner_ext, inner_concepts, id1, id2)

    # reason: str label for how each pair was derived
    pair_reason: dict[tuple[str, str], str] = {}

    bot_outer = min(outer_concepts, key=lambda c: len(c.extent)).stable_id()
    bot_inner = min(inner_concepts, key=lambda c: len(c.extent)).stable_id()
    pair_reason[(bot_outer, bot_inner)] = "bottom pair"

    for obj_idx, oid in mu_outer.items():
        iid = mu_inner.get(obj_idx)
        if iid is not None:
            pair = (oid, iid)
            obj_name = outer_obj_names[obj_idx]
            if pair not in pair_reason:
                pair_reason[pair] = f"atomic(obj={obj_name!r})"
            elif pair_reason[pair] == "bottom pair":
                pair_reason[pair] += f" + atomic(obj={obj_name!r})"

    changed = True
    while changed:
        changed = False
        pairs_list = list(pair_reason)
        for i in range(len(pairs_list)):
            for j in range(i + 1, len(pairs_list)):
                o1, i1 = pairs_list[i]
                o2, i2 = pairs_list[j]
                new_pair = (outer_join(o1, o2), inner_join(i1, i2))
                if new_pair not in pair_reason:
                    r1 = pair_reason[pairs_list[i]]
                    r2 = pair_reason[pairs_list[j]]
                    pair_reason[new_pair] = f"join({r1} ⊔ {r2})"
                    changed = True

    filled_by_outer: dict[str, set[str]] = {}
    for oid, iid in pair_reason:
        filled_by_outer.setdefault(oid, set()).add(iid)

    # --- Print report ---
    print("=" * 72)
    print("SUBDIRECT PRODUCT FILLING REPORT")
    print(f"Outer context: {view.context.n_objects} objects, "
          f"{view.context.n_attributes} attributes → "
          f"{view.lattice.n_concepts} outer concepts")
    print(f"Inner context: {template_view.context.n_objects} objects, "
          f"{template_view.context.n_attributes} attributes → "
          f"{template_view.lattice.n_concepts} inner concepts")
    print(f"Total filled (outer, inner) pairs: {len(pair_reason)}")
    print("=" * 72)

    outer_sorted = sorted(outer_concepts, key=lambda c: -len(c.extent))
    for c in outer_sorted:
        oid = c.stable_id()
        filled = filled_by_outer.get(oid, set())
        print(f"\nOUTER concept {oid}")
        print(f"  extent ({len(outer_ext[oid])}): {fmt_ext(outer_ext[oid], outer_obj_names)}")
        print(f"  intent ({len(outer_int[oid])}): {fmt_int(outer_int[oid], outer_attr_names)}")
        print(f"  filled inner nodes ({len(filled)}):")
        for iid in sorted(filled):
            reason = pair_reason.get((oid, iid), "?")
            iext_str = fmt_ext(inner_ext[iid], inner_obj_names)
            iint_str = fmt_int(inner_int[iid], inner_attr_names)
            print(f"    inner {iid}")
            print(f"      extent ({len(inner_ext[iid])}): {iext_str}")
            print(f"      intent ({len(inner_int[iid])}): {iint_str}")
            print(f"      reason: {reason}")

    # --- Bottom outer concept focused block ---
    print("\n" + "=" * 72)
    print(f"BOTTOM OUTER CONCEPT (⊥_outer): {bot_outer}")
    bot_filled = filled_by_outer.get(bot_outer, set())
    print(f"  extent ({len(outer_ext[bot_outer])}): "
          f"{fmt_ext(outer_ext[bot_outer], outer_obj_names)}")
    print(f"  intent ({len(outer_int[bot_outer])}): "
          f"{fmt_int(outer_int[bot_outer], outer_attr_names)}")
    print(f"  filled inner nodes ({len(bot_filled)}):")
    for iid in sorted(bot_filled):
        reason = pair_reason.get((bot_outer, iid), "?")
        print(f"    inner {iid}: extent={fmt_ext(inner_ext[iid], inner_obj_names)}, "
              f"intent={fmt_int(inner_int[iid], inner_attr_names)}")
        print(f"      reason: {reason}")
    print("=" * 72)


def debug_bottom_outer(view: ExplorationView) -> None:
    """
    Print a focused mathematical verification for the bottom-most outer concept.

    Covers:
      1. bottom outer concept id
      2. bottom outer extent
      3. bottom outer intent
      4. filled_by_outer[bottom_id]  — inner concept ids filled at this coordinate
      5. for each filled inner concept: id, extent, intent, and the reason it is
         filled (explicit bottom pair / atomic pair from object g / join-derived)
      6. total number of inner concepts
      7. number of filled inner concepts at this outer coordinate
      8. number of full B(K) concepts that project to this outer coordinate
         (equals 7 because φ: B(K) → B(K_outer) × B(K_inner) is injective)

    Also verifies that outer and inner contexts share the same object ordering —
    required for the obj_idx pairing in _compute_filled_pairs to be correct.

    Call this after building the ExplorationView tree but before or after
    plot_nested; it does not affect the diagram.
    """
    top_concept = view.lattice.top()
    top_id = top_concept.stable_id()
    template_view = view.children.get(top_id)

    if template_view is None:
        print("[debug_bottom_outer] No inner template found — "
              "no child view at top outer concept.")
        return

    # --- Object-order check ---
    outer_objs = list(view.context.objects)
    inner_objs = list(template_view.context.objects)
    if outer_objs == inner_objs:
        print("Object order between outer and inner contexts matches.")
    else:
        print("WARNING: object order MISMATCH between outer and inner contexts!")
        print(f"  outer: {outer_objs[:6]}...")
        print(f"  inner: {inner_objs[:6]}...")
        return

    # --- Setup ---
    mu_outer = _mu_local(view.lattice)
    mu_inner = _mu_local(template_view.lattice)

    outer_concepts = list(view.lattice.concepts)
    inner_concepts = list(template_view.lattice.concepts)

    outer_ext: dict[str, frozenset] = {c.stable_id(): c.extent for c in outer_concepts}
    inner_ext: dict[str, frozenset] = {c.stable_id(): c.extent for c in inner_concepts}
    outer_int: dict[str, frozenset] = {c.stable_id(): c.intent for c in outer_concepts}
    inner_int: dict[str, frozenset] = {c.stable_id(): c.intent for c in inner_concepts}

    outer_obj_names = view.context.objects
    inner_obj_names = template_view.context.objects
    outer_attr_names = view.context.attributes
    inner_attr_names = template_view.context.attributes

    def fmt_ext(ids, obj_names):
        return "{" + ", ".join(obj_names[i] for i in sorted(ids)) + "}" if ids else "∅"

    def fmt_int(ids, attr_names):
        return "{" + ", ".join(attr_names[i] for i in sorted(ids)) + "}" if ids else "∅"

    # --- Reproduce _compute_filled_pairs with provenance tracking ---
    def _lat_join(ext_by_id, concept_list, id1, id2):
        combined = ext_by_id[id1] | ext_by_id[id2]
        best = None
        for c in concept_list:
            if combined <= c.extent and (best is None or len(c.extent) < len(best.extent)):
                best = c
        return best.stable_id()

    def outer_join(id1, id2):
        return _lat_join(outer_ext, outer_concepts, id1, id2)

    def inner_join(id1, id2):
        return _lat_join(inner_ext, inner_concepts, id1, id2)

    pair_reason: dict[tuple[str, str], str] = {}

    bot_outer = min(outer_concepts, key=lambda c: len(c.extent)).stable_id()
    bot_inner = min(inner_concepts, key=lambda c: len(c.extent)).stable_id()
    pair_reason[(bot_outer, bot_inner)] = "bottom pair (⊥_outer, ⊥_inner)"

    for obj_idx, oid in mu_outer.items():
        iid = mu_inner.get(obj_idx)
        if iid is not None:
            pair = (oid, iid)
            obj_name = outer_obj_names[obj_idx]
            if pair not in pair_reason:
                pair_reason[pair] = f"atomic pair: μ_outer({obj_name!r})={oid}, μ_inner({obj_name!r})={iid}"
            elif "bottom pair" in pair_reason[pair]:
                pair_reason[pair] += f" + atomic pair: μ_outer({obj_name!r})={oid}, μ_inner({obj_name!r})={iid}"
            else:
                pair_reason[pair] += f" + atomic({obj_name!r})"

    changed = True
    while changed:
        changed = False
        pairs_list = list(pair_reason)
        for i in range(len(pairs_list)):
            for j in range(i + 1, len(pairs_list)):
                o1, i1 = pairs_list[i]
                o2, i2 = pairs_list[j]
                new_pair = (outer_join(o1, o2), inner_join(i1, i2))
                if new_pair not in pair_reason:
                    pair_reason[new_pair] = (
                        f"join-derived: ({o1},{i1}) ⊔ ({o2},{i2}) "
                        f"= ({new_pair[0]},{new_pair[1]})"
                    )
                    changed = True

    filled_by_outer: dict[str, set[str]] = {}
    for oid, iid in pair_reason:
        filled_by_outer.setdefault(oid, set()).add(iid)

    # --- Print ---
    sep = "=" * 72
    print(sep)
    print("BOTTOM OUTER CONCEPT — SUBDIRECT PRODUCT VERIFICATION")
    print(sep)

    # 1. id
    print(f"\n1. Bottom outer concept id:  {bot_outer}")

    # 2. extent
    bot_outer_ext = outer_ext[bot_outer]
    print(f"2. Bottom outer extent ({len(bot_outer_ext)}): "
          f"{fmt_ext(bot_outer_ext, outer_obj_names)}")

    # 3. intent
    bot_outer_int = outer_int[bot_outer]
    print(f"3. Bottom outer intent ({len(bot_outer_int)}): "
          f"{fmt_int(bot_outer_int, outer_attr_names)}")

    # 4. filled inner ids
    bot_filled = filled_by_outer.get(bot_outer, set())
    print(f"\n4. filled_by_outer[{bot_outer}]:")
    print(f"   {sorted(bot_filled) if bot_filled else '(empty)'}")

    # 5. per-filled-inner breakdown
    print(f"\n5. Filled inner concepts (detailed):")
    for iid in sorted(bot_filled):
        reason = pair_reason.get((bot_outer, iid), "?")
        iext = inner_ext[iid]
        iint = inner_int[iid]
        print(f"\n   inner concept id: {iid}")
        print(f"     extent ({len(iext)}): {fmt_ext(iext, inner_obj_names)}")
        print(f"     intent ({len(iint)}): {fmt_int(iint, inner_attr_names)}")
        print(f"     reason: {reason}")

    # 6. total inner concepts
    print(f"\n6. Total inner concepts in B(K_inner):  {len(inner_concepts)}")

    # 7. filled inner count
    print(f"7. Filled inner concepts at ⊥_outer:    {len(bot_filled)}")

    # 8. full concepts projecting to ⊥_outer
    #    φ: B(K) → B(K_outer) × B(K_inner) is injective, so each filled
    #    coordinate pair (⊥_outer, inner_id) corresponds to exactly one
    #    concept in B(K).  Therefore item 8 = item 7.
    print(f"8. Full B(K) concepts projecting to ⊥_outer: {len(bot_filled)}")
    print(f"   (equals item 7: φ is injective, so each filled coordinate pair")
    print(f"    (⊥_outer, inner_id) is the image of exactly one concept in B(K))")

    print(f"\n{sep}")


def exploration_view_to_nested_data(
    view: ExplorationView,
    object_label: Callable[[str], str] | None = None,
) -> dict[str, Any]:
    """
    Convert an ExplorationView tree into nested JSON-like data, recursively.

    Works for arbitrarily deep ExplorationView trees.

    Visual consistency: each click-through view uses the same node structure
    and positions as the template preview (derived from the ⊤ outer concept's
    child view), so the diagram seen after clicking always matches the preview
    drawn inside the bubble.

    For N-layer depth, click-through nodes also carry has_child/child and
    filled_inner for the next layer.  These are matched from the per-concept
    child view (built from the restricted object set) using intent-based lookup:
    attribute indices are invariant across views built from the same scale, so
    a concept's intent frozenset is a reliable cross-view key.
    """
    graph_data = lattice_to_graph_data_dimflux(view.lattice, stable_ids=True)

    # Reduced labeling: γ and μ for this view's lattice.
    gamma = _gamma_local(view.lattice)
    mu = _mu_local(view.lattice)

    by_concept_attrs: dict[str, list[str]] = {
        c.stable_id(): [] for c in view.lattice.concepts
    }
    by_concept_objs: dict[str, list[str]] = {
        c.stable_id(): [] for c in view.lattice.concepts
    }

    for attr_idx, cid in gamma.items():
        by_concept_attrs[cid].append(view.context.attributes[attr_idx])
    for obj_idx, cid in mu.items():
        by_concept_objs[cid].append(view.context.objects[obj_idx])

    # Subdirect decomposition: shared inner template from the ⊤ outer concept.
    top_concept = view.lattice.top()
    top_id = top_concept.stable_id()
    template_view = view.children.get(top_id)

    inner_template = None
    mu_by_name: dict[str, str] = {}
    inner_scale_names: list[str] = []
    filled_by_outer: dict[str, set[str]] = {}

    if template_view is not None:
        tdata = _inner_template_data(template_view)
        mu_by_name = tdata.pop("mu_by_name")
        inner_template = tdata
        inner_scale_names = list(template_view.scale_names)

        # Subdirect product filling: join-closure of coordinate pairs.
        # obj_idx must refer to the same object in both lattice contexts.
        assert list(view.context.objects) == list(template_view.context.objects), (
            "Object order mismatch between outer and inner contexts: "
            f"outer={list(view.context.objects)[:5]}... "
            f"inner={list(template_view.context.objects)[:5]}..."
        )

        # Apposition requires M_outer ∩ M_inner = ∅.  Without disjointness the
        # join-closure of factor-lattice pairs no longer equals image(φ) and the
        # filled-node computation is silently wrong.
        outer_attrs = set(view.context.attributes)
        inner_attrs = set(template_view.context.attributes)
        overlap = outer_attrs & inner_attrs
        if overlap:
            raise ValueError(
                "Outer and inner attribute sets must be disjoint for the subdirect "
                "product embedding to be valid (apposition K = K_outer | K_inner), "
                f"but the following attributes appear in both: {sorted(overlap)}"
            )

        mu_inner = _mu_local(template_view.lattice)
        filled_by_outer = _compute_filled_pairs(view.lattice, template_view.lattice, mu, mu_inner)

    _fmt_obj: Callable[[str], str] = object_label if object_label is not None else (lambda x: x)

    nodes: list[dict[str, Any]] = []

    for node in graph_data.nodes:
        new_objs = set(by_concept_objs[node.node_id])

        # Filled inner nodes: those whose coordinate pair (this outer concept,
        # inner concept) lies in the subdirect product of B(K).
        filled_inner = list(filled_by_outer.get(node.node_id, set()))

        # Build click-through child view for this concept.
        #
        # Two cases:
        #   inner_template present → hybrid: template structure for visual
        #     consistency + N-layer children from per-concept view via intent match.
        #   inner_template absent  → fallback: direct recursive child view
        #     (used when the top concept has no child, e.g. partial expansions).
        child_data = None
        if inner_template is not None:
            filled_set = filled_by_outer.get(node.node_id, set())

            # Per-concept child view: built from objects in this outer concept.
            per_concept_view = view.children.get(node.node_id)

            # Recursively build the per-concept data to get N-layer children
            # and the next-layer inner_template.
            per_concept_data: dict[str, Any] | None = None
            if per_concept_view is not None:
                per_concept_data = exploration_view_to_nested_data(
                    per_concept_view, object_label
                )

            # Intent-based lookup: attribute indices are invariant across views
            # built from the same scale, so frozenset(intent indices) is a
            # reliable cross-view key even though stable_ids differ.
            pnode_by_intent: dict[frozenset, dict[str, Any]] = {}
            if per_concept_data is not None:
                for pnode in per_concept_data["nodes"]:
                    pnode_by_intent[frozenset(pnode["metadata"]["intent"])] = pnode

            child_nodes: list[dict[str, Any]] = []
            for inode in inner_template["nodes"]:
                iid = inode["id"]
                # Object labels: objects introduced at THIS outer concept
                # whose μ_inner maps to this inner template node.
                obj_labels = sorted(
                    _fmt_obj(g) for g in new_objs if mu_by_name.get(g) == iid
                )

                # Match to per-concept view node by intent.
                pnode = pnode_by_intent.get(frozenset(inode["intent"]))
                has_child = pnode is not None and pnode["has_child"]
                child = pnode["child"] if has_child else None
                pc_filled_inner = pnode["filled_inner"] if pnode is not None else []

                child_nodes.append({
                    "id": iid,
                    "label": "",
                    "hover": iid,
                    "x": inode["x"],
                    "y": inode["y"],
                    "new_attrs": inode["new_attrs"],
                    "new_objects": obj_labels,
                    "filled_inner": pc_filled_inner,
                    "is_filled": iid in filled_set,
                    "has_child": has_child,
                    "child": child,
                    "metadata": {
                        "extent": inode["extent"],
                        "extent_size": inode["extent_size"],
                        "intent": inode["intent"],
                        "intent_size": inode["intent_size"],
                    },
                })

            # Inner template for the next layer comes from the per-concept view.
            next_inner_template = (
                per_concept_data["inner_template"]
                if per_concept_data is not None
                else None
            )

            child_data = {
                "name": node.label,
                "depth": template_view.depth,
                "scale_names": inner_scale_names,
                "inner_template": next_inner_template,
                "nodes": child_nodes,
                "edges": inner_template["edges"],
                "metadata": {
                    "context_size": (node.metadata["extent_size"], 0),
                    "n_concepts": len(inner_template["nodes"]),
                    "n_children": (
                        len(per_concept_view.children)
                        if per_concept_view is not None else 0
                    ),
                },
            }

        # Fallback for partial expansions: no template, but this specific
        # concept has a child view attached directly.
        if child_data is None and inner_template is None:
            child_view = view.children.get(node.node_id)
            if child_view is not None:
                child_data = exploration_view_to_nested_data(child_view, object_label)

        nodes.append(
            {
                "id": node.node_id,
                "label": node.label,
                "hover": node.hover_text,
                "x": node.x,
                "y": node.y,
                "metadata": node.metadata,
                "has_child": child_data is not None,
                "child": child_data,
                "new_attrs": sorted(by_concept_attrs[node.node_id]),
                "new_objects": sorted(_fmt_obj(g) for g in by_concept_objs[node.node_id]),
                "filled_inner": filled_inner,
            }
        )

    edges = [
        {
            "source": edge.source,
            "target": edge.target,
            "metadata": edge.metadata,
        }
        for edge in graph_data.edges
    ]

    return {
        "name": view.name,
        "depth": view.depth,
        "scale_names": list(view.scale_names),
        "nodes": nodes,
        "edges": edges,
        "inner_template": inner_template,
        "metadata": {
            "context_size": (
                view.context.n_objects,
                view.context.n_attributes,
            ),
            "n_concepts": view.lattice.n_concepts,
            "n_children": len(view.children),
        },
    }


def render_nested_html(
    data: dict[str, Any],
    width: int = 900,
    height: int = 700,
    title: str | None = None,
    show_child_previews: bool = True,
) -> str:
    """
    Render nested exploration data as an interactive D3 view with subdirect
    decomposition.

    Interaction model
    -----------------
    - Click a node with a child view: navigate into that child view.
    - Back / Root: navigate up.
    - Nodes at the outer level show the shared inner template (hollow/filled).
    """
    data_json = json.dumps(data)
    display_title = title or data["name"]
    show_child_previews_json = json.dumps(show_child_previews)

    return f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<script src="https://cdn.jsdelivr.net/npm/d3@7"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/jspdf/2.5.1/jspdf.umd.min.js"></script>
<style>
  body {{
    margin: 0;
    font-family: sans-serif;
    background: #ffffff;
  }}

  .conceptflow-container {{
    width: {width}px;
    max-width: 100%;
    border: 1px solid #ddd;
    border-radius: 10px;
    overflow: hidden;
    background: #fafafa;
  }}

  .conceptflow-header {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 12px;
    padding: 10px 14px;
    background: #f2f2f2;
    border-bottom: 1px solid #ddd;
  }}

  .conceptflow-title {{
    font-size: 16px;
    font-weight: 600;
  }}

  .conceptflow-subtitle {{
    font-size: 12px;
    color: #666;
    margin-top: 2px;
  }}

  .conceptflow-controls {{
    display: flex;
    gap: 6px;
  }}

  .conceptflow-button {{
    border: 1px solid #aaa;
    background: white;
    border-radius: 6px;
    padding: 5px 10px;
    cursor: pointer;
    font-size: 12px;
  }}

  .conceptflow-button:disabled {{
    opacity: 0.4;
    cursor: not-allowed;
  }}

  .conceptflow-button:hover:not(:disabled) {{
    background: #f7f7f7;
  }}

  .conceptflow-body {{
    display: grid;
    grid-template-columns: 1fr 280px;
    min-height: {height}px;
  }}

  .conceptflow-details {{
    border-left: 1px solid #ddd;
    background: #fafafa;
    padding: 12px;
    font-size: 12px;
    overflow: auto;
  }}

  .details-title {{
    font-weight: 700;
    margin-bottom: 8px;
  }}

  .details-section {{
    margin-top: 10px;
  }}

  .details-label {{
    font-weight: 600;
    color: #555;
  }}

  .breadcrumb {{
    font-size: 11px;
    color: #555;
    margin-top: 4px;
  }}

  svg {{
    display: block;
    background: white;
    cursor: grab;
  }}

  svg:active {{
    cursor: grabbing;
  }}

  .edge {{
    stroke: #555;
    stroke-width: 1.5;
    opacity: 0.65;
  }}

  .edge.highlighted {{
    stroke: #1f4e79;
    stroke-width: 3;
    opacity: 1;
  }}

  .node {{
    fill: white;
    stroke: black;
    stroke-width: 1.5;
    cursor: pointer;
  }}

  .node-with-child {{
    fill: #f8fbff;
    stroke: #1f4e79;
    stroke-width: 2.2;
    cursor: pointer;
  }}

  .node:hover,
  .node-with-child:hover {{
    stroke: #d97706;
    stroke-width: 3;
  }}

  .node.selected,
  .node-with-child.selected {{
    stroke: #d97706;
    stroke-width: 3.5;
  }}

  .inner-edge {{
    stroke: #777;
    stroke-width: 0.8;
    opacity: 0.8;
  }}

  .inner-node {{
    fill: white;
    stroke: #333;
    stroke-width: 0.8;
  }}

  .inner-node.filled {{
    fill: #1f4e79;
    stroke: #1f4e79;
  }}

  .node.filled {{
    fill: #1f4e79;
    stroke: black;
  }}


  /* Attribute labels: placed ABOVE node, small caps style */
  .attr-label {{
    font-size: 10px;
    text-anchor: middle;
    dominant-baseline: auto;
    fill: #1a1a2e;
    font-weight: 600;
    pointer-events: none;
    paint-order: stroke;
    stroke: white;
    stroke-width: 4px;
    stroke-linejoin: round;
  }}

  /* Object labels: placed BELOW node, italic */
  .obj-label {{
    font-size: 9px;
    text-anchor: middle;
    dominant-baseline: hanging;
    fill: #444;
    font-style: italic;
    pointer-events: none;
    paint-order: stroke;
    stroke: white;
    stroke-width: 3px;
    stroke-linejoin: round;
  }}

  /* Attribute labels inside the inner template preview */
  .inner-attr {{
    font-size: 5px;
    text-anchor: middle;
    dominant-baseline: auto;
    fill: #1a1a2e;
    pointer-events: none;
    paint-order: stroke;
    stroke: white;
    stroke-width: 2px;
    stroke-linejoin: round;
  }}
</style>
</head>
<body>
<div class="conceptflow-container">
  <div class="conceptflow-header">
    <div>
      <div id="conceptflow-title" class="conceptflow-title"></div>
      <div id="conceptflow-subtitle" class="conceptflow-subtitle"></div>
      <div id="conceptflow-breadcrumb" class="breadcrumb"></div>
    </div>
    <div class="conceptflow-controls">
      <button id="conceptflow-back" class="conceptflow-button">Back</button>
      <button id="conceptflow-root" class="conceptflow-button">Root</button>
      <button id="conceptflow-reset-zoom" class="conceptflow-button">Reset zoom</button>
      <button id="conceptflow-save-svg" class="conceptflow-button">Save SVG</button>
      <button id="conceptflow-save-png" class="conceptflow-button">Save PNG</button>
      <button id="conceptflow-save-pdf" class="conceptflow-button">Save PDF</button>
    </div>
  </div>

  <div class="conceptflow-body">
    <svg id="conceptflow-nested-svg" width="{width - 280}" height="{height}"></svg>

    <div id="conceptflow-details" class="conceptflow-details">
      <div class="details-title">Details</div>
      <div>Click a concept node to inspect its extent and intent.</div>
    </div>
  </div>
</div>

<script>
const rootData = {data_json};
const showChildPreviews = {show_child_previews_json};
const svgWidth = {width - 280};
const svgHeight = {height};

const svg = d3.select("#conceptflow-nested-svg");
const details = d3.select("#conceptflow-details");

const titleEl = d3.select("#conceptflow-title");
const subtitleEl = d3.select("#conceptflow-subtitle");
const breadcrumbEl = d3.select("#conceptflow-breadcrumb");

const backButton = d3.select("#conceptflow-back");
const rootButton = d3.select("#conceptflow-root");
const resetZoomButton = d3.select("#conceptflow-reset-zoom");

let currentView = rootData;
let viewStack = [];
let nameStack = [rootData.name];

let rootLayer = svg.append("g").attr("class", "root-layer");

const zoom = d3.zoom()
  .scaleExtent([0.3, 8])
  .on("zoom", (event) => {{
    rootLayer.attr("transform", event.transform);
  }});

svg.call(zoom);

function resetZoom(duration = 400) {{
  svg.transition()
    .duration(duration)
    .call(zoom.transform, d3.zoomIdentity);
}}

// Works for both outer nodes (metadata.extent_size) and inner template nodes (extent array).
function extentSize(d) {{
  if (d.metadata != null && d.metadata.extent_size !== undefined) return d.metadata.extent_size;
  if (d.extent != null) return d.extent.length;
  return 0;
}}

function computeTransform(nodes, targetWidth, targetHeight, padding) {{
  const topNode = nodes.reduce((best, d) =>
    extentSize(d) > extentSize(best) ? d : best, nodes[0]);
  const bottomNode = nodes.reduce((best, d) =>
    extentSize(d) < extentSize(best) ? d : best, nodes[0]);

  const bx = bottomNode.x ?? 0;
  const by = bottomNode.y ?? 0;
  const tx = topNode.x ?? 0;
  const ty = topNode.y ?? 0;

  const vx = tx - bx;
  const vy = ty - by;

  // Degenerate case: single node or all at same position.
  if (Math.abs(vx) < 0.001 && Math.abs(vy) < 0.001) {{
    return () => ({{ x: targetWidth / 2, y: targetHeight / 2 }});
  }}

  const currentAngle = Math.atan2(vy, vx);
  const desiredAngle = -Math.PI / 2;
  const angle = desiredAngle - currentAngle;

  const cosA = Math.cos(angle);
  const sinA = Math.sin(angle);

  function rotatePoint(d) {{
    const x = d.x ?? 0;
    const y = d.y ?? 0;
    return {{ x: x * cosA - y * sinA, y: x * sinA + y * cosA }};
  }}

  const rotatedById = new Map(nodes.map(d => [d.id, rotatePoint(d)]));
  const rotated = nodes.map(d => rotatedById.get(d.id));

  const xs = rotated.map(d => d.x);
  const ys = rotated.map(d => d.y);

  const minX = d3.min(xs);
  const maxX = d3.max(xs);
  const minY = d3.min(ys);
  const maxY = d3.max(ys);

  const spanX = Math.max(maxX - minX, 1);
  const spanY = Math.max(maxY - minY, 1);

  const scale = Math.min(
    (targetWidth - 2 * padding) / spanX,
    (targetHeight - 2 * padding) / spanY
  );

  return function(d) {{
    const p = rotatedById.get(d.id);
    return {{
      x: padding + (p.x - minX) * scale
        + (targetWidth - 2 * padding - spanX * scale) / 2,
      y: padding + (p.y - minY) * scale
        + (targetHeight - 2 * padding - spanY * scale) / 2,
    }};
  }};
}}

function formatList(values) {{
  if (!values || values.length === 0) return "∅";
  return values.join(", ");
}}

function formatAttrLabel(name) {{
  // "key_mode=minor" → "minor", "key_mode=major" → "major"
  // (but leave "bpm>=100" unchanged — the char before = is >)
  const idx = name.indexOf('=');
  if (idx > 0 && name[idx - 1] !== '>' && name[idx - 1] !== '<') {{
    return name.slice(idx + 1);
  }}
  // "political_support" → "Political Support"
  return name.replace(/_/g, ' ')
             .split(' ')
             .map(w => w.charAt(0).toUpperCase() + w.slice(1))
             .join(' ');
}}

function showDefaultDetails() {{
  details.html(`
    <div class="details-title">Details</div>
    <div>Click a concept node to inspect its extent and intent.</div>
    <div class="details-section">
      <div class="details-label">Current view</div>
      <div>${{currentView.name}}</div>
    </div>
    <div class="details-section">
      <div class="details-label">Scales</div>
      <div>${{formatList(currentView.scale_names)}}</div>
    </div>
    <div class="details-section">
      <div class="details-label">Concepts</div>
      <div>${{currentView.metadata?.n_concepts ?? currentView.nodes.length}}</div>
    </div>
  `);
}}

function showDetails(d) {{
  details.html(`
    <div class="details-title">Concept details</div>
    <div class="details-section">
      <div class="details-label">Attributes introduced here</div>
      <div>${{formatList(d.new_attrs)}}</div>
    </div>
    <div class="details-section">
      <div class="details-label">Objects introduced here</div>
      <div>${{formatList(d.new_objects)}}</div>
    </div>
    <div class="details-section">
      <div class="details-label">Extent (${{d.metadata?.extent_size ?? 0}})</div>
      <div>${{formatList(d.metadata?.extent)}}</div>
    </div>
    <div class="details-section">
      <div class="details-label">Intent (${{d.metadata?.intent_size ?? 0}})</div>
      <div>${{formatList(d.metadata?.intent)}}</div>
    </div>
    <div class="details-section">
      <div class="details-label">Filled inner nodes</div>
      <div>${{d.filled_inner ? d.filled_inner.length : 0}}</div>
    </div>
    <div class="details-section">
      <div class="details-label">Action</div>
      <div>${{d.has_child ? "Click again to enter this child view." : "No child view."}}</div>
    </div>
  `);
}}

function updateHeader() {{
  titleEl.text(currentView === rootData ? "{display_title}" : currentView.name);
  const scales = currentView.scale_names && currentView.scale_names.length
    ? currentView.scale_names.join(", ")
    : "none";
  subtitleEl.text(`view depth ${{currentView.depth}} · scales: ${{scales}} · concepts: ${{currentView.nodes.length}}`);
  breadcrumbEl.text(nameStack.join(" → "));
  backButton.property("disabled", viewStack.length === 0);
  rootButton.property("disabled", currentView === rootData);
}}

function enterView(childView) {{
  viewStack.push(currentView);
  nameStack.push(childView.name);
  currentView = childView;
  renderView(currentView);
}}

function goBack() {{
  if (viewStack.length === 0) return;
  currentView = viewStack.pop();
  nameStack.pop();
  renderView(currentView);
}}

function goRoot() {{
  currentView = rootData;
  viewStack = [];
  nameStack = [rootData.name];
  renderView(currentView);
}}

// ---------------------------------------------------------------------
// Subdirect decomposition preview.
//
// Draw the shared inner template inside an outer node bubble.
// Nodes in `filledInner` are rendered filled (dark blue); all others hollow.
// Attribute labels appear above each inner node where introduced (γ(m)).
// Object labels are omitted in the preview to avoid clutter.
// ---------------------------------------------------------------------
function drawTemplatePreview(parentGroup, innerTemplate, filledInner, radius, showAttrLabels) {{
  if (!innerTemplate || innerTemplate.nodes.length === 0) return;

  const filledSet = new Set(filledInner);
  const innerSize = radius * 1.55;
  const padding = 8;

  const transform = computeTransform(innerTemplate.nodes, innerSize, innerSize, padding);

  const childGroup = parentGroup
    .append("g")
    .attr("class", "child-preview")
    .attr("transform", `translate(${{-innerSize / 2}}, ${{-innerSize / 2}})`);

  const nodeById = new Map(innerTemplate.nodes.map(d => [d.id, d]));

  childGroup
    .selectAll("line.inner-edge")
    .data(innerTemplate.edges)
    .join("line")
    .attr("class", "inner-edge")
    .attr("x1", d => transform(nodeById.get(d.source)).x)
    .attr("y1", d => transform(nodeById.get(d.source)).y)
    .attr("x2", d => transform(nodeById.get(d.target)).x)
    .attr("y2", d => transform(nodeById.get(d.target)).y);

  childGroup
    .selectAll("circle.inner-node")
    .data(innerTemplate.nodes)
    .join("circle")
    .attr("class", d => filledSet.has(d.id) ? "inner-node filled" : "inner-node")
    .attr("cx", d => transform(d).x)
    .attr("cy", d => transform(d).y)
    .attr("r", 3.5);

  // Attribute labels only in the designated reference bubble (top outer node).
  if (showAttrLabels) {{
    childGroup
      .selectAll("text.inner-attr")
      .data(innerTemplate.nodes.filter(n => n.new_attrs && n.new_attrs.length > 0))
      .join("text")
      .attr("class", "inner-attr")
      .attr("x", d => transform(d).x)
      .attr("y", d => transform(d).y - 6)
      .text(d => d.new_attrs.map(formatAttrLabel).join(", "));
  }}
}}

// Fallback for inner views (when no template is present at this level).
function drawChildPreview(parentGroup, childData, radius) {{
  if (!childData) return;

  const innerSize = radius * 1.35;
  const padding = 8;

  const transform = computeTransform(childData.nodes, innerSize, innerSize, padding);

  const childGroup = parentGroup
    .append("g")
    .attr("class", "child-preview")
    .attr("transform", `translate(${{-innerSize / 2}}, ${{-innerSize / 2}})`);

  const nodeById = new Map(childData.nodes.map(d => [d.id, d]));

  childGroup
    .selectAll("line.inner-edge")
    .data(childData.edges)
    .join("line")
    .attr("class", "inner-edge")
    .attr("x1", d => transform(nodeById.get(d.source)).x)
    .attr("y1", d => transform(nodeById.get(d.source)).y)
    .attr("x2", d => transform(nodeById.get(d.target)).x)
    .attr("y2", d => transform(nodeById.get(d.target)).y);

  childGroup
    .selectAll("circle.inner-node")
    .data(childData.nodes)
    .join("circle")
    .attr("class", "inner-node")
    .attr("cx", d => transform(d).x)
    .attr("cy", d => transform(d).y)
    .attr("r", 4);
}}

function renderView(viewData) {{
  currentView = viewData;

  rootLayer.selectAll("*").remove();
  resetZoom(0);
  updateHeader();
  showDefaultDetails();

  const hasTemplate = !!viewData.inner_template;

  const transform = computeTransform(viewData.nodes, svgWidth, svgHeight, 100);
  const nodeById = new Map(viewData.nodes.map(d => [d.id, d]));

  viewData.nodes.forEach(d => {{
    const p = transform(d);
    d.screenX = p.x;
    d.screenY = p.y;
    // Outer level: uniform radius so the template preview fits uniformly.
    // Inner level (click-through): all nodes same size.
    d.radius = hasTemplate ? 48 : 18;
  }});

  const edgeSelection = rootLayer
    .append("g")
    .selectAll("line.edge")
    .data(viewData.edges)
    .join("line")
    .attr("class", "edge")
    .attr("x1", d => nodeById.get(d.source).screenX)
    .attr("y1", d => nodeById.get(d.source).screenY)
    .attr("x2", d => nodeById.get(d.target).screenX)
    .attr("y2", d => nodeById.get(d.target).screenY);

  const nodeGroup = rootLayer
    .append("g")
    .selectAll("g.node-group")
    .data(viewData.nodes)
    .join("g")
    .attr("class", "node-group")
    .attr("transform", d => `translate(${{d.screenX}}, ${{d.screenY}})`);

  const circles = nodeGroup
    .append("circle")
    .attr("class", d => {{
      if (d.has_child) return "node-with-child";
      if (d.is_filled) return "node filled";
      return "node";
    }})
    .attr("r", d => d.radius)
    .on("mouseover", function(event, d) {{
      d3.select(this).classed("selected", true);
      edgeSelection.classed("highlighted", e =>
        e.source === d.id || e.target === d.id
      );
    }})
    .on("mouseout", function(event, d) {{
      d3.select(this).classed("selected", false);
      edgeSelection.classed("highlighted", false);
    }})
    .on("click", function(event, d) {{
      event.stopPropagation();
      showDetails(d);
      if (d.has_child && d.child) {{
        enterView(d.child);
      }}
    }});

  circles.append("title").text(d => d.hover);

  // Draw inner template preview (subdirect decomposition) or plain child preview.
  // Attr labels shown only in the top outer node's bubble (largest extent).
  const topOuterId = viewData.nodes.reduce((best, d) =>
    extentSize(d) > extentSize(best) ? d : best, viewData.nodes[0]).id;

  if (showChildPreviews) {{
    if (hasTemplate) {{
      nodeGroup.each(function(d) {{
        drawTemplatePreview(
          d3.select(this),
          viewData.inner_template,
          d.filled_inner || [],
          d.radius,
          d.id === topOuterId
        );
      }});
    }} else {{
      nodeGroup.each(function(d) {{
        if (d.has_child && d.child) {{
          drawChildPreview(d3.select(this), d.child, d.radius);
        }}
      }});
    }}
  }}

  // Attribute labels ABOVE each node — comma-separated on one line.
  nodeGroup.each(function(d) {{
    if (!d.new_attrs || d.new_attrs.length === 0) return;
    d3.select(this).append("text")
      .attr("class", "attr-label")
      .attr("x", 0)
      .attr("y", -(d.radius + 10))
      .text(d.new_attrs.map(formatAttrLabel).join(", "));
  }});

  // Object labels BELOW each node — comma-separated on one line.
  nodeGroup.each(function(d) {{
    if (!d.new_objects || d.new_objects.length === 0) return;
    d3.select(this).append("text")
      .attr("class", "obj-label")
      .attr("x", 0)
      .attr("y", d.radius + 13)
      .text(d.new_objects.join(", "));
  }});
}}

function getStyledSVGString() {{
  const svgEl = document.getElementById("conceptflow-nested-svg");
  const clone = svgEl.cloneNode(true);
  const css = `
    .edge {{ stroke: #555; stroke-width: 1.5; opacity: 0.65; }}
    .edge.highlighted {{ stroke: #1f4e79; stroke-width: 3; opacity: 1; }}
    .node {{ fill: white; stroke: black; stroke-width: 1.5; }}
    .node-with-child {{ fill: #f8fbff; stroke: #1f4e79; stroke-width: 2.2; }}
    .node.filled {{ fill: #1f4e79; stroke: black; }}
    .node.selected, .node-with-child.selected {{ stroke: #d97706; stroke-width: 3.5; }}
    .inner-edge {{ stroke: #777; stroke-width: 0.8; opacity: 0.8; }}
    .inner-node {{ fill: white; stroke: #333; stroke-width: 0.8; }}
    .inner-node.filled {{ fill: #1f4e79; stroke: #1f4e79; }}
    .attr-label {{ font-size: 10px; text-anchor: middle; dominant-baseline: auto; fill: #1a1a2e; font-weight: 600; paint-order: stroke; stroke: white; stroke-width: 4px; stroke-linejoin: round; }}
    .obj-label {{ font-size: 9px; text-anchor: middle; dominant-baseline: hanging; fill: #444; font-style: italic; paint-order: stroke; stroke: white; stroke-width: 3px; stroke-linejoin: round; }}
    .inner-attr {{ font-size: 5px; text-anchor: middle; dominant-baseline: auto; fill: #1a1a2e; paint-order: stroke; stroke: white; stroke-width: 2px; stroke-linejoin: round; }}
  `;
  const styleEl = document.createElementNS("http://www.w3.org/2000/svg", "style");
  styleEl.textContent = css;
  clone.insertBefore(styleEl, clone.firstChild);
  clone.setAttribute("xmlns", "http://www.w3.org/2000/svg");
  return new XMLSerializer().serializeToString(clone);
}}

function saveSVG() {{
  const svgStr = getStyledSVGString();
  const blob = new Blob([svgStr], {{type: "image/svg+xml;charset=utf-8"}});
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = (currentView.name || "diagram").replace(/[^a-z0-9]/gi, "_").toLowerCase() + ".svg";
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}}

function savePNG() {{
  const svgEl = document.getElementById("conceptflow-nested-svg");
  const w = parseInt(svgEl.getAttribute("width"));
  const h = parseInt(svgEl.getAttribute("height"));
  const svgStr = getStyledSVGString();
  const canvas = document.createElement("canvas");
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d");
  ctx.fillStyle = "white";
  ctx.fillRect(0, 0, w, h);
  const blob = new Blob([svgStr], {{type: "image/svg+xml;charset=utf-8"}});
  const url = URL.createObjectURL(blob);
  const img = new Image();
  img.onload = function() {{
    ctx.drawImage(img, 0, 0);
    URL.revokeObjectURL(url);
    const name = (currentView.name || "diagram").replace(/[^a-z0-9]/gi, "_").toLowerCase();
    canvas.toBlob(function(pngBlob) {{
      const pngUrl = URL.createObjectURL(pngBlob);
      const a = document.createElement("a");
      a.href = pngUrl;
      a.download = name + ".png";
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(pngUrl);
    }});
  }};
  img.src = url;
}}

backButton.on("click", goBack);
rootButton.on("click", goRoot);
resetZoomButton.on("click", () => resetZoom(500));
function savePDF() {{
  const svgEl = document.getElementById("conceptflow-nested-svg");
  const w = parseInt(svgEl.getAttribute("width"));
  const h = parseInt(svgEl.getAttribute("height"));
  const svgStr = getStyledSVGString();
  const scale = 2;
  const canvas = document.createElement("canvas");
  canvas.width = w * scale;
  canvas.height = h * scale;
  const ctx = canvas.getContext("2d");
  ctx.scale(scale, scale);
  ctx.fillStyle = "white";
  ctx.fillRect(0, 0, w, h);
  const blob = new Blob([svgStr], {{type: "image/svg+xml;charset=utf-8"}});
  const url = URL.createObjectURL(blob);
  const img = new Image();
  img.onload = function() {{
    ctx.drawImage(img, 0, 0);
    URL.revokeObjectURL(url);
    const {{ jsPDF }} = window.jspdf;
    const pdf = new jsPDF({{
      orientation: w >= h ? "landscape" : "portrait",
      unit: "px",
      format: [w, h],
    }});
    pdf.addImage(canvas.toDataURL("image/png"), "PNG", 0, 0, w, h);
    const name = (currentView.name || "diagram").replace(/[^a-z0-9]/gi, "_").toLowerCase();
    pdf.save(name + ".pdf");
  }};
  img.src = url;
}}

d3.select("#conceptflow-save-svg").on("click", saveSVG);
d3.select("#conceptflow-save-png").on("click", savePNG);
d3.select("#conceptflow-save-pdf").on("click", savePDF);

renderView(rootData);
</script>
</body>
</html>
"""

def plot_nested(
    view: ExplorationView,
    mode: str = "navigator",
    show_child_previews: bool = True,
    width: int = 900,
    height: int = 700,
    title: str | None = None,
    object_label: Callable[[str], str] | None = None,
) -> HTMLFigure:
    """
    Plot a nested exploration view with subdirect decomposition.

    Parameters
    ----------
    view:
        Root ExplorationView to render.
    width, height:
        Figure size in pixels.
    show_child_previews:
        Whether to draw the inner template preview inside outer nodes.

    Returns
    -------
    HTMLFigure
        Notebook-displayable and exportable HTML figure.
    """
    if mode != "navigator":
        raise ValueError(
            f'Unknown nested visualization mode "{mode}". '
            'Currently only "navigator" is supported.'
        )
    data = exploration_view_to_nested_data(view, object_label=object_label)
    html = render_nested_html(
        data,
        width=width,
        height=height,
        title=title,
        show_child_previews=show_child_previews,
    )
    return HTMLFigure(html)
