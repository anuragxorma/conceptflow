"""
Eurovision nested line diagram: voting cluster support × musical attributes.

Objects: each year's Eurovision winner (1975–2025, excluding 2020).

Level 1 (outer lattice): voting-cluster support — four boolean attributes.
    For each cluster type, True if the winner's own cluster gave them an
    average of ≥ 8 points; False otherwise.
    - regional  (DichotomicScale) — e.g. Scandinavia voted strongly for SE
    - cultural  (DichotomicScale) — e.g. Nordic bloc voted strongly for SE
    - historical (DichotomicScale)
    - political  (DichotomicScale)

    Four DichotomicScales → 8 binary attributes total, well within DimFlux.

Level 2 (inner lattice per outer node): musical attributes combined.
    - OrdinalScale: bpm_cat slow (<100) ≤ medium (100–124) ≤ fast (≥125)
    - DichotomicScale: key_mode minor vs. major

Run from the project root:

    python -m examples.eurovision_nested_diagram

Output:

    eurovision_nested.html
"""

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from conceptflow import ExplorationBuilder, ManyValuedContext
from conceptflow.preprocessing import DichotomicScale, GeneralScale, ThresholdScale
from conceptflow.visualization import debug_bottom_outer, plot_nested
from conceptflow.visualization.dimflux_layout import lattice_to_graph_data_dimflux
from conceptflow.visualization.nested import _compute_filled_pairs, _mu_local


DATASET_ROOT = Path("examples/eurovision-dataset/data/senior")

CLUSTER_SUPPORT_THRESHOLD = 8  # average points from cluster members


REGIONAL_CLUSTERS = {
    "British Isles": ["GB", "GB-WLS", "IE"],
    "Scandinavia": ["DK", "FI", "IS", "NO", "SE"],
    "Baltic States": ["EE", "LV", "LT"],
    "Benelux": ["BE", "NL", "LU"],
    "Iberian Peninsula": ["AD", "ES", "FR", "PT"],
    "Central Europe": ["AT", "CH", "CZ", "DE", "HU", "PL", "SK"],
    "Mediterranean": ["IT", "MC", "MT", "SM"],
    "Balkans": ["AL", "BA", "CS", "CY", "GR", "HR", "ME", "MK", "RS", "SI", "TR", "YU"],
    "Eastern Europe": ["BY", "BG", "MD", "RO", "RU", "UA"],
    "Caucasus": ["AM", "AZ", "GE"],
    "Non-European": ["AU", "IL", "KZ", "MA"],
}

CULTURAL_CLUSTERS = {
    "Anglophone": ["AU", "GB", "GB-WLS", "IE"],
    "Nordic": ["DK", "FI", "IS", "NO", "SE"],
    "Baltic": ["EE", "LV", "LT"],
    "Germanic": ["AT", "BE", "CH", "DE", "LU", "NL"],
    "Romance": ["AD", "FR", "IT", "MC", "MD", "MT", "PT", "RO", "SM", "ES"],
    "East-Central Europe": ["CZ", "HU", "PL", "SK", "SI", "HR"],
    "East Slavic": ["BY", "RU", "UA"],
    "Balkan": ["AL", "BA", "BG", "CS", "ME", "MK", "RS", "YU"],
    "Hellenic": ["CY", "GR"],
    "Turkic": ["AZ", "KZ", "TR"],
    "Caucasian": ["AM", "GE"],
    "Semitic": ["IL", "MA"],
}

HISTORICAL_CLUSTERS = {
    "Former Soviet Union": ["AM", "AZ", "BY", "EE", "GE", "KZ", "LV", "LT", "MD", "RU", "UA"],
    "Former Yugoslavia": ["BA", "CS", "HR", "ME", "MK", "RS", "SI", "YU"],
    "Former Eastern Bloc": ["AL", "BG", "CZ", "HU", "PL", "RO", "SK"],
    "Western Bloc": ["AU", "BE", "DE", "DK", "ES", "FR", "GB", "GB-WLS", "GR", "IL", "IS", "IT", "LU", "NL", "NO", "PT", "TR"],
    "Neutral": ["AD", "AT", "CH", "FI", "IE", "MC", "SE", "SM"],
    "Non-Aligned Movement": ["CY", "MA", "MT"],
}

POLITICAL_CLUSTERS = {
    "EU Eurozone": ["AT", "BE", "BG", "HR", "CY", "EE", "FI", "FR", "DE", "GR", "IE", "IT", "LV", "LT", "LU", "MT", "NL", "PT", "SK", "SI", "ES"],
    "EU Non-Eurozone": ["CZ", "DK", "HU", "PL", "RO", "SE"],
    "EU Candidates": ["AL", "BA", "GE", "MD", "ME", "MK", "RS", "TR", "UA"],
    "EFTA / EEA": ["CH", "IS", "NO"],
    "Post-Brexit": ["GB", "GB-WLS"],
    "Eurasian Economic Union": ["AM", "BY", "KZ", "RU"],
    "Non-aligned": ["AD", "AZ", "MC", "SM"],
    "Non-European": ["AU", "IL", "MA"],
    "Defunct States": ["CS", "YU"],
}


def load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def cluster_of(country: str, clusters: dict) -> str | None:
    for name, members in clusters.items():
        if country in members:
            return name
    return None


def got_cluster_support(
    cluster_dict: dict,
    cluster_name: str | None,
    winner_country: str,
    eligible_voters: set,
    votes: dict,
) -> bool:
    if cluster_name is None:
        return False
    cluster_members = cluster_dict[cluster_name]
    eligible_members = [
        m for m in cluster_members
        if m != winner_country and m in eligible_voters
    ]
    if not eligible_members:
        return False
    avg_points = sum(votes.get(m, 0) for m in eligible_members) / len(eligible_members)
    return avg_points >= CLUSTER_SUPPORT_THRESHOLD


def key_mode(tone: str) -> str:
    return "minor" if tone and "minor" in tone.lower() else "major"


def bpm_category(bpm: int) -> str:
    if bpm >= 125:
        return "fast"
    if bpm >= 100:
        return "medium"
    return "slow"


def load_year_record(year: int) -> dict | None:
    year_dir = DATASET_ROOT / str(year)
    final_path = year_dir / "rounds" / "final.json"

    if not final_path.exists():
        return None

    final_round = load_json(final_path)
    performance = final_round["performances"][0]  # winner is first

    # Load contestant details
    contestants = {}
    for path in (year_dir / "contestants").glob("*/contestant.json"):
        c = load_json(path)
        contestants[c["id"]] = c

    winner = contestants.get(performance["contestantId"])
    if winner is None:
        return None

    # Extract votes: post-2016 uses whichever of jury/public is larger
    if year >= 2016:
        jury = next((s for s in performance["scores"] if s.get("name") == "jury"), None)
        tele = next((s for s in performance["scores"] if s.get("name") == "public"), None)
        jury_votes = jury["votes"] if jury else {}
        tele_votes = tele["votes"] if tele else {}
        votes = jury_votes if sum(jury_votes.values()) >= sum(tele_votes.values()) else tele_votes
    else:
        total = next((s for s in performance["scores"] if s.get("name") == "total"), None)
        if total is None:
            return None
        votes = total["votes"]

    eligible_voters = set(votes.keys())
    country = winner["country"]
    bpm = winner.get("bpm")
    tone = winner.get("tone") or ""

    if bpm is None:
        return None

    regional_name = cluster_of(country, REGIONAL_CLUSTERS)
    cultural_name = cluster_of(country, CULTURAL_CLUSTERS)
    historical_name = cluster_of(country, HISTORICAL_CLUSTERS)
    political_name = cluster_of(country, POLITICAL_CLUSTERS)

    return {
        "country": country,
        "regional": got_cluster_support(REGIONAL_CLUSTERS, regional_name, country, eligible_voters, votes),
        "cultural": got_cluster_support(CULTURAL_CLUSTERS, cultural_name, country, eligible_voters, votes),
        "historical": got_cluster_support(HISTORICAL_CLUSTERS, historical_name, country, eligible_voters, votes),
        "political": got_cluster_support(POLITICAL_CLUSTERS, political_name, country, eligible_voters, votes),
        "bpm": bpm,
        "key_mode": key_mode(tone),
    }


def build_dataframe() -> pd.DataFrame:
    records = {}
    for year in range(1975, 2026):
        if year == 2020:
            continue
        try:
            record = load_year_record(year)
        except Exception as exc:
            print(f"  {year}: skipped ({exc})")
            continue
        if record is None:
            print(f"  {year}: missing data")
            continue
        records[str(year)] = record

    df = pd.DataFrame.from_dict(records, orient="index")
    df.index.name = "year"
    return df


def _gamma(lattice) -> dict[int, str]:
    """
    Attribute concept map: γ(m) = outer concept with the LARGEST extent
    that has attribute index m in its intent.

    In the Hasse diagram this is the highest concept introducing m.
    Attribute labels are placed ABOVE this node.
    """
    result = {}
    for attr_idx in range(lattice.context.n_attributes):
        candidates = [c for c in lattice.concepts if attr_idx in c.intent]
        if candidates:
            gamma = max(candidates, key=lambda c: len(c.extent))
            result[attr_idx] = gamma.stable_id()
    return result


def _mu(lattice) -> dict[int, str]:
    """
    Object concept map: μ(g) = outer concept with the SMALLEST non-empty
    extent that still contains object index g.

    In the Hasse diagram this is the lowest concept introducing g.
    Object labels are placed BELOW this node.
    """
    result = {}
    for obj_idx in range(lattice.context.n_objects):
        candidates = [c for c in lattice.concepts if obj_idx in c.extent]
        if candidates:
            mu = min(candidates, key=lambda c: len(c.extent))
            result[obj_idx] = mu.stable_id()
    return result


def dump_subdirect_d3_json(root, out_path) -> None:
    """
    Export the outer lattice + inner template + subdirect decomposition data
    as a single JSON file ready for D3.

    Outer nodes carry:
    - x, y        — DimFlux coordinates
    - size         — |extent| (use for circle radius)
    - new_attrs    — attributes introduced HERE (γ(m) = this node); label ABOVE
    - new_objects  — objects introduced HERE (μ(g) = this node); label BELOW
    - filled_inner — inner template node IDs whose extent ∩ new_objects ≠ ∅

    Inner template nodes (from the top outer concept, full 50-object inner
    lattice) carry the same fields, plus new_attrs/new_objects for labelling
    the template diagram.
    """
    outer_lattice = root.lattice
    outer_context = root.context

    # Outer DimFlux layout.
    outer_gd = lattice_to_graph_data_dimflux(outer_lattice, stable_ids=True)

    # γ and μ for the outer lattice.
    gamma_outer = _gamma(outer_lattice)
    mu_outer = _mu(outer_lattice)

    by_concept_attrs: dict[str, list[str]] = {c.stable_id(): [] for c in outer_lattice.concepts}
    by_concept_objs: dict[str, list[str]] = {c.stable_id(): [] for c in outer_lattice.concepts}

    for attr_idx, cid in gamma_outer.items():
        by_concept_attrs[cid].append(outer_context.attributes[attr_idx])
    for obj_idx, cid in mu_outer.items():
        by_concept_objs[cid].append(outer_context.objects[obj_idx])

    # Inner template: child view of the top outer concept (full 50-object extent).
    top_concept = outer_lattice.top()
    top_id = top_concept.stable_id()
    template_view = root.children[top_id]

    inner_lattice = template_view.lattice
    inner_context = template_view.context
    inner_gd = lattice_to_graph_data_dimflux(inner_lattice, stable_ids=True)

    # γ and μ for the inner template.
    gamma_inner = _gamma(inner_lattice)
    mu_inner = _mu(inner_lattice)

    inner_by_attrs: dict[str, list[str]] = {c.stable_id(): [] for c in inner_lattice.concepts}
    inner_by_objs: dict[str, list[str]] = {c.stable_id(): [] for c in inner_lattice.concepts}

    for attr_idx, cid in gamma_inner.items():
        inner_by_attrs[cid].append(inner_context.attributes[attr_idx])
    for obj_idx, cid in mu_inner.items():
        inner_by_objs[cid].append(inner_context.objects[obj_idx])

    # Subdirect decomposition: for each outer concept, find which inner
    # template nodes are filled by its newly introduced objects.
    inner_node_extent: dict[str, set[str]] = {
        node.node_id: set(node.metadata["extent"])
        for node in inner_gd.nodes
    }

    filled_inner: dict[str, list[str]] = {}
    for concept in outer_lattice.concepts:
        cid = concept.stable_id()
        new_objs = set(by_concept_objs[cid])
        filled_inner[cid] = [
            iid for iid, iext in inner_node_extent.items()
            if iext & new_objs
        ]

    # Assemble JSON.
    d3 = {
        "outer": {
            "nodes": [
                {
                    "id": node.node_id,
                    "label": node.label,
                    "x": node.x,
                    "y": node.y,
                    "size": len(node.metadata["extent"]),
                    "extent": sorted(node.metadata["extent"]),
                    "intent": sorted(node.metadata["intent"]),
                    "new_attrs": sorted(by_concept_attrs[node.node_id]),
                    "new_objects": sorted(by_concept_objs[node.node_id]),
                    "filled_inner": filled_inner[node.node_id],
                }
                for node in outer_gd.nodes
            ],
            "links": [
                {"source": e.source, "target": e.target}
                for e in outer_gd.edges
            ],
        },
        "inner_template": {
            "nodes": [
                {
                    "id": node.node_id,
                    "label": node.label,
                    "x": node.x,
                    "y": node.y,
                    "size": len(node.metadata["extent"]),
                    "extent": sorted(node.metadata["extent"]),
                    "intent": sorted(node.metadata["intent"]),
                    "new_attrs": sorted(inner_by_attrs[node.node_id]),
                    "new_objects": sorted(inner_by_objs[node.node_id]),
                }
                for node in inner_gd.nodes
            ],
            "links": [
                {"source": e.source, "target": e.target}
                for e in inner_gd.edges
            ],
        },
    }

    out = Path(out_path)
    out.write_text(json.dumps(d3, indent=2), encoding="utf-8")
    print(
        f"Wrote subdirect D3 JSON: {out}  "
        f"({len(d3['outer']['nodes'])} outer nodes, "
        f"{len(d3['inner_template']['nodes'])} inner template nodes)"
    )


def compute_implication_basis(root) -> list[dict]:
    """
    Compute the Duquenne-Guigues (stem) basis for the combined formal context
    K = K_outer | K_inner built from the ExplorationView.

    Returns a list of dicts, one per implication, each with:
      premise       — attribute names in the premise
      conclusion    — attribute names in the conclusion (extra attributes forced)
      objects       — object names whose attributes satisfy the premise
      premise_outer / premise_inner — premise split by scale level
      concl_outer  / concl_inner   — conclusion split by scale level
      unfilled_pairs — count of (outer, inner) diagram pairs this implication explains
    """
    outer_ctx = root.context
    top_id = root.lattice.top().stable_id()
    inner_ctx = root.children[top_id].context

    outer_names = list(outer_ctx.attributes)
    inner_names = list(inner_ctx.attributes)
    all_names   = outer_names + inner_names
    n_outer     = len(outer_names)
    n_all       = len(all_names)
    n_obj       = outer_ctx.n_objects
    obj_names   = list(outer_ctx.objects)

    # Combined incidence matrix: rows = objects, cols = outer attrs then inner attrs.
    inc = np.zeros((n_obj, n_all), dtype=bool)
    inc[:, :n_outer] = outer_ctx.incidence
    inc[:, n_outer:] = inner_ctx.incidence

    def ext(A: frozenset) -> frozenset:
        if not A:
            return frozenset(range(n_obj))
        mask = np.ones(n_obj, dtype=bool)
        for a in A:
            mask &= inc[:, a]
        return frozenset(np.where(mask)[0])

    def clo(A: frozenset) -> frozenset:
        S = ext(A)
        if not S:
            return frozenset(range(n_all))
        mask = np.ones(n_all, dtype=bool)
        for g in S:
            mask &= inc[g, :]
        return frozenset(np.where(mask)[0])

    # All 2^n_all subsets — fine for n_all ≤ ~16.
    all_sets = [frozenset(c)
                for r in range(n_all + 1)
                for c in combinations(range(n_all), r)]
    closures = {A: clo(A) for A in all_sets}

    # Pseudo-intents → Duquenne-Guigues basis.
    # P is a pseudo-intent iff clo(P) ≠ P and for every pseudo-intent Q ⊊ P: clo(Q) ⊆ P.
    pseudo_intents: dict[frozenset, frozenset] = {}
    for P in sorted(all_sets, key=lambda s: (len(s), sorted(s))):
        if closures[P] == P:
            continue
        if all(Qcl <= P for Q, Qcl in pseudo_intents.items() if Q < P):
            pseudo_intents[P] = closures[P]

    # Pre-compute filled pairs for the diagram-impact count.
    mu_o = _mu_local(root.lattice)
    mu_i = _mu_local(root.children[top_id].lattice)
    filled_by_outer = _compute_filled_pairs(root.lattice, root.children[top_id].lattice, mu_o, mu_i)

    oc_list = list(root.lattice.concepts)
    ic_list = list(root.children[top_id].lattice.concepts)
    oi = {c.stable_id(): c.intent for c in oc_list}
    ii = {c.stable_id(): c.intent for c in ic_list}
    bot_o = min(oc_list, key=lambda c: len(c.extent)).stable_id()
    bot_i = min(ic_list, key=lambda c: len(c.extent)).stable_id()

    results = []
    for P, Pcl in sorted(pseudo_intents.items(), key=lambda x: (len(x[0]), sorted(x[0]))):
        prem_names = [all_names[i] for i in sorted(P)]
        conc_names = [all_names[i] for i in sorted(Pcl - P)]
        objs       = sorted(obj_names[g] for g in ext(P))

        prem_outer = [all_names[i] for i in sorted(P)       if i < n_outer]
        prem_inner = [all_names[i] for i in sorted(P)       if i >= n_outer]
        conc_outer = [all_names[i] for i in sorted(Pcl - P) if i < n_outer]
        conc_inner = [all_names[i] for i in sorted(Pcl - P) if i >= n_outer]

        # Count unfilled pairs for which this implication fires.
        n_unfilled = 0
        for oc in oc_list:
            if oc.stable_id() == bot_o:
                continue
            filled = filled_by_outer.get(oc.stable_id(), set())
            for ic in ic_list:
                if ic.stable_id() == bot_i or ic.stable_id() in filled:
                    continue
                combined = frozenset(oi[oc.stable_id()]) | frozenset(a + n_outer for a in ii[ic.stable_id()])
                if P <= combined and not (Pcl <= combined):
                    n_unfilled += 1

        results.append(dict(
            premise=prem_names, conclusion=conc_names, objects=objs,
            premise_outer=prem_outer, premise_inner=prem_inner,
            concl_outer=conc_outer, concl_inner=conc_inner,
            unfilled_pairs=n_unfilled,
        ))
    return results


def print_implication_report(basis: list[dict]) -> None:
    sep = "=" * 68
    print(f"\n{sep}")
    print(f"IMPLICATION BASIS (Duquenne-Guigues)  —  {len(basis)} implications")
    print(sep)

    for k, imp in enumerate(basis):
        po, pi = imp["premise_outer"], imp["premise_inner"]
        co, ci = imp["concl_outer"],   imp["concl_inner"]
        if not imp["objects"]:
            itype = "impossible premise (vacuous)"
        else:
            prem_side = ("outer + inner" if po and pi else
                         "outer" if po else "inner")
            conc_side = ("outer + inner" if co and ci else
                         "outer" if co else "inner")
            itype = f"{prem_side} → {conc_side}"

        prem = ", ".join(imp["premise"])   or "∅"
        conc = ", ".join(imp["conclusion"]) or "∅"
        objs = ", ".join(imp["objects"])

        print(f"\n[{k+1}]  {{{prem}}}  →  {{{conc}}}")
        print(f"     Type   : {itype}")
        print(f"     Objects ({len(imp['objects'])}): {objs if objs else '(none — vacuous)'}")
        print(f"     Diagram: causes {imp['unfilled_pairs']} unfilled (outer, inner) pairs")


def main() -> None:
    print("Loading Eurovision winner data + voting records...")
    df = build_dataframe()
    print(f"\nLoaded {len(df)} winners.")

    print("\nVoting cluster support:")
    for col in ["regional", "cultural", "historical", "political"]:
        n_true = df[col].sum()
        print(f"  {col}: {n_true}/{len(df)} winners got cluster support")

    print("\nMusic distribution:")
    print(f"  bpm (slow <100 / medium 100–149 / fast >=150):")
    print(f"    slow={(df['bpm'] < 100).sum()}  medium={((df['bpm'] >= 100) & (df['bpm'] < 150)).sum()}  fast={(df['bpm'] >= 150).sum()}")
    print(f"  key_mode: {df['key_mode'].value_counts().to_dict()}")

    # Build year → "CC YEAR" display map before dropping country from the MVC.
    year_to_display = {year: f"{cc} {year}" for year, cc in df["country"].items()}
    mvc_df = df.drop(columns=["country"])

    mvc = ManyValuedContext.from_dataframe(mvc_df)
    builder = ExplorationBuilder(mvc, algorithm="nextclosure")

    # Outer lattice: which combination of cluster types supported the winner?
    # GeneralScale: True → {attr}, False → {} (standard FCA boolean encoding).
    # 4 binary attributes total → fast DimFlux, 16 outer concepts.
    # Concepts read as: "winners who got exactly these cluster types supporting them".
    root = builder.root(
        name="Eurovision winners: voting cluster support",
        scales=[
             GeneralScale("political",  mapping={True: {"political_support"},  False: frozenset()}, attribute_order=["political_support"]),
            GeneralScale("regional",   mapping={True: {"regional_support"},   False: frozenset()}, attribute_order=["regional_support"]),
            GeneralScale("cultural",   mapping={True: {"cultural_support"},   False: frozenset()}, attribute_order=["cultural_support"]),
            GeneralScale("historical", mapping={True: {"historical_support"}, False: frozenset()}, attribute_order=["historical_support"]),
           
        ],
    )

    # Inner lattice: BPM + key mode combined.
    tempo_scale = ThresholdScale("bpm", thresholds=[100, 150])
    mode_scale = DichotomicScale("key_mode", true_value="minor", false_value="major")

    builder.expand_all(
        parent=root,
        scales=[tempo_scale, mode_scale],
        name_template="music: {label}",
        min_extent_size=1,
        include_top=True,
        include_bottom=True,
    )

    debug_bottom_outer(root)

    basis = compute_implication_basis(root)
    print_implication_report(basis)

    dump_subdirect_d3_json(root, "eurovision_d3.json")

    fig = plot_nested(
        root,
        width=1400,
        height=900,
        title="Eurovision winners: voting cluster support → BPM × key mode",
        object_label=lambda y: year_to_display.get(y, y),
    )
    out_path = fig.open("eurovision_nested.html")
    print(f"\nWrote and opened: {out_path}")


if __name__ == "__main__":
    main()
