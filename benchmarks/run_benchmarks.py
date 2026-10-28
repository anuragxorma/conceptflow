"""
Benchmark ConceptFlow against fcapy and concepts.

Each measurement runs in its own Python process with a time limit, on
contexts written to .cxt files, so every library reads the same data and file
reading is not timed. The result counts (concepts, cover edges, implications)
are compared across libraries, so a fast wrong answer would show up.

Setup (fcapy and concepts are not ConceptFlow dependencies):

    pip install conceptflow fcapy concepts

Usage:

    python benchmarks/run_benchmarks.py                 # all, 300 s limit
    python benchmarks/run_benchmarks.py --quick         # small contexts only
    python benchmarks/run_benchmarks.py --timeout 60 --output results.json
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import tempfile
import time
import warnings
from importlib import metadata
from pathlib import Path

# (name, how to build it). Random contexts use make_random_context with a
# fixed seed, so every run uses the same data.
DATASETS = [
    ("livingbeings_en (8×9)", ("bundled", "livingbeings_en")),
    ("seasoningplanner_de (56×37)", ("bundled", "seasoningplanner_de")),
    ("random 100×15, 30%", ("random", 100, 15, 0.3)),
    ("random 200×50, 15%", ("random", 200, 50, 0.15)),
    ("random 500×25, 30%", ("random", 500, 25, 0.3)),
    ("random 100×20, 60%", ("random", 100, 20, 0.6)),
    ("random 2000×30, 20%", ("random", 2000, 30, 0.2)),
    ("Mushroom (8124×119)", ("mushroom",)),
]
QUICK = 4  # --quick: the first four data sets

TASKS = {
    "concepts": "all formal concepts",
    "lattice": "concepts and the cover relation (Hasse diagram)",
    "stability": "exact stability of every concept (lattice built untimed)",
    "stability_bounds": "lower and upper stability bounds (lattice built untimed)",
    "canonical_basis": "Duquenne-Guigues basis",
}

LIBRARIES = ("conceptflow", "fcapy", "concepts")


# ---------------------------------------------------------------------------
# Worker: one library, one task, one file (runs in a subprocess)
# ---------------------------------------------------------------------------


def _conceptflow(task, path):
    from conceptflow import ConceptLattice
    from conceptflow.algorithms import compute_canonical_basis, enumerate_concepts
    from conceptflow.io import read_cxt
    from conceptflow.metrics import concept_stability, stability_bounds

    context = read_cxt(path)
    if task == "concepts":
        return lambda: {"concepts": len(enumerate_concepts(context, method="closebyone"))}
    if task == "lattice":
        def run():
            lattice = ConceptLattice.from_context(context, algorithm="closebyone")
            return {"concepts": lattice.n_concepts, "edges": len(lattice.edges)}
        return run
    if task == "stability":
        lattice = ConceptLattice.from_context(context, algorithm="closebyone")
        return lambda: {"concepts": len(concept_stability(lattice))}
    if task == "stability_bounds":
        lattice = ConceptLattice.from_context(context, algorithm="closebyone")
        return lambda: {"concepts": len(stability_bounds(lattice)[0])}
    if task == "canonical_basis":
        return lambda: {"implications": len(compute_canonical_basis(context))}
    return None


def _fcapy(task, path):
    import fcapy.algorithms.concept_construction as construction
    from fcapy.context import FormalContext
    from fcapy.lattice import ConceptLattice

    context = FormalContext.read_cxt(path)
    if task == "concepts":
        return lambda: {"concepts": sum(1 for _ in construction.close_by_one(context))}
    if task == "lattice":
        def run():
            lattice = ConceptLattice.from_context(context)  # default: Lindig
            edges = sum(len(lattice.children(i)) for i in range(len(lattice)))
            return {"concepts": len(lattice), "edges": edges}
        return run
    if task in ("stability", "stability_bounds"):
        lattice = ConceptLattice.from_context(context)

        def run():
            lattice.calc_concepts_measures(task, context)
            return {"concepts": len(lattice)}
        return run
    return None  # no canonical basis


def _concepts(task, path):
    import concepts

    if task != "lattice":
        return None  # computes the lattice only

    def run():
        # A fresh Context each time: the lattice is cached on the context.
        context = concepts.Context.fromfile(path, frmat="cxt")
        lattice = context.lattice
        return {
            "concepts": len(lattice),
            "edges": sum(len(concept.upper_neighbors) for concept in lattice),
        }
    return run


def _worker(library, task, path, repeats):
    warnings.simplefilter("ignore")
    run = {"conceptflow": _conceptflow, "fcapy": _fcapy, "concepts": _concepts}[library](
        task, path
    )
    if run is None:
        print(json.dumps({"status": "unsupported"}))
        return
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        counts = run()
        times.append(time.perf_counter() - start)
        if times[-1] > 5:  # slow enough that one run is representative
            break
    print(json.dumps({"status": "ok", "seconds": min(times), "counts": counts}))


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _write_datasets(folder: Path, datasets) -> dict[str, Path]:
    from conceptflow.datasets import load_context, load_mushroom, make_random_context
    from conceptflow.io import write_cxt

    paths = {}
    for index, (name, spec) in enumerate(datasets):
        if spec[0] == "bundled":
            context = load_context(spec[1])
        elif spec[0] == "random":
            context = make_random_context(*spec[1:], random_state=index)
        else:
            context = load_mushroom()
        path = folder / f"dataset{index}.cxt"
        write_cxt(context, path)
        paths[name] = path
    return paths


def _measure(library, task, path, timeout, repeats):
    command = [sys.executable, __file__, "--worker", library, task, str(path), str(repeats)]
    try:
        done = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"status": "timeout"}
    if done.returncode != 0:
        lines = done.stderr.strip().splitlines()
        return {"status": "error", "error": lines[-1] if lines else "failed"}
    return json.loads(done.stdout.strip().splitlines()[-1])


def _cell(result, timeout):
    if result["status"] == "ok":
        seconds = result["seconds"]
        if seconds < 0.001:
            return "< 0.001 s"
        return f"{seconds:.3f} s" if seconds < 1 else f"{seconds:.1f} s"
    return {
        "unsupported": "—",
        "timeout": f"> {timeout:g} s",
        "error": "error",
    }[result["status"]]


def _versions():
    versions = {"python": platform.python_version()}
    for package in ("conceptflow", "fcapy", "concepts", "numpy", "scikit-learn"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "not installed"
    return versions


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--timeout", type=float, default=300, help="seconds per measurement")
    parser.add_argument(
        "--repeats", type=int, default=3, help="runs per measurement (fastest kept)"
    )
    parser.add_argument("--quick", action="store_true", help="only the small data sets")
    parser.add_argument("--output", type=Path, help="also write the results as JSON")
    args = parser.parse_args(argv)

    datasets = DATASETS[:QUICK] if args.quick else DATASETS
    results = []
    with tempfile.TemporaryDirectory() as folder:
        paths = _write_datasets(Path(folder), datasets)
        for name, path in paths.items():
            for task in TASKS:
                row = {"dataset": name, "task": task}
                for library in LIBRARIES:
                    row[library] = _measure(library, task, path, args.timeout, args.repeats)
                results.append(row)
                print(
                    f"{name:30} {task:16} "
                    + "  ".join(f"{lib}: {_cell(row[lib], args.timeout):>10}" for lib in LIBRARIES),
                    flush=True,
                )

    # The libraries must agree on every count they report.
    disagreements = []
    for row in results:
        counts = [row[lib]["counts"] for lib in LIBRARIES if row[lib]["status"] == "ok"]
        for key in {key for c in counts for key in c}:
            values = {c[key] for c in counts if key in c}
            if len(values) > 1:
                disagreements.append((row["dataset"], row["task"], key, sorted(values)))
    if disagreements:
        print(f"\nDISAGREEMENTS: {disagreements}")
    else:
        print("\nResults agree across libraries.")

    print("\n| Data set | Task | " + " | ".join(LIBRARIES) + " |")
    print("|---|---|" + "---|" * len(LIBRARIES))
    for row in results:
        print(
            f"| {row['dataset']} | {row['task']} | "
            + " | ".join(_cell(row[lib], args.timeout) for lib in LIBRARIES)
            + " |"
        )

    if args.output:
        args.output.write_text(
            json.dumps(
                {"versions": _versions(), "machine": platform.processor() or platform.machine(),
                 "timeout": args.timeout, "tasks": TASKS, "results": results},
                indent=1,
            )
        )
    return 1 if disagreements else 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        _, _, library, task, path, repeats = sys.argv
        _worker(library, task, path, int(repeats))
    else:
        sys.exit(main())
