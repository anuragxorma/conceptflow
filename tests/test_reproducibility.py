"""
The same input gives the same output, in every Python process.

Python orders sets of strings differently from one process to the next
(hash randomization, ``PYTHONHASHSEED``), so code that iterates over a set
can produce node ids, label lists, rule orders or file contents in another
order each run. ``outputs()`` computes many results; the test runs it in
processes with different hash seeds and compares every result.

DimFlux stops its search after a time limit, so its coordinates may differ
between runs by design; everything else about its drawings is compared.
"""

import hashlib
import io
import json
import os
import re
import subprocess
import sys

import numpy as np
import pandas as pd

SEEDS = ["0", "1", "4242"]


def _digest(value) -> str:
    if isinstance(value, pd.DataFrame):
        text = value.to_json(orient="split", default_handler=str)
    elif isinstance(value, np.ndarray):
        text = json.dumps(value.tolist(), default=str)
    elif isinstance(value, str):
        text = value
    else:
        text = repr(value)
    return hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()


def _without_coordinates(value):
    if isinstance(value, dict):
        return {k: _without_coordinates(v) for k, v in value.items() if k not in {"x", "y"}}
    if isinstance(value, list):
        return [_without_coordinates(v) for v in value]
    return value


def _graph_data_text(graph_data, coordinates=True):
    nodes = [
        (n.node_id, n.label, n.hover_text, n.concept.signature(), repr(n.metadata),
         (n.x, n.y) if coordinates else None)
        for n in graph_data.nodes
    ]
    edges = [(e.source, e.target, repr(e.metadata)) for e in graph_data.edges]
    return repr((nodes, edges, repr(graph_data.metadata)))


def _many_valued_data():
    rng = np.random.default_rng(7)
    n = 14
    return pd.DataFrame(
        {
            "color": rng.choice(["red", "blue", "green", "teal"], n),
            "size": rng.uniform(0, 100, n).round(),
            "shape": rng.choice(["round", "square", "oval"], n),
            "grade": rng.choice(["low", "mid", "high"], n),
        },
        index=[f"o{i}" for i in range(n)],
    )


def outputs() -> dict[str, str]:
    """Many results of ConceptFlow, each as a digest of its exact text."""
    import tempfile
    import warnings

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from conceptflow import ExplorationBuilder, ManyValuedContext, plot_lattice
    from conceptflow.algorithms import (
        clarify_context,
        compute_canonical_basis,
        compute_luxenburger_basis,
        compute_minimal_generators,
        compute_proper_premise_basis,
        reduce_context,
    )
    from conceptflow.cluster import ConceptLatticeEstimator
    from conceptflow.decomposition import ASSO, GreConD
    from conceptflow.exploration import nested_combinations, nested_view
    from conceptflow.io import context_to_json, write_csv, write_cxt
    from conceptflow.metrics import (
        concept_levels,
        concept_stability,
        delta_measure,
        drawing_quality_scores,
        stability_bounds,
    )
    from conceptflow.preprocessing import (
        ConceptualScaler,
        ContranominalScale,
        InterordinalScale,
        NominalScale,
        OrdinalScale,
        ThresholdScale,
    )
    from conceptflow.rules import ImplicationBasisEstimator, ImplicationOutlierDetector
    from conceptflow.visualization import plot_nested
    from conceptflow.visualization.nested import exploration_view_to_nested_data
    from tests.test_feature_matrix import INPUT_NAMES, make_lattice, split_in_two

    matplotlib.rcParams["svg.hashsalt"] = "conceptflow"
    found: dict[str, str] = {}

    def add(name, value):
        found[name] = _digest(value)

    with warnings.catch_warnings(), tempfile.TemporaryDirectory() as tmp:
        warnings.simplefilter("ignore")
        for input_name in INPUT_NAMES:
            lattice = make_lattice(input_name)
            context = lattice.context
            key = input_name.replace(" ", "_")
            add(f"{key}/concepts", [c.signature() for c in lattice.concepts])
            add(f"{key}/edges", [(a.signature(), b.signature()) for a, b in lattice.edges])

            layered = plot_lattice(lattice, backend="graph_data", layout="layered")
            add(f"{key}/layered", _graph_data_text(layered))
            dimflux = plot_lattice(lattice, backend="graph_data", layout="dimflux",
                                   timeout_ms=50)
            add(f"{key}/dimflux", _graph_data_text(dimflux, coordinates=False))
            add(f"{key}/d3", plot_lattice(lattice, backend="d3", layout="layered").html)
            add(f"{key}/graphviz", plot_lattice(lattice, backend="graphviz").source)
            add(f"{key}/tikz", plot_lattice(lattice, backend="tikz", layout="layered").source)
            figure = plot_lattice(lattice, backend="matplotlib", layout="layered")
            buffer = io.StringIO()
            figure.savefig(buffer, format="svg", metadata={"Date": None})
            plt.close(figure)
            add(f"{key}/matplotlib", buffer.getvalue())
            add(f"{key}/scores", drawing_quality_scores(layered))

            add(f"{key}/canonical", compute_canonical_basis(context))
            add(f"{key}/proper", compute_proper_premise_basis(context))
            add(f"{key}/luxenburger", compute_luxenburger_basis(context))
            add(f"{key}/generators", list(compute_minimal_generators(context).items()))
            for measure in (concept_stability, delta_measure, concept_levels):
                add(f"{key}/{measure.__name__}", measure(lattice))
            add(f"{key}/bounds", stability_bounds(lattice))
            clarified = clarify_context(context)
            add(f"{key}/clarify", (clarified.context, list(clarified.object_groups.items()),
                                   list(clarified.attribute_groups.items())))
            reduced = reduce_context(context)
            add(f"{key}/reduce", (reduced.context, reduced.removed_objects,
                                  reduced.removed_attributes))

            add(f"{key}/json", context_to_json(context))
            write_csv(context, f"{tmp}/{key}.csv")
            add(f"{key}/csv", open(f"{tmp}/{key}.csv", encoding="utf-8").read())
            if not any("\n" in n for n in context.objects + context.attributes):
                write_cxt(context, f"{tmp}/{key}.cxt")
                add(f"{key}/cxt", open(f"{tmp}/{key}.cxt", encoding="utf-8").read())

            outer, inner = split_in_two(context)
            view = nested_view(outer, inner)
            add(f"{key}/nested_combinations", nested_combinations(view))
            nested = exploration_view_to_nested_data(view, timeout_ms=50)
            add(f"{key}/nested_data", _without_coordinates(nested))

            if context.n_objects and context.n_attributes:
                X = context.to_dataframe()
                for name, estimator in (
                    ("lattice", ConceptLatticeEstimator(output="dataframe")),
                    ("stable", ConceptLatticeEstimator(select="stability", n_concepts=4)),
                    ("implications", ImplicationBasisEstimator(output="dataframe")),
                    ("grecond", GreConD(output="dataframe")),
                    ("asso", ASSO(output="dataframe")),
                ):
                    result = estimator.fit_transform(X)
                    add(f"{key}/{name}", (list(estimator.get_feature_names_out()),
                                          np.asarray(result).tolist()))
                detector = ImplicationOutlierDetector(contamination="auto").fit(X)
                add(f"{key}/outliers", (detector.rules_, detector.score_samples(X).tolist(),
                                        detector.explain(X)))

        # Many-valued data, scaling and nested exploration over three levels.
        data = _many_valued_data()
        scalers = {
            "nominal": [NominalScale("color")],
            "contranominal": [ContranominalScale("shape")],
            "ordinal": [OrdinalScale("grade", ["low", "mid", "high"])],
            "interordinal": [InterordinalScale("grade", ["low", "mid", "high"])],
            "threshold": [ThresholdScale("size", [33, 66])],
        }
        for name, scales in scalers.items():
            scaler = ConceptualScaler(scales=scales, output="dataframe")
            add(f"scaling/{name}", (scaler.fit_transform(data),
                                   list(scaler.get_feature_names_out())))
        mvc = ManyValuedContext.from_dataframe(data)
        add("scaling/unique_values", [sorted(map(str, mvc.unique_values(a)))
                                      for a in mvc.attributes])
        builder = ExplorationBuilder(mvc)
        root = builder.root("color", [NominalScale("color")])
        for child in builder.expand_all(root, scales=[ThresholdScale("size", [33, 66])]):
            builder.expand_all(child, scales=[ContranominalScale("shape")])
        add("nested/three_levels",
            _without_coordinates(exploration_view_to_nested_data(root, timeout_ms=50)))
        add("nested/combinations", nested_combinations(root))
        html = plot_nested(root, timeout_ms=50).html
        # The page apart from its data, whose coordinates come from DimFlux.
        add("nested/page_without_data", re.sub(r"const rootData = .*?;\n", "", html))
    return found


def _start(seed: str) -> subprocess.Popen:
    environment = dict(os.environ, PYTHONHASHSEED=seed,
                       PYTHONPATH=os.pathsep.join(p for p in sys.path if p))
    code = (
        "import json; from tests.test_reproducibility import outputs; "
        "print(json.dumps(outputs()))"
    )
    return subprocess.Popen(
        [sys.executable, "-c", code], env=environment, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )


def test_outputs_do_not_depend_on_the_hash_seed():
    processes = {seed: _start(seed) for seed in SEEDS}  # in parallel
    runs = {}
    for seed, process in processes.items():
        stdout, stderr = process.communicate(timeout=1200)
        assert process.returncode == 0, stderr[-3000:]
        runs[seed] = json.loads(stdout.strip().splitlines()[-1])
    first = runs[SEEDS[0]]
    assert len(first) > 200
    differ = sorted(
        name for name in first
        if any(run.get(name) != first[name] for run in runs.values())
    )
    assert not differ, f"{len(differ)} outputs change with PYTHONHASHSEED: {differ}"
