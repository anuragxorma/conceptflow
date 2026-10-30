"""
Smoke test of an installed ConceptFlow (a built wheel or sdist, or the
package from PyPI), not of the source tree.

    python -I smoke_test_installed.py EXPECTED_VERSION

Run it with ``-I`` from outside the repository: isolated mode ignores the
current directory and PYTHONPATH, so ``import conceptflow`` can only find the
installed package. The script checks that, too.

Each step uses a small example with a known answer and only what
``pip install conceptflow`` provides (no LaTeX or Graphviz).
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import conceptflow as cf


def check(label, condition):
    if not condition:
        sys.exit(f"FAILED: {label}")
    print(f"ok  {label}")


expected_version = sys.argv[1]
location = Path(cf.__file__).resolve()
check(f"version is {expected_version} (got {cf.__version__})", cf.__version__ == expected_version)
check(f"imported from an installation ({location.parent})", "site-packages" in location.parts)

# Context, lattice, canonical basis.
from conceptflow.algorithms import compute_canonical_basis  # noqa: E402

context = cf.FormalContext.from_array(
    [[1, 1, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1]], attributes=["a", "b", "c"]
)
lattice = cf.ConceptLattice.from_context(context)
check("lattice of the README context has 5 concepts", lattice.n_concepts == 5)
basis = {(i.premise_names(context), i.conclusion_names(context))
         for i in compute_canonical_basis(context)}
check("canonical basis is ab->c, ac->b, bc->a",
      basis == {(("a", "b"), ("c",)), (("a", "c"), ("b",)), (("b", "c"), ("a",))})

# Conceptual scaling.
from conceptflow.preprocessing import ConceptualScaler, NominalScale, ThresholdScale  # noqa: E402

data = pd.DataFrame({"color": ["red", "blue", "red"], "size": [1.0, 5.0, 9.0]},
                    index=["x", "y", "z"])
scaled = ConceptualScaler(
    scales=[NominalScale("color"), ThresholdScale("size", [4])], output="dataframe"
).fit_transform(data)
check("conceptual scaling", scaled.astype(int).values.tolist() == [[0, 1, 0], [1, 0, 1], [0, 1, 1]])

# Bundled data sets (package data).
from conceptflow.datasets import list_contexts, load_context  # noqa: E402

check("20 bundled FCA-repository contexts are listed", len(list_contexts()) == 20)
living = load_context("livingbeings_en")
check("bundled context loads (living beings: 8 x 9)",
      (living.n_objects, living.n_attributes) == (8, 9))
check("living beings lattice has 19 concepts",
      cf.ConceptLattice.from_context(living).n_concepts == 19)

# Factorization.
from conceptflow.decomposition import GreConD  # noqa: E402

model = GreConD().fit(np.asarray(living.incidence, dtype=int))
rebuilt = model.inverse_transform(model.transform(np.asarray(living.incidence, dtype=int)))
check("GreConD rebuilds the context exactly", np.array_equal(rebuilt, living.incidence))

# Visualization without external programs: interactive page, matplotlib,
# TikZ and DOT source, the DimFlux layout (a dependency).
living_lattice = cf.ConceptLattice.from_context(living)
html = cf.plot_lattice(living_lattice, layout="layered").html
check("interactive page has the pinned D3 and the data", "d3@7.9.0" in html and "graphData" in html)
graph = cf.plot_lattice(living_lattice, backend="graph_data", layout="dimflux")
check("DimFlux layout places every concept",
      all(node.x is not None for node in graph.nodes) and len(graph.nodes) == 19)
check("TikZ source", "\\begin{tikzpicture}" in
      cf.plot_lattice(living_lattice, backend="tikz", layout="layered").source)
check("DOT source", "digraph" in cf.plot_lattice(living_lattice, backend="graphviz").source)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
with tempfile.TemporaryDirectory() as folder:
    path = Path(folder) / "lattice.png"
    cf.plot_lattice(living_lattice, backend="matplotlib", layout="layered").savefig(path)
    check("matplotlib figure saved", path.stat().st_size > 1000)

from conceptflow.visualization import plot_nested  # noqa: E402

nested = plot_nested(
    living.subcontext(attributes=list(range(5))),
    inner=living.subcontext(attributes=list(range(5, 9))),
)
check("nested line diagram page", "conceptflow-nested-svg" in nested.html)

print(f"\nConceptFlow {cf.__version__} from {location.parent}: all smoke checks passed.")
