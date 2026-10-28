"""
D3 lattice visualization example using Ganter & Wille's triangles context
(bundled with ConceptFlow).

Run with:

    python examples/visualize_lattice.py

Output:

    triangles_lattice.html
"""

import conceptflow as cf
from conceptflow.datasets import load_context


def main() -> None:
    ctx = load_context("triangles_en")
    lattice = cf.ConceptLattice.from_context(ctx)

    print("Context: triangles_en")
    print(f"Context size: ({ctx.n_objects}, {ctx.n_attributes})")
    print(f"Number of concepts: {lattice.n_concepts}")

    graph_data = cf.plot_lattice(
        lattice,
        backend="graph_data",
        layout="dimflux",
    )

    fig = cf.visualization.render_with_d3(
        graph_data,
        width=1150,
        height=780,
    )

    output_path = fig.open("triangles_lattice.html")

    print(f"Wrote and opened {output_path}")


if __name__ == "__main__":
    main()
