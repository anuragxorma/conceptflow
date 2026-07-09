"""
D3 lattice visualization example using the triangles context.

Run from project root:

    python -m examples.visualize_lattice

Output:

    triangles_lattice.html
"""

from pathlib import Path

import conceptflow as cf


def main() -> None:
    cxt_path = Path("data/triangles.cxt")

    if not cxt_path.exists():
        raise FileNotFoundError(f"Context file not found: {cxt_path}")

    ctx = cf.FormalContext.from_cxt(cxt_path)

    lattice = cf.ConceptLattice.from_context(
        ctx,
        algorithm="nextclosure",
    )

    print(f"Loaded context: {cxt_path}")
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