"""
Concept lattice construction using CloseByOne.

Run from the project root:

    python examples/closebyone_lattice.py
"""

import numpy as np

import conceptflow as cf


def main():
    context = cf.FormalContext.from_array(
        np.array([
            [1, 1, 0],
            [1, 0, 1],
            [0, 1, 1],
        ]),
        objects=["g1", "g2", "g3"],
        attributes=["m1", "m2", "m3"],
    )

    lattice = cf.ConceptLattice.from_context(
        context,
        algorithm="closebyone",
    )

    print(context)
    print(lattice)
    print("Number of concepts:", lattice.n_concepts)
    print("Number of Hasse edges:", len(lattice.edges))


if __name__ == "__main__":
    main()