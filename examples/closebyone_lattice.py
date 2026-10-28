"""
Concept lattice construction with Close-by-One, on Ganter & Wille's
"living beings and water" context (bundled with ConceptFlow).

Run with:

    python examples/closebyone_lattice.py
"""

import conceptflow as cf
from conceptflow.datasets import load_context


def main():
    context = load_context("livingbeings_en")
    lattice = cf.ConceptLattice.from_context(context, algorithm="closebyone")
    same = cf.ConceptLattice.from_context(context, algorithm="nextclosure")

    print(context)
    print(lattice)
    print("Number of concepts:", lattice.n_concepts)
    print("Number of Hasse edges:", len(lattice.edges))
    print("Same lattice as NextClosure:", lattice == same)


if __name__ == "__main__":
    main()
