"""
Basic concept lattice construction, on Ganter & Wille's "living beings and
water" context (bundled with ConceptFlow).

Run with:

    python examples/basic_lattice.py
"""

import conceptflow as cf
from conceptflow.datasets import load_context


def main():
    context = load_context("livingbeings_en")
    lattice = cf.ConceptLattice.from_context(context)

    print(context)
    print(lattice)
    print("Number of concepts:", lattice.n_concepts)
    print("Number of Hasse edges:", len(lattice.edges))

    print("\nConcepts (objects | shared attributes):")
    for concept in lattice.concepts:
        objects = ", ".join(context.objects[g] for g in sorted(concept.extent)) or "-"
        attributes = ", ".join(context.attributes[m] for m in sorted(concept.intent)) or "-"
        print(f"  {concept.stable_id()}  {objects}  |  {attributes}")


if __name__ == "__main__":
    main()
