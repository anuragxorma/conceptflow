"""
Ordinal two-factorization: exact for a two-factorizable context, and
Ord2Factor for Ganter & Wille's triangles context (bundled), which is not.

Run with:

    python examples/exact_ordinal_two_factorization.py
"""

import numpy as np

import conceptflow as cf
from conceptflow.datasets import load_context
from conceptflow.decomposition import ExactOrdinalTwoFactorizer, Ord2Factor


def main():
    context = cf.FormalContext.from_array(
        np.array([
            [1, 0, 0],
            [1, 1, 0],
            [1, 1, 1],
        ]),
        objects=["g1", "g2", "g3"],
        attributes=["m1", "m2", "m3"],
    )

    model = ExactOrdinalTwoFactorizer()
    model.fit(context)

    factor_1, factor_2 = model.factors_

    print("Context:")
    print(context.to_dataframe())

    print("\nCoverage:", model.coverage_)

    print("\nFactor 1:")
    print(sorted(factor_1.relation))

    print("\nFactor 2:")
    print(sorted(factor_2.relation))

    # Most contexts are not two-factorizable: Ord2Factor drops as few
    # crosses as possible, then factorizes the rest.
    triangles = load_context("triangles_en")
    for method in ("exact", "greedy"):
        model = Ord2Factor(method=method, random_state=0).fit(triangles)
        dropped = sorted(
            (triangles.objects[g], triangles.attributes[m]) for g, m in model.removed_incidence_
        )
        print(f"\nOrd2Factor ({method}): coverage {model.coverage_:.2f}, dropped {dropped}")


if __name__ == "__main__":
    main()
