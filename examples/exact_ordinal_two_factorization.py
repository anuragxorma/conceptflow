"""
Exact ordinal two-factorization example.

Run from the project root:

    python examples/exact_ordinal_two_factorization.py
"""

import numpy as np

import conceptflow as cf
from conceptflow.decomposition import ExactOrdinalTwoFactorizer


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


if __name__ == "__main__":
    main()