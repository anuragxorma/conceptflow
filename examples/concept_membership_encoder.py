"""
Concept membership feature extraction.

Run from the project root:

    python examples/concept_membership_encoder.py
"""

import numpy as np

import conceptflow as cf
from conceptflow.feature_extraction import (
    ConceptMembershipEncoder,
)


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

    encoder = ConceptMembershipEncoder(
        algorithm="nextclosure",
        output="dataframe",
    )

    features = encoder.fit_transform(context)

    print("Original formal context:")
    print(context.to_dataframe())

    print("\nConcept membership features:")
    print(features)

    print("\nConcept feature names:")
    for concept, feature_name in zip(
        encoder.concepts_,
        encoder.feature_names_out_,
    ):
        print(feature_name, "->", concept.signature())


if __name__ == "__main__":
    main()