"""
scikit-learn style ConceptFlow pipeline.

Run from the project root:

    python examples/sklearn_pipeline.py
"""

import pandas as pd
from sklearn.pipeline import Pipeline

from conceptflow.cluster import ConceptLatticeEstimator
from conceptflow.preprocessing import (
    ConceptualScaler,
    NominalScale,
    OrdinalScale,
)


def main():
    data = pd.DataFrame(
        {
            "color": ["red", "blue", "red", "green"],
            "risk": ["low", "medium", "high", "medium"],
        },
        index=["g1", "g2", "g3", "g4"],
    )

    pipe = Pipeline([
        (
            "scaling",
            ConceptualScaler(
                scales=[
                    NominalScale("color"),
                    OrdinalScale(
                        "risk",
                        levels=["low", "medium", "high"],
                        mode="ge",
                    ),
                ],
                output="context",
            ),
        ),
        (
            "lattice",
            ConceptLatticeEstimator(algorithm="nextclosure"),
        ),
    ])

    pipe.fit(data)

    lattice_step = pipe.named_steps["lattice"]

    print("Number of concepts:", len(lattice_step.concepts_))
    print("Number of Hasse edges:", len(lattice_step.edges_))

    print("\nFirst few concepts:")
    for concept in lattice_step.concepts_[:5]:
        print(concept.signature())


if __name__ == "__main__":
    main()