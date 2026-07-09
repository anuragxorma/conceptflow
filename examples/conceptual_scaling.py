"""
Conceptual scaling example.

Run from the project root:

    python examples/conceptual_scaling.py
"""

import pandas as pd

from conceptflow.preprocessing import (
    ConceptualScaler,
    NominalScale,
    OrdinalScale,
    ThresholdScale,
)


def main():
    data = pd.DataFrame(
        {
            "region": ["EU", "EU", "US", "Asia"],
            "risk": ["low", "medium", "high", "medium"],
            "score": [20, 45, 80, 60],
        },
        index=["g1", "g2", "g3", "g4"],
    )

    scaler = ConceptualScaler(
        scales=[
            NominalScale("region"),
            OrdinalScale(
                "risk",
                levels=["low", "medium", "high"],
                mode="ge",
            ),
            ThresholdScale("score", thresholds=[30, 60]),
        ],
        output="dataframe",
    )

    scaled = scaler.fit_transform(data)

    print("Original data:")
    print(data)

    print("\nScaled binary data:")
    print(scaled)

    print("\nGenerated attributes:")
    print(scaler.get_feature_names_out())


if __name__ == "__main__":
    main()