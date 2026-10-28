"""
Interpretable concept features on real data: the UCI Mushroom data set.

The raw many-valued data is scaled nominally, the most stable frequent
concepts become features, and a logistic regression predicts whether a
mushroom is poisonous. Every feature is a concept -- a set of attributes
that a group of mushrooms shares -- so the model can be read directly.

The data (about 370 KB) is downloaded on the first run and cached.

Run with:

    python examples/mushroom_classification.py
"""

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline

from conceptflow.cluster import ConceptLatticeEstimator
from conceptflow.datasets import load_mushroom
from conceptflow.preprocessing import ConceptualScaler, NominalScale


def main():
    raw = load_mushroom(scaled=False).data
    X = raw.drop(columns="class")
    y = (raw["class"] == "poisonous").astype(int)

    pipeline = make_pipeline(
        # Unknown stalk roots get no stalk-root attribute; a value that is
        # missing from a cross-validation training fold is ignored.
        ConceptualScaler(
            [NominalScale(column) for column in X.columns],
            missing="false",
            handle_unknown="ignore",
            output="dataframe",
        ),
        # Concepts shared by at least 10% of the mushrooms; the 20 most
        # stable ones become features.
        ConceptLatticeEstimator(min_support=0.1, select="stability", n_concepts=20),
        LogisticRegression(max_iter=1000),
    )

    scores = cross_val_score(pipeline, X, y, cv=5)
    print(f"Accuracy (5-fold cross-validation): {scores.mean():.3f}")

    pipeline.fit(X, y)
    concepts, model = pipeline[1], pipeline[2]
    attributes = concepts.context_.attributes
    print("\nThe concept features with the largest weights (positive: poisonous):")
    ranked = sorted(zip(concepts.concepts_, model.coef_[0]), key=lambda pair: -abs(pair[1]))
    for concept, weight in ranked[:5]:
        names = " & ".join(attributes[m] for m in sorted(concept.intent))
        print(f"  {weight:+.2f}  {len(concept.extent):5} mushrooms  {names}")


if __name__ == "__main__":
    main()
