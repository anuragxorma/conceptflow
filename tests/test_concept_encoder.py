import numpy as np
import pandas as pd
from scipy import sparse

from conceptflow import ConceptLattice, FormalContext
from conceptflow.feature_extraction import ConceptMembershipEncoder


def make_context():
    return FormalContext.from_array(
        np.array([
            [1, 1, 0],
            [1, 0, 1],
            [0, 1, 1],
        ]),
        objects=["g1", "g2", "g3"],
        attributes=["m1", "m2", "m3"],
    )


def test_concept_membership_encoder_fit_transform_context_dataframe_output():
    ctx = make_context()

    encoder = ConceptMembershipEncoder(output="dataframe")
    features = encoder.fit_transform(ctx)

    assert isinstance(features, pd.DataFrame)
    assert features.shape == (3, 8)
    assert features.index.tolist() == ["g1", "g2", "g3"]
    assert len(encoder.concepts_) == 8


def test_concept_membership_encoder_array_output():
    ctx = make_context()

    encoder = ConceptMembershipEncoder(output="array")
    features = encoder.fit_transform(ctx)

    assert isinstance(features, np.ndarray)
    assert features.dtype == bool
    assert features.shape == (3, 8)


def test_concept_membership_encoder_from_lattice():
    ctx = make_context()
    lattice = ConceptLattice.from_context(ctx)

    encoder = ConceptMembershipEncoder.from_lattice(lattice)
    features = encoder.transform(ctx)

    assert features.shape == (3, 8)
    assert len(encoder.concepts_) == lattice.n_concepts


def test_concept_membership_encoder_get_feature_names_out():
    ctx = make_context()

    encoder = ConceptMembershipEncoder()
    encoder.fit(ctx)

    names = encoder.get_feature_names_out()

    assert len(names) == 8
    assert all(name.startswith("C_") for name in names)


def test_concept_membership_encoder_transform_after_fit():
    ctx = make_context()

    encoder = ConceptMembershipEncoder()
    encoder.fit(ctx)

    features = encoder.transform(ctx)

    assert features.shape == (3, 8)


def test_concept_membership_features_match_extent_membership():
    ctx = make_context()

    encoder = ConceptMembershipEncoder()
    features = encoder.fit_transform(ctx)

    for concept, feature_name in zip(encoder.concepts_, encoder.feature_names_out_):
        for object_index, object_name in enumerate(ctx.objects):
            assert features.loc[object_name, feature_name] == (
                object_index in concept.extent
            )

def test_concept_membership_encoder_set_output_pandas():
    ctx = make_context()

    encoder = ConceptMembershipEncoder(output="array")
    encoder.set_output(transform="pandas")

    features = encoder.fit_transform(ctx)

    assert isinstance(features, pd.DataFrame)
    assert features.shape == (3, 8)



def test_concept_membership_encoder_sparse_output():
    ctx = make_context()

    encoder = ConceptMembershipEncoder(output="sparse")
    features = encoder.fit_transform(ctx)

    assert sparse.issparse(features)
    assert features.shape == (3, 8)
    assert features.dtype == bool