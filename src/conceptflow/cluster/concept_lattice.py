"""
Sklearn-compatible concept lattice estimator.
"""

from __future__ import annotations

import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from conceptflow.core import ConceptLattice as CoreConceptLattice
from conceptflow.validation import check_binary_context_input


class ConceptLatticeEstimator(TransformerMixin, BaseEstimator):
    """
    Sklearn-compatible estimator for concept lattice construction.

    Fits a formal concept lattice from binary data and transforms objects
    into concept-membership feature vectors (one boolean feature per concept).

    The fitted lattice is accessible via ``get_lattice()`` for FCA-specific
    downstream work.

    Parameters
    ----------
    algorithm:
        Concept enumeration algorithm. Supported values are
        ``"nextclosure"`` and ``"bruteforce"``.

    output:
        Output format for transform. Supported values are:
        - ``"array"``: numpy bool array (default, pipeline-compatible)
        - ``"dataframe"``: pandas DataFrame with concept stable IDs as columns
        - ``"sparse"``: scipy sparse CSR matrix
    """

    def __init__(self, algorithm: str = "nextclosure", output: str = "array"):
        self.algorithm = algorithm
        self.output = output

    def fit(self, X, y=None):
        """
        Build the concept lattice from input data.
        """
        if self.output not in {"array", "dataframe", "sparse"}:
            raise ValueError('output must be "array", "dataframe", or "sparse".')

        context = check_binary_context_input(X)

        self.context_ = context
        self.n_features_in_ = context.n_attributes
        self.feature_names_in_ = context.attributes
        self.object_names_in_ = context.objects
        self.lattice_ = CoreConceptLattice.from_context(
            context,
            algorithm=self.algorithm,
        )
        self.concepts_ = self.lattice_.concepts
        self.edges_ = self.lattice_.edges
        self.feature_names_out_ = tuple(
            concept.stable_id() for concept in self.concepts_
        )

        return self

    def transform(self, X):
        """
        Transform objects into concept-membership feature vectors.

        Each column corresponds to one formal concept. The value is True if
        the object belongs to the concept's extent, False otherwise.
        """
        check_is_fitted(self, "concepts_")

        context = check_binary_context_input(X)

        if context.n_attributes != self.n_features_in_:
            raise ValueError(
                f"X has {context.n_attributes} features, but "
                f"{self.__class__.__name__} is expecting "
                f"{self.n_features_in_} features as input"
            )

        data = [
            [object_index in concept.extent for concept in self.concepts_]
            for object_index in range(context.n_objects)
        ]

        output_df = pd.DataFrame(
            data,
            index=context.objects,
            columns=self.feature_names_out_,
            dtype=bool,
        )

        if self.output == "array":
            return output_df.to_numpy(dtype=bool)

        if self.output == "sparse":
            return sparse.csr_matrix(output_df.values, dtype=bool)

        return output_df

    def get_feature_names_out(self, input_features=None):
        """
        Return concept stable IDs as output feature names.
        """
        check_is_fitted(self, "feature_names_out_")
        return pd.Index(self.feature_names_out_, dtype="object").to_numpy()

    def get_lattice(self) -> CoreConceptLattice:
        """
        Return the fitted ConceptLattice.
        """
        check_is_fitted(self, "lattice_")
        return self.lattice_

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.sparse = True
        return tags
