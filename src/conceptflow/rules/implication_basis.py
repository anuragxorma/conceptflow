"""
Sklearn-compatible Duquenne-Guigues (stem) basis estimator.
"""

from __future__ import annotations

import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from conceptflow.algorithms._membership import contains_sets
from conceptflow.algorithms.implications import Implication, compute_canonical_basis
from conceptflow.validation import check_binary_context_input
from conceptflow.validation.input import check_feature_names, remember_feature_names


class ImplicationBasisEstimator(TransformerMixin, BaseEstimator):
    """
    Sklearn-compatible estimator for the Duquenne-Guigues (stem) basis.

    Fits the canonical basis of exact implications from binary data and
    transforms objects into premise-satisfaction feature vectors: one
    boolean feature per implication, True if the object's own attributes
    contain that implication's premise.

    The fitted implications are accessible via ``get_implications()`` for
    FCA-specific downstream work (e.g. printing them, or checking which
    ones a new object would violate).

    Parameters
    ----------
    output:
        Output format for transform. Supported values are:
        - ``"array"``: numpy bool array (default, pipeline-compatible)
        - ``"dataframe"``: pandas DataFrame with implication labels as columns
        - ``"sparse"``: scipy sparse CSR matrix
    """

    def __init__(self, output: str = "array"):
        self.output = output

    def fit(self, X, y=None):
        """
        Compute the canonical basis from input data.
        """
        if self.output not in {"array", "dataframe", "sparse"}:
            raise ValueError('output must be "array", "dataframe", or "sparse".')

        context = check_binary_context_input(X)
        remember_feature_names(self, X)

        self.context_ = context
        self.n_features_in_ = context.n_attributes
        self.feature_names_in_ = context.attributes
        self.object_names_in_ = context.objects
        self.implications_ = compute_canonical_basis(context)
        self.feature_names_out_ = tuple(
            self._implication_label(imp) for imp in self.implications_
        )

        return self

    def _implication_label(self, imp: Implication) -> str:
        premise = ", ".join(imp.premise_names(self.context_)) or "∅"
        conclusion = ", ".join(imp.conclusion_names(self.context_))
        return f"{{{premise}}}->{{{conclusion}}}"

    def transform(self, X):
        """
        Transform objects into premise-satisfaction feature vectors.

        Each column corresponds to one implication in the fitted basis.
        The value is True if the object's attribute set contains that
        implication's premise (regardless of whether it was one of the
        objects used to fit the basis), False otherwise.
        """
        check_is_fitted(self, "implications_")

        check_feature_names(self, X)
        context = check_binary_context_input(X)

        if context.n_attributes != self.n_features_in_:
            raise ValueError(
                f"X has {context.n_attributes} features, but "
                f"{self.__class__.__name__} is expecting "
                f"{self.n_features_in_} features as input"
            )

        data = contains_sets(
            context.incidence, [imp.premise for imp in self.implications_]
        )

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

    def get_implications(self) -> tuple[Implication, ...]:
        """
        Return the fitted canonical basis.
        """
        check_is_fitted(self, "implications_")
        return tuple(self.implications_)

    def get_feature_names_out(self, input_features=None):
        """
        Return one label per implication as output feature names.
        """
        check_is_fitted(self, "feature_names_out_")
        return pd.Index(self.feature_names_out_, dtype="object").to_numpy()

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.sparse = True
        # Outputs are boolean membership indicators (or 0/1/NaN floats),
        # whatever the input dtype; tell sklearn not to expect preservation.
        tags.transformer_tags.preserves_dtype = []
        return tags
