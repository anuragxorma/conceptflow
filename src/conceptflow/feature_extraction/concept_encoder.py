"""
Concept-based feature extraction.

This module provides sklearn-compatible encoders that turn FCA structures
into machine-learning feature representations.
"""

from __future__ import annotations

import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from conceptflow.algorithms._membership import contains_sets
from conceptflow.core import ConceptLattice as CoreConceptLattice
from conceptflow.validation import check_binary_context_input
from conceptflow.validation.input import check_feature_names, remember_feature_names


class ConceptMembershipEncoder(TransformerMixin, BaseEstimator):
    """
    Encode objects by membership in formal concepts.

    Each output feature corresponds to one formal concept. For each object,
    the feature value is True if the object belongs to the concept extent,
    and False otherwise.

    Parameters
    ----------
    algorithm:
        Concept enumeration algorithm used when fitting from a context.

    output:
        Output format. Supported values are:
        - "dataframe" (default)
        - "array"
        - "sparse"

    Notes
    -----
    To fit directly from a pre-computed ConceptLattice, use the classmethod
    ``ConceptMembershipEncoder.from_lattice(lattice)`` instead of ``fit()``.
    """

    def __init__(
        self,
        algorithm: str = "nextclosure",
        output: str = "dataframe",
    ):
        self.algorithm = algorithm
        self.output = output

    @classmethod
    def from_lattice(
        cls, lattice: CoreConceptLattice, output: str = "dataframe"
    ) -> "ConceptMembershipEncoder":
        """
        Return a fitted encoder from a pre-computed ConceptLattice.

        Use this when you already have a lattice and want to skip
        re-computing concepts during fit.

        Parameters
        ----------
        lattice:
            A fitted CoreConceptLattice.

        output:
            Output format for transform. Same options as the constructor.

        Returns
        -------
        ConceptMembershipEncoder
            A fitted encoder instance.
        """
        encoder = cls(output=output)
        encoder._fit_from_lattice(lattice)
        return encoder

    def fit(self, X, y=None):
        """
        Learn concepts from a binary FormalContext or array input.

        Parameters
        ----------
        X:
            Binary FormalContext, numpy array, pandas DataFrame, or
            scipy sparse matrix. To fit from a ConceptLattice directly,
            use ``ConceptMembershipEncoder.from_lattice(lattice)`` instead.
        """
        context = check_binary_context_input(X)
        remember_feature_names(self, X)
        lattice = CoreConceptLattice.from_context(
            context,
            algorithm=self.algorithm,
        )
        self._fit_from_lattice(lattice)
        return self

    def _fit_from_lattice(self, lattice: CoreConceptLattice) -> None:
        if self.output not in {"dataframe", "array", "sparse"}:
            raise ValueError('output must be "dataframe", "array", or "sparse".')

        self.context_ = lattice.context
        self.lattice_ = lattice
        self.concepts_ = lattice.concepts
        self.feature_names_out_ = tuple(
            concept.stable_id() for concept in self.concepts_
        )
        self.object_names_in_ = lattice.context.objects
        self.feature_names_in_ = lattice.context.attributes
        self.n_features_in_ = lattice.context.n_attributes

    def transform(self, X):
        """
        Transform objects into concept-membership features.
        """
        check_is_fitted(self, "concepts_")

        check_feature_names(self, X)
        context = check_binary_context_input(X)

        if context.n_attributes != self.n_features_in_:
            raise ValueError(
                f"X has {context.n_attributes} features, but "
                f"{self.__class__.__name__} is expecting "
                f"{self.n_features_in_} features as input"
            )

        # Membership must be recomputed from each object's own attribute set,
        # not looked up by position in the fit-time extents: a concept's
        # extent only lists fit-time object indices, which are meaningless
        # for new data (different objects, different order, or a different
        # batch size). An object belongs to a concept iff its attribute set
        # contains that concept's intent -- the standard FCA notion of
        # concept membership, which generalizes correctly to objects the
        # lattice was never fit on.
        data = contains_sets(
            context.incidence, [concept.intent for concept in self.concepts_]
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

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.sparse = True
        # Outputs are boolean membership indicators (or 0/1/NaN floats),
        # whatever the input dtype; tell sklearn not to expect preservation.
        tags.transformer_tags.preserves_dtype = []
        return tags

    def get_feature_names_out(self, input_features=None):
        """
        Return concept feature names.
        """
        check_is_fitted(self, "feature_names_out_")

        return pd.Index(self.feature_names_out_, dtype="object").to_numpy()
