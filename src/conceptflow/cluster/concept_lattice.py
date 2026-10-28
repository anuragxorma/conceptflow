"""
Sklearn-compatible concept lattice estimator.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from conceptflow.algorithms._membership import contains_sets
from conceptflow.core import ConceptLattice as CoreConceptLattice
from conceptflow.validation import check_binary_context_input
from conceptflow.validation.input import (
    check_feature_names,
    check_output_param,
    remember_feature_names,
)


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
        Concept enumeration algorithm: ``"nextclosure"``, ``"closebyone"``
        or ``"bruteforce"``.

    output:
        Output format for transform. Supported values are:
        - ``"array"``: numpy bool array (default, pipeline-compatible)
        - ``"dataframe"``: pandas DataFrame with concept stable IDs as columns
        - ``"sparse"``: scipy sparse CSR matrix

    min_support:
        Only use concepts with at least this many objects (int) or this
        fraction of the objects (float): the iceberg lattice. ``None`` uses
        the full lattice.

    max_concepts:
        Stop fitting with ``ConceptLimitExceeded`` past this many concepts.
        ``None`` (default) means no limit; past 50,000 concepts a
        ``LargeLatticeWarning`` suggests ``min_support`` or this option.

    select:
        Rank concepts by ``"stability"`` (exact),
        ``"delta"`` (Δ-measure) or ``"support"`` (extent size) and keep the
        ``n_concepts`` best as features. ``None`` keeps all concepts.

    n_concepts:
        Number of concepts to keep when ``select`` is set (ignored when
        ``select`` is None, so parameter grids can mix both).

    Attributes
    ----------
    lattice_:
        The fitted lattice (``get_lattice()``).

    concepts_:
        The concepts used as features: all concepts of ``lattice_``, or the
        selected ones.

    edges_:
        Cover edges of the whole ``lattice_`` (not only of the selected
        concepts).
    """

    def __init__(
        self,
        algorithm: str = "nextclosure",
        output: str = "array",
        min_support=None,
        max_concepts=None,
        select=None,
        n_concepts=None,
    ):
        self.algorithm = algorithm
        self.output = output
        self.min_support = min_support
        self.max_concepts = max_concepts
        self.select = select
        self.n_concepts = n_concepts

    def fit(self, X, y=None):
        """
        Build the concept lattice from input data.
        """
        check_output_param(self)

        if self.select not in {None, "stability", "delta", "support"}:
            raise ValueError('select must be None, "stability", "delta" or "support".')

        if self.select is not None and (
            isinstance(self.n_concepts, bool)
            or not isinstance(self.n_concepts, (int, np.integer))
            or self.n_concepts < 1
        ):
            raise ValueError("n_concepts must be a positive integer when select is set.")

        context = check_binary_context_input(X)
        remember_feature_names(self, X)

        self.context_ = context
        self.n_features_in_ = context.n_attributes
        self.feature_names_in_ = context.attributes
        self.object_names_in_ = context.objects
        self.lattice_ = CoreConceptLattice.from_context(
            context,
            algorithm=self.algorithm,
            min_support=self.min_support,
            max_concepts=self.max_concepts,
        )
        self.concepts_ = self._selected_concepts(self.lattice_)
        self.edges_ = self.lattice_.edges
        self.feature_names_out_ = tuple(
            concept.stable_id() for concept in self.concepts_
        )

        return self

    def _selected_concepts(self, lattice: CoreConceptLattice):
        if self.select is None:
            return lattice.concepts

        from conceptflow.metrics.measures import concept_stability, delta_measure

        if self.select == "stability":
            scores = concept_stability(lattice)
        elif self.select == "delta":
            scores = delta_measure(lattice)
        else:
            scores = np.array([len(c.extent) for c in lattice.concepts])

        # Best scores first; ties keep lattice order. The chosen concepts are
        # then listed in lattice order, so feature order is reproducible.
        best = np.argsort(-np.asarray(scores, dtype=float), kind="stable")[: self.n_concepts]
        return tuple(lattice.concepts[i] for i in sorted(best))

    def transform(self, X):
        """
        Transform objects into concept-membership feature vectors.

        Each column corresponds to one formal concept. The value is True if
        the object belongs to the concept's extent, False otherwise.
        """
        check_is_fitted(self, "concepts_")
        check_output_param(self)

        check_feature_names(self, X)
        context = check_binary_context_input(X)

        if context.n_attributes != self.n_features_in_:
            raise ValueError(
                f"X has {context.n_attributes} features, but "
                f"{self.__class__.__name__} is expecting "
                f"{self.n_features_in_} features as input"
            )

        # Membership must be recomputed from this object's own attribute set,
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
        # Outputs are boolean membership indicators (or 0/1/NaN floats),
        # whatever the input dtype; tell sklearn not to expect preservation.
        tags.transformer_tags.preserves_dtype = []
        return tags
