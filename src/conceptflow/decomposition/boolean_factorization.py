"""
Boolean matrix factorization: GreConD and ASSO.

A Boolean factorization writes a binary data matrix X (objects x attributes)
approximately as a Boolean product ``X ≈ S ∘ B``: ``B`` (factors x
attributes) says which attributes make up each factor, ``S`` (objects x
factors) which objects have it, and an object has an attribute iff it has
some factor containing it.

- ``GreConD`` (Belohlavek & Vychodil, "Discovery of optimal factors in
  binary data via a novel method of matrix decomposition", J. Comput. Syst.
  Sci. 76, 2010) uses formal concepts as factors: each factor is a concept
  (extent, intent), so it never covers a 0, and factors are added greedily
  until the requested share of the 1s is covered.
- ``ASSO`` (Miettinen et al., "The Discrete Basis Problem", IEEE TKDE 20,
  2008) builds candidate factors from pairwise association confidences and
  picks them greedily, trading covered 1s against covered 0s. Its factors may
  cover some 0s, which often gives fewer, broader factors.

Both are scikit-learn transformers, like ``sklearn.decomposition.NMF``:
``fit`` learns ``components_`` (B), ``transform`` gives S for any data, and
``inverse_transform`` rebuilds the data from S.
"""

from __future__ import annotations

import numbers

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator, ClassNamePrefixFeaturesOutMixin, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from conceptflow.algorithms._membership import contains_sets
from conceptflow.core import Concept
from conceptflow.validation import check_binary_context_input
from conceptflow.validation.input import check_feature_names, remember_feature_names


def _counting_dtype(n_cells: int):
    # float32 sums 0/1 values exactly up to 2**24; beyond that use float64.
    return np.float32 if n_cells < 2**24 else np.float64


class _BooleanFactorization(ClassNamePrefixFeaturesOutMixin, TransformerMixin, BaseEstimator):
    """Shared input handling, output formats and reconstruction."""

    def _validate_common(self) -> None:
        if self.output not in {"array", "dataframe", "sparse"}:
            raise ValueError('output must be "array", "dataframe", or "sparse".')
        n = self.n_components
        if n is not None and (
            isinstance(n, bool) or not isinstance(n, numbers.Integral) or n < 1
        ):
            raise ValueError("n_components must be None or a positive integer.")

    def _check_fit_input(self, X) -> np.ndarray:
        self._validate_common()
        context = check_binary_context_input(X)
        remember_feature_names(self, X)
        self.n_features_in_ = context.n_attributes
        self.feature_names_in_ = context.attributes
        return np.asarray(context.incidence)

    def _check_transform_input(self, X):
        check_is_fitted(self, "components_")
        self._validate_common()
        check_feature_names(self, X)
        context = check_binary_context_input(X)
        if context.n_attributes != self.n_features_in_:
            raise ValueError(
                f"X has {context.n_attributes} features, but "
                f"{self.__class__.__name__} is expecting {self.n_features_in_} "
                "features as input"
            )
        return context

    def _finish_fit(self, data: np.ndarray, components: list[np.ndarray]) -> None:
        n_features = data.shape[1]
        self.components_ = (
            np.array(components, dtype=bool).reshape(len(components), n_features)
        )
        self.n_components_ = len(components)
        self._n_features_out = self.n_components_

        usage = self._usage(data)
        rebuilt = self._product(usage)
        self.reconstruction_error_ = int((rebuilt != data).sum())
        ones = int(data.sum())
        self.coverage_ = float((rebuilt & data).sum() / ones) if ones else 1.0

    def _product(self, usage: np.ndarray) -> np.ndarray:
        if usage.shape[1] == 0:
            return np.zeros((usage.shape[0], self.components_.shape[1]), dtype=bool)
        dtype = _counting_dtype(usage.size)
        return (usage.astype(dtype) @ self.components_.astype(dtype)) > 0

    def transform(self, X):
        """
        Factor memberships S: one boolean column per factor.
        """
        context = self._check_transform_input(X)
        usage = self._usage(np.asarray(context.incidence))
        names = self.get_feature_names_out()

        if self.output == "array":
            return usage
        if self.output == "sparse":
            return sparse.csr_matrix(usage, dtype=bool)
        return pd.DataFrame(usage, index=context.objects, columns=names, dtype=bool)

    def inverse_transform(self, X):
        """
        Rebuild binary data from factor memberships: the Boolean product
        ``S ∘ components_``.
        """
        check_is_fitted(self, "components_")
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        elif sparse.issparse(X):
            X = X.toarray()
        usage = np.asarray(X)
        if usage.ndim != 2 or usage.shape[1] != self.n_components_:
            raise ValueError(
                f"Expected an array with {self.n_components_} columns (one per factor)."
            )
        return self._product(usage > 0)

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.sparse = True
        tags.transformer_tags.preserves_dtype = []
        return tags


class GreConD(_BooleanFactorization):
    """
    Boolean factorization with formal concepts as factors (GreConD).

    Factors are chosen greedily: each new factor is grown attribute by
    attribute, always taking the attribute whose concept covers the most
    still-uncovered 1s, and is added once no attribute improves it. Factors
    never cover a 0, so ``X`` is rebuilt exactly once all 1s are covered.

    Parameters
    ----------
    n_components:
        Maximum number of factors. ``None`` (default): as many as needed to
        reach ``min_coverage``.

    min_coverage:
        Stop once this share of the 1s is covered (default 1.0: rebuild X
        exactly). Lower values give fewer factors that explain most of X.

    output:
        ``"array"`` (default), ``"dataframe"`` or ``"sparse"``.

    Attributes
    ----------
    components_:
        Boolean array (n_components_, n_features): the factors' attributes
        (their intents).

    factors_:
        The factors as formal concepts of the training data (extents are
        indices of training objects).

    n_components_:
        Number of factors found.

    coverage_:
        Share of the training data's 1s covered by the factors.

    reconstruction_error_:
        Number of training cells that ``inverse_transform(transform(X))``
        gets wrong (uncovered 1s; GreConD never covers a 0).

    Notes
    -----
    ``transform`` gives an object a factor iff it has all of the factor's
    attributes, so for the training data the memberships are exactly the
    factors' extents.
    """

    def __init__(self, n_components=None, min_coverage: float = 1.0, output: str = "array"):
        self.n_components = n_components
        self.min_coverage = min_coverage
        self.output = output

    def fit(self, X, y=None):
        """Find the factors of binary data X."""
        if (
            isinstance(self.min_coverage, bool)
            or not isinstance(self.min_coverage, numbers.Real)
            or not 0.0 < self.min_coverage <= 1.0
        ):
            raise ValueError("min_coverage must be a number in (0, 1].")
        data = self._check_fit_input(X)
        n_objects, n_features = data.shape

        dtype = _counting_dtype(data.size)
        absent = (~data).astype(dtype)
        uncovered = data.copy()
        ones = int(data.sum())
        allowed_uncovered = ones - int(np.ceil(self.min_coverage * ones - 1e-9))

        components: list[np.ndarray] = []
        extents: list[np.ndarray] = []
        while int(uncovered.sum()) > allowed_uncovered and (
            self.n_components is None or len(components) < self.n_components
        ):
            intent = np.zeros(n_features, dtype=bool)
            extent = np.ones(n_objects, dtype=bool)
            value = 0
            while True:
                # Candidate j: the concept generated by intent + {j}.
                candidate_extents = extent[:, None] & data  # column j: extent of intent+{j}
                as_float = candidate_extents.astype(dtype)
                # Row j: attributes shared by all objects of that extent.
                candidate_intents = (as_float.T @ absent) == 0
                covered = ((as_float.T @ uncovered.astype(dtype)) * candidate_intents).sum(axis=1)
                covered[intent] = -1
                best = int(np.argmax(covered))
                if covered[best] <= value:
                    break
                value = int(covered[best])
                extent = candidate_extents[:, best]
                intent = candidate_intents[best]

            components.append(intent)
            extents.append(extent)
            uncovered &= ~(extent[:, None] & intent[None, :])

        self.factors_ = tuple(
            Concept(
                extent=frozenset(np.flatnonzero(e).tolist()),
                intent=frozenset(np.flatnonzero(i).tolist()),
            )
            for e, i in zip(extents, components)
        )
        self._finish_fit(data, components)
        return self

    def _usage(self, data: np.ndarray) -> np.ndarray:
        if self.components_.shape[0] == 0:
            return np.zeros((data.shape[0], 0), dtype=bool)
        intents = [frozenset(np.flatnonzero(row).tolist()) for row in self.components_]
        return contains_sets(data, intents)


class ASSO(_BooleanFactorization):
    """
    Boolean factorization from association rules (ASSO).

    Candidate factors are the rows of the association matrix: candidate
    ``i`` contains attribute ``j`` iff the confidence of ``i -> j`` (the
    share of objects with ``i`` that also have ``j``) is at least
    ``threshold``. Factors are then picked greedily: each step takes the
    candidate, and the objects using it, that most increase

        positive_weight * (covered 1s) - negative_weight * (covered 0s).

    Unlike GreConD, factors may cover some 0s.

    Parameters
    ----------
    n_components:
        Number of factors. ``None`` (default): keep adding factors while one
        improves the cover (at most one per attribute).

    threshold:
        Confidence threshold τ in (0, 1] for the association matrix
        (default 0.7). Lower values give broader candidate factors.

    positive_weight, negative_weight:
        The weights w⁺ and w⁻ of covered 1s and covered 0s (default 1 each).
        A larger ``negative_weight`` avoids covering 0s.

    output:
        ``"array"`` (default), ``"dataframe"`` or ``"sparse"``.

    Attributes
    ----------
    components_:
        Boolean array (n_components_, n_features): the factors' attributes.

    n_components_, coverage_, reconstruction_error_:
        As for ``GreConD``; the error counts uncovered 1s and covered 0s.

    Notes
    -----
    ``transform`` assigns factors to each object independently, with the
    same greedy rule as ``fit``: factors in order, each used if it improves
    the weighted cover of that object's row.
    """

    def __init__(
        self,
        n_components=None,
        threshold: float = 0.7,
        positive_weight: float = 1.0,
        negative_weight: float = 1.0,
        output: str = "array",
    ):
        self.n_components = n_components
        self.threshold = threshold
        self.positive_weight = positive_weight
        self.negative_weight = negative_weight
        self.output = output

    def _check_weights(self) -> None:
        if (
            isinstance(self.threshold, bool)
            or not isinstance(self.threshold, numbers.Real)
            or not 0.0 < self.threshold <= 1.0
        ):
            raise ValueError("threshold must be a number in (0, 1].")
        for name in ("positive_weight", "negative_weight"):
            weight = getattr(self, name)
            if isinstance(weight, bool) or not isinstance(weight, numbers.Real) or weight <= 0:
                raise ValueError(f"{name} must be a positive number.")

    def fit(self, X, y=None):
        """Find the factors of binary data X."""
        self._check_weights()
        data = self._check_fit_input(X)
        n_features = data.shape[1]
        dtype = _counting_dtype(data.size)
        as_float = data.astype(dtype)

        support = as_float.sum(axis=0)
        together = as_float.T @ as_float
        with np.errstate(divide="ignore", invalid="ignore"):
            confidence = np.where(support[:, None] > 0, together / support[:, None], 0.0)
        candidates = confidence >= self.threshold - 1e-12

        limit = n_features if self.n_components is None else self.n_components
        components: list[np.ndarray] = []
        covered = np.zeros_like(data)
        while len(components) < limit:
            gains = self._row_gains(data, covered, candidates)
            totals = np.where(gains > 0, gains, 0).sum(axis=0)
            best = int(np.argmax(totals))
            if totals[best] <= 0:
                break
            components.append(candidates[best].copy())
            users = gains[:, best] > 0
            covered |= users[:, None] & candidates[best][None, :]

        self._finish_fit(data, components)
        return self

    def _row_gains(self, data, covered, candidates) -> np.ndarray:
        """Per object and candidate: the change in weighted cover if used."""
        dtype = _counting_dtype(data.size)
        new_ones = (data & ~covered).astype(dtype)
        new_zeros = (~data & ~covered).astype(dtype)
        as_float = candidates.astype(dtype).T
        return self.positive_weight * (new_ones @ as_float) - self.negative_weight * (
            new_zeros @ as_float
        )

    def _usage(self, data: np.ndarray) -> np.ndarray:
        usage = np.zeros((data.shape[0], self.components_.shape[0]), dtype=bool)
        covered = np.zeros_like(data)
        for k, component in enumerate(self.components_):
            gain = self._row_gains(data, covered, component[None, :])[:, 0]
            usage[:, k] = gain > 0
            covered |= usage[:, k][:, None] & component[None, :]
        return usage
