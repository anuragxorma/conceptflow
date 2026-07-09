"""
Sklearn-compatible conceptual scaler.

This module transforms many-valued contexts into binary formal contexts
using explicit conceptual scale definitions.
"""

from __future__ import annotations

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

import numpy as np
import pandas as pd
from scipy import sparse

from conceptflow.core import FormalContext, ManyValuedContext
from conceptflow.preprocessing.scales import Scale


class ConceptualScaler(TransformerMixin, BaseEstimator):
    """
    Apply conceptual scales to many-valued data.

    Parameters
    ----------
    scales:
        List of Scale objects.

    output:
        Output format for transform. Supported values are:
        - "array": numpy boolean array (default, pipeline-compatible)
        - "dataframe": pandas DataFrame
        - "sparse": scipy sparse CSR matrix
        - "context": FormalContext (FCA domain object; not pipeline-compatible
          unless the next step also accepts FormalContext)
    """

    def __init__(self, scales=None, output: str = "array"):
        self.scales = scales
        self.output = output

    def fit(self, X, y=None):
        """
        Validate scales and remember input metadata.
        """
        mvc = self._to_many_valued_context(X)

        if self.scales is None:
            raise ValueError(
                "ConceptualScaler currently requires explicit scales. "
                "Automatic scaling is not implemented yet."
            )

        if not isinstance(self.scales, list):
            raise TypeError("scales must be a list of Scale objects.")

        for scale in self.scales:
            if not isinstance(scale, Scale):
                raise TypeError("Every entry in scales must be a Scale object.")

            if scale.source_attribute not in mvc.attributes:
                raise KeyError(
                    f'Scale refers to unknown source attribute '
                    f'"{scale.source_attribute}".'
                )

        if self.output not in {"context", "dataframe", "array", "sparse"}:
            raise ValueError(
                'output must be one of "context", "dataframe", "array", or "sparse".'
            )

        self.input_context_ = mvc
        self.scales_ = list(self.scales)
        self.feature_names_in_ = mvc.attributes
        self.n_features_in_ = mvc.n_attributes
        self.object_names_in_ = mvc.objects
        self.scaled_attributes_ = self._scaled_attributes(mvc)

        return self

    def transform(self, X):
        """
        Apply conceptual scaling.
        """
        check_is_fitted(self, "scales_")

        mvc = self._to_many_valued_context(X)

        if mvc.n_attributes != self.n_features_in_:
            raise ValueError(
                f"X has {mvc.n_attributes} features, but ConceptualScaler is "
                f"expecting {self.n_features_in_} features as input."
            )

        binary_df = self._scale_to_dataframe(mvc)

        if self.output == "dataframe":
            return binary_df

        if self.output == "array":
            return binary_df.to_numpy(dtype=bool)

        if self.output == "sparse":
            return sparse.csr_matrix(binary_df.values, dtype=bool)

        return FormalContext.from_dataframe(binary_df)

    def _to_many_valued_context(self, X) -> ManyValuedContext:
        if isinstance(X, ManyValuedContext):
            return X

        if isinstance(X, pd.DataFrame):
            if X.shape[0] == 0:
                raise ValueError(
                    f"0 sample(s) (shape={X.shape}) while a minimum of 1 is required."
                )
            if X.shape[1] == 0:
                raise ValueError(
                    f"0 feature(s) (shape={X.shape}) while a minimum of 1 is required."
                )
            return ManyValuedContext.from_dataframe(X)

        if sparse.issparse(X):
            raise TypeError(
                "Sparse input is not supported by ConceptualScaler. "
                "Pass a pandas DataFrame with named columns instead."
            )

        arr = np.asarray(X)

        if arr.ndim != 2:
            raise ValueError(
                f"Expected 2D array, got {arr.ndim}D array instead. "
                "Reshape your data using array.reshape(-1, 1) if it "
                "contains a single feature."
            )

        if np.iscomplexobj(arr):
            raise ValueError("Complex data not supported by ConceptualScaler.")

        if arr.shape[0] == 0:
            raise ValueError(
                f"0 sample(s) (shape={arr.shape}) while a minimum of 1 is required."
            )

        if arr.shape[1] == 0:
            raise ValueError(
                f"0 feature(s) (shape={arr.shape}) while a minimum of 1 is required."
            )

        return ManyValuedContext.from_array(
            arr,
            attributes=[f"x{i}" for i in range(arr.shape[1])],
        )

    def _scaled_attributes(self, mvc: ManyValuedContext) -> list[str]:
        attrs: list[str] = []

        for scale in self.scales_:
            try:
                attrs.extend(scale.binary_attributes(mvc))
            except TypeError as exc:
                raise TypeError(
                    f"argument must be a string or number, not "
                    f"'{type(mvc.data[scale.source_attribute].iloc[0]).__name__}'"
                ) from exc

        if len(set(attrs)) != len(attrs):
            duplicates = sorted(
                {attr for attr in attrs if attrs.count(attr) > 1}
            )
            raise ValueError(
                "Duplicate scaled attribute names generated: "
                f"{duplicates}"
            )

        return attrs

    def _scale_object(
        self,
        obj: str,
        mvc: ManyValuedContext,
    ) -> dict[str, bool]:
        binary_row: dict[str, bool] = {}

        for scale in self.scales_:
            value = mvc.get_value(obj, scale.source_attribute)
            encoded = scale.encode_value(value, mvc)

            overlap = set(binary_row) & set(encoded)
            if overlap:
                raise ValueError(
                    "Binary attribute name collision detected during scaling: "
                    f"{sorted(overlap)}"
                )

            binary_row.update(encoded)

        return binary_row

    def _scale_to_dataframe(self, mvc: ManyValuedContext) -> pd.DataFrame:
        data = []

        for obj in mvc.objects:
            row_encoding = self._scale_object(obj, mvc)
            row = [row_encoding.get(attr, False) for attr in self.scaled_attributes_]
            data.append(row)

        return pd.DataFrame(
            data,
            index=mvc.objects,
            columns=self.scaled_attributes_,
            dtype=bool,
        )

    def get_feature_names_out(self, input_features=None):
        """
        Return output feature names for the scaled binary attributes.
        """
        check_is_fitted(self, "scaled_attributes_")

        if input_features is not None:
            input_features = tuple(map(str, input_features))

            if input_features != tuple(self.feature_names_in_):
                raise ValueError(
                    "input_features does not match feature_names_in_."
                )

        return pd.Index(self.scaled_attributes_, dtype="object").to_numpy()
