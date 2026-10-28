"""
Sklearn-compatible conceptual scaler.

This module transforms many-valued contexts into binary formal contexts
using explicit conceptual scale definitions.
"""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from conceptflow.core import FormalContext, IncompleteContext, ManyValuedContext
from conceptflow.core.incomplete_context import is_missing
from conceptflow.preprocessing.scales import Scale, UnknownValueError


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

    missing:
        How to scale a missing value (None, NaN, NaT, ``pd.NA`` or the
        string ``"?"``) of a scaled attribute:

        - "error": raise a ValueError (default). Missing values are never
          silently turned into crosses or non-crosses.
        - "false": the object gets none of that scale's binary attributes
          (closed-world assumption on the *binary* attributes). Note that
          for scales whose attributes are negative statements, such as
          ContranominalScale ("x!=v") or OrdinalScale(mode="le"), "none of
          the attributes" places the object at the bottom of that scale.
        - "unknown": a binary attribute is known-yes if it holds for every
          value the scale can take, known-no if it holds for none, and
          unknown otherwise (ThresholdScale has an open domain, so all its
          attributes become unknown). Outputs become three-valued:
          "context" returns an IncompleteContext, while "array" and
          "dataframe" return floats (1.0 / 0.0, NaN for unknown), the
          scikit-learn convention for missing values. "sparse" output is not supported in this mode.
          Unknown entries are independent of each other, so implications
          that follow from a scale itself (e.g. ``size>=L -> size>=M``) can
          be reported as "possible" by ``IncompleteContext.implication_status``.

    handle_unknown:
        What ``transform`` does with a value that ``NominalScale`` or
        ``ContranominalScale`` did not see during ``fit`` (as in
        scikit-learn's ``OneHotEncoder``):

        - "error": raise a ValueError (default);
        - "ignore": the value equals none of the known values, so the object
          gets no ``x=v`` attribute and every ``x!=v`` attribute. Useful in
          cross-validation, where a rare value may be missing from a
          training fold. Scales with explicitly given values (ordinal,
          dichotomic, general) still raise.
    """

    def __init__(
        self,
        scales=None,
        output: str = "array",
        missing: str = "error",
        handle_unknown: str = "error",
    ):
        self.scales = scales
        self.output = output
        self.missing = missing
        self.handle_unknown = handle_unknown

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

        scales = []
        for scale in self.scales:
            if not isinstance(scale, Scale):
                raise TypeError("Every entry in scales must be a Scale object.")
            scale = self._resolve_position(scale, mvc)

            if scale.source_attribute not in mvc.attributes:
                raise KeyError(
                    f'Scale refers to unknown source attribute '
                    f'"{scale.source_attribute}".'
                )
            scales.append(scale)

        self._validate_output_params()

        if self.missing == "error":
            for scale in scales:
                self._check_no_missing(mvc, scale.source_attribute)

        self.input_context_ = mvc
        self.scales_ = scales
        self.feature_names_in_ = mvc.attributes
        self.n_features_in_ = mvc.n_attributes
        self.object_names_in_ = mvc.objects
        self.scaled_attributes_ = self._scaled_attributes(mvc)

        return self

    @staticmethod
    def _resolve_position(scale: Scale, mvc: ManyValuedContext) -> Scale:
        """A copy of a scale given by column position, naming that column."""
        position = getattr(scale, "source_position", None)
        if position is None:
            return scale
        if not 0 <= position < mvc.n_attributes:
            raise IndexError(
                f"Scale refers to column position {position}, but the data has "
                f"{mvc.n_attributes} columns."
            )
        resolved = copy.copy(scale)
        resolved.source_attribute = mvc.attributes[position]
        return resolved

    def __setstate__(self, state):
        # Scalers pickled before the ``missing`` parameter existed (0.1.x)
        # behave like the current default.
        state.setdefault("missing", "error")
        state.setdefault("handle_unknown", "error")
        super().__setstate__(state)

    def _validate_output_params(self) -> None:
        if self.output not in {"context", "dataframe", "array", "sparse"}:
            raise ValueError(
                'output must be one of "context", "dataframe", "array", or "sparse".'
            )

        if self.missing not in {"error", "false", "unknown"}:
            raise ValueError('missing must be one of "error", "false", or "unknown".')

        if self.handle_unknown not in {"error", "ignore"}:
            raise ValueError('handle_unknown must be "error" or "ignore".')

        if self.missing == "unknown" and self.output == "sparse":
            raise ValueError(
                'output="sparse" cannot represent unknown entries; use '
                'missing="unknown" with "context", "dataframe" or "array".'
            )

    def transform(self, X):
        """
        Apply conceptual scaling.
        """
        check_is_fitted(self, "scales_")
        # Parameters may have been changed with set_params() after fit.
        self._validate_output_params()

        mvc = self._to_many_valued_context(X)

        if mvc.n_attributes != self.n_features_in_:
            raise ValueError(
                f"X has {mvc.n_attributes} features, but ConceptualScaler is "
                f"expecting {self.n_features_in_} features as input."
            )

        # Scales look their columns up by name, e.g. a scaler fitted on a
        # DataFrame cannot read a plain array (whose columns are x0, x1, ...).
        missing_columns = [
            scale.source_attribute
            for scale in self.scales_
            if scale.source_attribute not in mvc.attributes
        ]
        if missing_columns:
            raise ValueError(
                "The feature names should match those that were passed during "
                f"fit. Missing: {sorted(set(missing_columns))[:5]}."
            )

        binary_df = self._scale_to_dataframe(mvc)

        if self.missing == "unknown":
            if self.output == "context":
                return IncompleteContext.from_dataframe(binary_df)

            # 1.0 / 0.0 / NaN: scikit-learn's convention for missing values.
            # (Nullable-boolean frames break sklearn's numeric helpers.)
            values = binary_df.astype("Float64").to_numpy(dtype=float, na_value=np.nan)

            if self.output == "array":
                return values

            return pd.DataFrame(values, index=binary_df.index, columns=binary_df.columns)

        if self.output == "dataframe":
            return binary_df

        if self.output == "array":
            return binary_df.to_numpy(dtype=bool)

        if self.output == "sparse":
            return sparse.csr_matrix(binary_df.values, dtype=bool)

        return FormalContext.from_dataframe(binary_df)

    @staticmethod
    def _check_no_missing(mvc: ManyValuedContext, attribute: str) -> None:
        column = mvc.column(attribute)
        missing_objects = list(column.index[column.map(is_missing).astype(bool)])

        if missing_objects:
            shown = ", ".join(map(repr, missing_objects[:5]))
            more = "" if len(missing_objects) <= 5 else f" and {len(missing_objects) - 5} more"
            raise ValueError(
                f'Input contains NaN: attribute "{attribute}" has missing values for objects '
                f"{shown}{more}. Pass missing=\"false\" to treat them as having "
                "none of the scale's attributes, or missing=\"unknown\" to keep "
                "them as unknown entries (IncompleteContext)."
            )

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

    def _unknown_encoding(self, scale: Scale) -> dict[str, bool | None]:
        """
        Encode a missing value as three-valued attributes.

        An attribute that holds for every value in the scale's domain is
        known-yes, one that holds for none is known-no, and the rest are
        unknown. Without a finite domain everything is unknown.
        """
        attributes = scale.binary_attributes(self.input_context_)
        domain = scale.domain(self.input_context_)

        if not domain:
            return {attr: None for attr in attributes}

        encodings = [scale.encode_value(value, self.input_context_) for value in domain]

        result: dict[str, bool | None] = {}
        for attr in attributes:
            values = {bool(encoding[attr]) for encoding in encodings}
            result[attr] = values.pop() if len(values) == 1 else None
        return result

    def _scale_to_dataframe(self, mvc: ManyValuedContext) -> pd.DataFrame:
        # Column by column: each distinct value is encoded once.
        columns: dict[str, list] = {}

        for scale in self.scales_:
            attributes = scale.binary_attributes(self.input_context_)
            values = mvc.column(scale.source_attribute).tolist()
            missing = [is_missing(value) for value in values]
            if self.missing == "error" and any(missing):
                self._check_no_missing(mvc, scale.source_attribute)

            cache: dict = {}
            rows = []
            for value, is_gap in zip(values, missing):
                if is_gap:
                    key = ("missing",)
                else:
                    try:
                        key = (type(value), value)
                        hash(key)
                    except TypeError:
                        key = None  # unhashable: not cached
                encoded = cache.get(key) if key is not None else None
                if encoded is None:
                    encoded = self._encode(scale, value, is_gap)
                    if key is not None:
                        cache[key] = encoded
                rows.append(encoded)

            for attribute in attributes:
                if attribute in columns:
                    raise ValueError(
                        "Binary attribute name collision detected during scaling: "
                        f"{[attribute]}"
                    )
                columns[attribute] = [row.get(attribute, False) for row in rows]

        return pd.DataFrame(
            {attribute: columns[attribute] for attribute in self.scaled_attributes_},
            index=mvc.objects,
            columns=self.scaled_attributes_,
            dtype="boolean" if self.missing == "unknown" else bool,
        )

    def _encode(self, scale: Scale, value, is_gap: bool) -> dict[str, bool | None]:
        if is_gap:
            if self.missing == "false":
                return {attr: False for attr in scale.binary_attributes(self.input_context_)}
            return self._unknown_encoding(scale)
        # Vocabulary is always resolved from the fit-time context, even when
        # scaling transform-time data, so scales whose binary attributes are
        # derived from observed values (NominalScale, ContranominalScale)
        # stay fixed between fit and transform.
        try:
            return scale.encode_value(value, self.input_context_)
        except UnknownValueError:
            fallback = scale.encode_unknown(self.input_context_)
            if self.handle_unknown == "ignore" and fallback is not None:
                return fallback
            raise

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = self.missing != "error"
        # Many-valued data is routinely categorical (strings).
        tags.input_tags.string = True
        # Outputs are boolean membership indicators (or 0/1/NaN floats),
        # whatever the input dtype; tell sklearn not to expect preservation.
        tags.transformer_tags.preserves_dtype = []
        return tags

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
