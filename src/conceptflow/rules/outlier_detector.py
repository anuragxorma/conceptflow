"""
Sklearn-compatible outlier detection with implications.
"""

from __future__ import annotations

import math
import numbers
import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, OutlierMixin
from sklearn.utils.validation import check_is_fitted

from conceptflow.algorithms._membership import contains_sets
from conceptflow.algorithms.enumeration import _INTERNAL_PREFIXES
from conceptflow.algorithms.implications import (
    _resolve_min_support,
    compute_canonical_basis,
    compute_luxenburger_basis,
)
from conceptflow.validation import check_binary_context_input
from conceptflow.validation.input import check_feature_names, remember_feature_names

# Fitting warns above this many rules.
_MANY_RULES = 100_000


class ImplicationOutlierDetector(OutlierMixin, BaseEstimator):
    """
    Detect objects that break rules holding for (almost) all other objects.

    ``fit`` mines rules from binary data: near-implications with confidence
    of at least ``min_confidence`` (the Luxenburger basis) and, optionally,
    exact implications (the canonical basis), each with enough support.
    An object breaks a rule ``premise -> conclusion`` if it has every
    premise attribute but lacks part of the conclusion.

    Breaking a rule is weighted by how surprising it is,
    ``-log2(1 - confidence)``, with Laplace-smoothed confidence
    ``(support + 1) / (premise support + 2)`` so exact rules get a finite
    weight that grows with their evidence. ``score_samples`` returns minus
    the total weight of the broken rules: 0 for objects breaking nothing,
    lower for more unusual objects (sklearn convention: lower = more
    abnormal). ``explain`` lists the broken rules, so every flag can be
    traced back to concrete attributes.

    Parameters
    ----------
    min_confidence:
        Smallest confidence of the near-implications used as rules.

    min_support:
        Rules need at least this many objects (int) or this fraction of
        objects (float) satisfying premise and conclusion.

    contamination:
        A float in (0, 0.5] (default 0.05): flag about this share of the
        training objects, the most unusual ones. The threshold is the
        corresponding quantile of the training scores; scores are discrete,
        so objects tied at the threshold are all flagged and the share can
        be a little larger. Objects breaking no rule are never flagged.
        ``"auto"``: flag every object that breaks any rule -- simple to
        interpret, but on real data this can be a large share, since
        near-implications have exceptions by definition.

    include_exact:
        Also use exact implications (confidence 1). An object breaking one
        of them is unlike every training object with that premise.

    Attributes
    ----------
    rules_:
        List of ``(premise, conclusion, support, premise_support)`` with
        attribute-index sets.

    weights_:
        Weight of each rule.

    offset_:
        ``decision_function = score_samples - offset_``; negative means
        outlier.
    """

    def __init__(
        self,
        min_confidence: float = 0.9,
        min_support=0.05,
        contamination=0.05,
        include_exact: bool = True,
    ):
        self.min_confidence = min_confidence
        self.min_support = min_support
        self.contamination = contamination
        self.include_exact = include_exact

    def _validate_params(self) -> None:
        if not isinstance(self.min_confidence, numbers.Real) or not (
            0.0 < self.min_confidence <= 1.0
        ):
            raise ValueError("min_confidence must lie in (0, 1].")
        if self.contamination != "auto" and not (
            isinstance(self.contamination, numbers.Real)
            and not isinstance(self.contamination, bool)
            and 0.0 < self.contamination <= 0.5
        ):
            raise ValueError('contamination must be "auto" or a float in (0, 0.5].')

    def fit(self, X, y=None):
        """Mine the rules from binary data X."""
        self._validate_params()
        context = check_binary_context_input(X)
        remember_feature_names(self, X)

        self.n_features_in_ = context.n_attributes
        self.feature_names_in_ = context.attributes
        threshold = max(1, _resolve_min_support(self.min_support, context.n_objects))

        rules = []
        if self.min_confidence < 1.0:
            for rule in compute_luxenburger_basis(
                context, min_confidence=self.min_confidence, min_support=threshold
            ):
                rules.append(
                    (rule.premise, rule.conclusion, rule.support, rule.premise_support)
                )

        if self.include_exact:
            for imp in compute_canonical_basis(context):
                support = len(context.attribute_derivation(imp.premise))
                if support >= threshold and imp.conclusion:
                    rules.append((imp.premise, imp.conclusion, support, support))

        self.rules_ = rules
        if len(rules) > _MANY_RULES:
            warnings.warn(
                f"{len(rules):,} rules were mined, which makes fitting and scoring "
                "slow. Raise min_support or min_confidence to keep fewer, "
                "stronger rules.",
                UserWarning,
                skip_file_prefixes=_INTERNAL_PREFIXES,
            )
        self.weights_ = np.array(
            [
                -math.log2(1.0 - (support + 1) / (premise_support + 2))
                for _, _, support, premise_support in rules
            ],
            dtype=float,
        )

        if self.contamination == "auto":
            # Any broken rule makes an object an outlier.
            self.offset_ = -1e-12
        else:
            scores = self.score_samples(X)
            cutoff = float(np.percentile(scores, 100.0 * self.contamination, method="lower"))
            # Scores are discrete, so many objects can tie at the cutoff; they
            # are as unusual as the cutoff and are flagged too. Objects that
            # break no rule (score 0) are never flagged.
            self.offset_ = float(np.nextafter(cutoff, np.inf)) if cutoff < 0 else -1e-12

        return self

    def _violations(self, X) -> np.ndarray:
        check_is_fitted(self, "rules_")
        check_feature_names(self, X)
        context = check_binary_context_input(X)
        if context.n_attributes != self.n_features_in_:
            raise ValueError(
                f"X has {context.n_attributes} features, but "
                f"{self.__class__.__name__} is expecting {self.n_features_in_} "
                "features as input"
            )
        if not self.rules_:
            return np.zeros((context.n_objects, 0), dtype=bool)

        has_premise = contains_sets(context.incidence, [r[0] for r in self.rules_])
        has_rule = contains_sets(context.incidence, [r[0] | r[1] for r in self.rules_])
        return has_premise & ~has_rule

    def score_samples(self, X) -> np.ndarray:
        """Minus the total weight of the rules each object breaks (0 = none)."""
        violations = self._violations(X)
        if violations.shape[1] == 0:
            return np.zeros(violations.shape[0])
        return -(violations.astype(float) @ self.weights_)

    def decision_function(self, X) -> np.ndarray:
        """``score_samples(X) - offset_``; negative values are outliers."""
        return self.score_samples(X) - self.offset_

    def predict(self, X) -> np.ndarray:
        """-1 for outliers, 1 for inliers."""
        return np.where(self.decision_function(X) < 0, -1, 1)

    def explain(self, X) -> pd.DataFrame:
        """
        The broken rules of every object in X.

        One row per (object, broken rule) with the object's position,
        premise, conclusion, the missing conclusion attributes, the rule's
        confidence and weight.
        """
        violations = self._violations(X)
        context = check_binary_context_input(X)
        names = self.feature_names_in_
        rows = []

        for g, k in zip(*np.nonzero(violations)):
            premise, conclusion, support, premise_support = self.rules_[k]
            missing = sorted(m for m in conclusion if not context.incidence[g, m])
            rows.append(
                {
                    "object": int(g),
                    "premise": tuple(names[m] for m in sorted(premise)),
                    "conclusion": tuple(names[m] for m in sorted(conclusion)),
                    "missing": tuple(names[m] for m in missing),
                    "confidence": support / premise_support,
                    "weight": float(self.weights_[k]),
                }
            )

        return pd.DataFrame(
            rows, columns=["object", "premise", "conclusion", "missing", "confidence", "weight"]
        )

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.sparse = True
        return tags
