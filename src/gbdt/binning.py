"""Quantile binning: map continuous features to small integer bins.

Histogram-based GBDTs (LightGBM, XGBoost ``tree_method="hist"``, sklearn's
HistGradientBoosting) never look at raw feature values while growing trees.
Each feature is discretised once, before training, into at most
``max_bins - 1`` bins whose edges are (approximate) quantiles of the training
distribution. One extra bin, the last one, is reserved for missing values.

Split finding then only has to scan ``max_bins`` histogram cells per feature
instead of ``n_samples`` sorted values, and the binned matrix fits in ``uint8``.
"""

from __future__ import annotations

import numpy as np


class BinMapper:
    """Learn per-feature bin edges and transform data into ``uint8`` bin codes.

    Bin ``b`` of feature ``j`` holds the values ``t[b-1] < x <= t[b]`` where
    ``t = thresholds_[j]``. Consequently the split "bin <= b goes left" is
    exactly the split ``x <= thresholds_[j][b]`` in the original feature space.

    Parameters
    ----------
    max_bins:
        Total number of bins including the missing-value bin (at most 256 so
        that codes fit in ``uint8``). Features with fewer distinct values than
        ``max_bins - 1`` get one bin per distinct value (exact splits).
    subsample:
        Number of rows used to estimate quantiles (``None`` = all rows).
    random_state:
        Seed for the quantile subsample.
    """

    def __init__(self, max_bins: int = 256, subsample: int | None = 200_000, random_state: int = 0):
        if not 3 <= max_bins <= 256:
            raise ValueError("max_bins must be in [3, 256]")
        self.max_bins = max_bins
        self.subsample = subsample
        self.random_state = random_state

    @property
    def missing_bin(self) -> int:
        """Index of the bin that stores missing values (always the last one)."""
        return self.max_bins - 1

    def fit(self, X: np.ndarray) -> "BinMapper":
        """Compute bin thresholds from (a subsample of) ``X``."""
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError("X must be 2-dimensional")
        n, d = X.shape
        if self.subsample is not None and n > self.subsample:
            rng = np.random.default_rng(self.random_state)
            X = X[rng.choice(n, self.subsample, replace=False)]

        n_value_bins = self.max_bins - 1
        thresholds: list[np.ndarray] = []
        for j in range(d):
            col = X[:, j]
            col = col[~np.isnan(col)]
            if col.size == 0:
                t = np.empty(0)
            else:
                uniq = np.unique(col)
                if uniq.size <= n_value_bins:
                    # one bin per distinct value; edges at midpoints
                    t = 0.5 * (uniq[:-1] + uniq[1:])
                else:
                    qs = np.linspace(0.0, 1.0, n_value_bins + 1)[1:-1]
                    t = np.unique(np.quantile(col, qs, method="midpoint"))
            thresholds.append(t.astype(np.float64))
        self.thresholds_ = thresholds
        # number of non-missing bins actually used by each feature
        self.n_bins_ = np.array([t.size + 1 for t in thresholds], dtype=np.int32)
        self.n_features_in_ = d
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Return bin codes as a **feature-major** ``(n_features, n_samples)`` uint8 array.

        Feature-major layout makes the histogram kernel (which loops over rows of
        one feature at a time) read contiguous memory.
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError(f"expected X with {self.n_features_in_} columns")
        n, d = X.shape
        out = np.empty((d, n), dtype=np.uint8)
        for j in range(d):
            col = X[:, j]
            codes = np.searchsorted(self.thresholds_[j], col, side="left")
            codes[np.isnan(col)] = self.missing_bin
            out[j] = codes
        return out

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def bin_threshold(self, feature: int, bin_index: int) -> float:
        """Real-valued threshold equivalent to the split ``bin <= bin_index``."""
        t = self.thresholds_[feature]
        return float(t[bin_index]) if bin_index < t.size else float("inf")
