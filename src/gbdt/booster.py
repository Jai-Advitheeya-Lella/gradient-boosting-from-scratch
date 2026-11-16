"""The boosting loop and the scikit-learn-style estimators."""

from __future__ import annotations

import time
from typing import Iterator, Sequence

import numba
import numpy as np

from . import _kernels as K
from .binning import BinMapper
from .losses import Logistic, Loss, Softmax, get_loss
from .metrics import HIGHER_IS_BETTER, METRICS
from .tree import Tree, TreeGrower, TreeParams


class _BaseGBDT:
    """Gradient-boosted decision trees with second-order (Newton) leaf fitting.

    At round ``m`` the booster computes ``g_i = dl/dF`` and ``h_i = d2l/dF2`` at
    the current margins ``F_{m-1}(x_i)``, grows a tree that greedily maximises
    the regularised second-order gain, and updates
    ``F_m = F_{m-1} + eta * tree``. See README for the derivation.

    Parameters
    ----------
    loss:
        Name (``"squared_error"``, ``"pseudo_huber"``, ``"logistic"``,
        ``"softmax"``) or a :class:`gbdt.losses.Loss` instance.
    n_estimators:
        Maximum number of boosting rounds.
    learning_rate:
        Shrinkage ``eta`` applied to every leaf weight.
    max_depth, max_leaves, grow_policy:
        Tree shape. ``depthwise`` grows level by level (XGBoost); ``lossguide``
        always splits the best leaf next (LightGBM). ``max_depth=0`` /
        ``max_leaves=0`` mean unlimited.
    reg_lambda, reg_alpha:
        L2 / L1 penalty on leaf weights.
    gamma:
        Complexity penalty per leaf; a split is made only if its gain exceeds it.
    min_child_weight:
        Minimum hessian sum in each child.
    min_samples_leaf:
        Minimum number of rows in each child.
    subsample:
        Fraction of rows sampled (without replacement) per round.
    sampling:
        ``"uniform"`` or ``"goss"`` (gradient-based one-side sampling, LightGBM).
    goss_top_rate, goss_other_rate:
        GOSS keeps the ``top_rate`` fraction of rows with largest ``|g|`` and a
        random ``other_rate`` fraction of the rest, re-weighted by
        ``(1 - top_rate) / other_rate``.
    colsample_bytree, colsample_bynode:
        Fraction of features sampled per tree / per split.
    max_bins:
        Number of histogram bins per feature, including the missing-value bin.
    early_stopping_rounds:
        Stop when the metric on the **last** ``eval_set`` entry has not improved
        for this many rounds. Prediction then uses the best iteration.
    eval_metric:
        Metric name from :data:`gbdt.metrics.METRICS` (or ``"loss"``); defaults to
        the loss' natural metric.
    n_threads:
        Number of numba threads (``None`` = numba default).
    random_state:
        Seed for all sampling.
    verbose:
        Print evaluation results every ``verbose`` rounds (0 = silent).
    """

    _default_loss = "squared_error"

    def __init__(
        self,
        loss: str | Loss | None = None,
        *,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 6,
        max_leaves: int = 0,
        grow_policy: str = "depthwise",
        reg_lambda: float = 1.0,
        reg_alpha: float = 0.0,
        gamma: float = 0.0,
        min_child_weight: float = 1.0,
        min_samples_leaf: int = 1,
        min_split_gain: float = 0.0,
        subsample: float = 1.0,
        sampling: str = "uniform",
        goss_top_rate: float = 0.2,
        goss_other_rate: float = 0.1,
        colsample_bytree: float = 1.0,
        colsample_bynode: float = 1.0,
        max_bins: int = 256,
        early_stopping_rounds: int | None = None,
        eval_metric: str | None = None,
        n_threads: int | None = None,
        random_state: int = 0,
        verbose: int = 0,
    ):
        self.loss = loss if loss is not None else self._default_loss
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.max_leaves = max_leaves
        self.grow_policy = grow_policy
        self.reg_lambda = reg_lambda
        self.reg_alpha = reg_alpha
        self.gamma = gamma
        self.min_child_weight = min_child_weight
        self.min_samples_leaf = min_samples_leaf
        self.min_split_gain = min_split_gain
        self.subsample = subsample
        self.sampling = sampling
        self.goss_top_rate = goss_top_rate
        self.goss_other_rate = goss_other_rate
        self.colsample_bytree = colsample_bytree
        self.colsample_bynode = colsample_bynode
        self.max_bins = max_bins
        self.early_stopping_rounds = early_stopping_rounds
        self.eval_metric = eval_metric
        self.n_threads = n_threads
        self.random_state = random_state
        self.verbose = verbose

    # ------------------------------------------------------------------ #
    # Hooks overridden by the classifier
    # ------------------------------------------------------------------ #
    def _prepare_target(self, y: np.ndarray) -> np.ndarray:
        return np.asarray(y, dtype=np.float64)

    def _encode_target(self, y: np.ndarray) -> np.ndarray:
        return np.asarray(y, dtype=np.float64)

    def _make_loss(self) -> Loss:
        return get_loss(self.loss)

    # ------------------------------------------------------------------ #
    def _tree_params(self) -> TreeParams:
        return TreeParams(
            max_depth=self.max_depth,
            max_leaves=self.max_leaves,
            grow_policy=self.grow_policy,
            reg_lambda=self.reg_lambda,
            reg_alpha=self.reg_alpha,
            gamma=self.gamma,
            min_child_weight=self.min_child_weight,
            min_samples_leaf=self.min_samples_leaf,
            min_split_gain=self.min_split_gain,
            colsample_bynode=self.colsample_bynode,
            learning_rate=self.learning_rate,
        )

    def _sample_rows(self, g: np.ndarray, h: np.ndarray, n: int, rng: np.random.Generator):
        """Return (rows, g, h) for this round; GOSS re-weights g and h."""
        if self.sampling == "goss":
            a, b = self.goss_top_rate, self.goss_other_rate
            mag = np.abs(g).sum(axis=1)
            n_top = int(a * n)
            n_other = int(b * n)
            order = np.argpartition(-mag, n_top) if n_top < n else np.arange(n)
            top = order[:n_top]
            rest = order[n_top:]
            other = rng.choice(rest, size=min(n_other, rest.shape[0]), replace=False)
            factor = (1.0 - a) / b
            g = g.copy()
            h = h.copy()
            g[other] *= factor
            h[other] *= factor
            rows = np.sort(np.concatenate([top, other]))
            return rows.astype(np.int32), g, h
        if self.sampling != "uniform":
            raise ValueError("sampling must be 'uniform' or 'goss'")
        if self.subsample < 1.0:
            m = max(1, int(round(self.subsample * n)))
            rows = np.sort(rng.choice(n, size=m, replace=False)).astype(np.int32)
            return rows, g, h
        return self._all_rows, g, h

    def _metric_value(self, name: str, y: np.ndarray, raw: np.ndarray) -> float:
        if name == "loss":
            return self.loss_.loss(y, raw)
        pred = self.loss_.transform(raw)
        return METRICS[name](y, pred)

    # ------------------------------------------------------------------ #
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
        eval_set: Sequence[tuple[np.ndarray, np.ndarray]] | None = None,
    ):
        """Fit the booster.

        ``eval_set`` is a list of ``(X, y)`` pairs evaluated after every round;
        results are stored in ``evals_result_`` (also ``"time"``: seconds since
        the start of ``fit``, per round).
        """
        t_start = time.perf_counter()
        if self.n_threads is not None:
            numba.set_num_threads(min(int(self.n_threads), numba.config.NUMBA_NUM_THREADS))
        X = np.asarray(X, dtype=np.float64)
        y = self._prepare_target(y)
        n, d = X.shape
        rng = np.random.default_rng(self.random_state)
        w = None if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)

        self.bin_mapper_ = BinMapper(self.max_bins, random_state=self.random_state).fit(X)
        XbT = self.bin_mapper_.transform(X)
        mb = self.bin_mapper_.missing_bin
        self.n_features_in_ = d
        self.loss_ = self._make_loss()
        n_out = self.loss_.n_outputs
        metric = self.eval_metric or self.loss_.default_metric
        self.eval_metric_ = metric

        self.base_score_ = self.loss_.init_score(y, w)
        raw = np.tile(self.base_score_, (n, 1))
        evals = []
        for X_e, y_e in eval_set or []:
            X_e = np.asarray(X_e, dtype=np.float64)
            evals.append(
                (self.bin_mapper_.transform(X_e), self._encode_target(y_e), np.tile(self.base_score_, (X_e.shape[0], 1)))
            )
        self.evals_result_: dict[str, dict[str, list[float]]] = {
            f"valid_{i}": {metric: []} for i in range(len(evals))
        }
        self.evals_result_["time"] = {"seconds": []}

        params = self._tree_params()
        self._all_rows = np.arange(n, dtype=np.int32)
        self.trees_: list[list[Tree]] = []
        higher = metric in HIGHER_IS_BETTER
        best_score = -np.inf if higher else np.inf
        self.best_iteration_ = None
        rounds_since_best = 0

        for it in range(self.n_estimators):
            g, h = self.loss_.grad_hess(y, raw)
            if w is not None:
                g = g * w[:, None]
                h = h * w[:, None]
            rows, g, h = self._sample_rows(g, h, n, rng)
            if self.colsample_bytree < 1.0:
                k = max(1, int(round(self.colsample_bytree * d)))
                features = np.sort(rng.choice(d, size=k, replace=False))
            else:
                features = np.arange(d)
            round_trees = []
            for k in range(n_out):
                grower = TreeGrower(
                    XbT, g[:, k], h[:, k], rows, features, self.bin_mapper_.n_bins_,
                    self.bin_mapper_.thresholds_, mb, params, rng,
                )
                tree = grower.grow()
                round_trees.append(tree)
                col = np.ascontiguousarray(raw[:, k])
                tree.predict_binned(XbT, mb, col)
                raw[:, k] = col
                for XbT_e, _, raw_e in evals:
                    col = np.ascontiguousarray(raw_e[:, k])
                    tree.predict_binned(XbT_e, mb, col)
                    raw_e[:, k] = col
            self.trees_.append(round_trees)

            # ---- evaluation / early stopping (last eval set decides) ----
            msg = [f"[{it + 1}]"]
            score = None
            for i, (_, y_e, raw_e) in enumerate(evals):
                score = self._metric_value(metric, y_e, raw_e)
                self.evals_result_[f"valid_{i}"][metric].append(score)
                msg.append(f"valid_{i}-{metric}: {score:.6f}")
            self.evals_result_["time"]["seconds"].append(time.perf_counter() - t_start)
            if self.verbose and (it + 1) % self.verbose == 0:
                print("  ".join(msg))
            if score is not None:
                improved = score > best_score if higher else score < best_score
                if improved:
                    best_score = score
                    self.best_iteration_ = it
                    rounds_since_best = 0
                else:
                    rounds_since_best += 1
                if self.early_stopping_rounds is not None and rounds_since_best >= self.early_stopping_rounds:
                    break

        self.best_score_ = best_score if evals else None
        self.n_iter_ = len(self.trees_)
        self.fit_time_ = time.perf_counter() - t_start
        del self._all_rows
        return self

    # ------------------------------------------------------------------ #
    # Prediction
    # ------------------------------------------------------------------ #
    def _n_rounds(self, n_iterations: int | None) -> int:
        if n_iterations is not None:
            return min(n_iterations, len(self.trees_))
        if self.early_stopping_rounds is not None and self.best_iteration_ is not None:
            return self.best_iteration_ + 1
        return len(self.trees_)

    def _pack(self, n_rounds: int):
        trees = [(t, k) for r in self.trees_[:n_rounds] for k, t in enumerate(r)]
        offsets = np.zeros(len(trees) + 1, dtype=np.int64)
        for i, (t, _) in enumerate(trees):
            offsets[i + 1] = offsets[i] + t.n_nodes
        cat = lambda attr: (  # noqa: E731
            np.concatenate([getattr(t, attr) for t, _ in trees]) if trees else np.zeros(0)
        )
        out_col = np.array([k for _, k in trees], dtype=np.int64)
        return (
            cat("feature").astype(np.int64),
            cat("bin_threshold").astype(np.int64),
            cat("default_left").astype(np.bool_),
            cat("left").astype(np.int64),
            cat("right").astype(np.int64),
            cat("value").astype(np.float64),
            offsets,
            out_col,
        )

    def predict_raw(self, X: np.ndarray, n_iterations: int | None = None) -> np.ndarray:
        """Raw margins ``F(x)`` of shape ``(n_samples, n_outputs)``."""
        X = np.asarray(X, dtype=np.float64)
        Xb = np.ascontiguousarray(self.bin_mapper_.transform(X).T)
        out = np.tile(self.base_score_, (X.shape[0], 1))
        f, b, dl, l, r, v, off, oc = self._pack(self._n_rounds(n_iterations))
        if off.shape[0] > 1:
            K.predict_packed(Xb, f, b, dl, l, r, v, off, oc, self.bin_mapper_.missing_bin, out)
        return out

    def staged_predict_raw(self, X: np.ndarray, step: int = 1) -> Iterator[tuple[int, np.ndarray]]:
        """Yield ``(n_rounds, margins)`` after every ``step`` rounds."""
        X = np.asarray(X, dtype=np.float64)
        XbT = self.bin_mapper_.transform(X)
        mb = self.bin_mapper_.missing_bin
        raw = np.tile(self.base_score_, (X.shape[0], 1))
        for i, round_trees in enumerate(self.trees_):
            for k, t in enumerate(round_trees):
                col = np.ascontiguousarray(raw[:, k])
                t.predict_binned(XbT, mb, col)
                raw[:, k] = col
            if (i + 1) % step == 0 or i + 1 == len(self.trees_):
                yield i + 1, raw.copy()

    def apply(self, X: np.ndarray, n_iterations: int | None = None) -> np.ndarray:
        """Index of the leaf reached in every tree, shape ``(n_samples, n_trees)``."""
        X = np.asarray(X, dtype=np.float64)
        Xb = np.ascontiguousarray(self.bin_mapper_.transform(X).T)
        f, b, dl, l, r, _, off, _ = self._pack(self._n_rounds(n_iterations))
        out = np.zeros((X.shape[0], off.shape[0] - 1), dtype=np.int64)
        K.leaf_index_packed(Xb, f, b, dl, l, r, off, self.bin_mapper_.missing_bin, out)
        return out

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #
    def get_importance(self, importance_type: str = "total_gain", n_iterations: int | None = None) -> np.ndarray:
        """Per-feature importance.

        ``weight``: number of splits; ``total_gain`` / ``gain``: summed / mean
        split gain; ``total_cover`` / ``cover``: summed / mean hessian mass of
        the split nodes (same definitions as XGBoost).
        """
        d = self.n_features_in_
        weight = np.zeros(d)
        gain = np.zeros(d)
        cover = np.zeros(d)
        for round_trees in self.trees_[: self._n_rounds(n_iterations)]:
            for t in round_trees:
                internal = t.left != -1
                f = t.feature[internal]
                np.add.at(weight, f, 1.0)
                np.add.at(gain, f, t.gain[internal])
                np.add.at(cover, f, t.cover[internal])
        with np.errstate(invalid="ignore", divide="ignore"):
            table = {
                "weight": weight,
                "total_gain": gain,
                "gain": np.where(weight > 0, gain / weight, 0.0),
                "total_cover": cover,
                "cover": np.where(weight > 0, cover / weight, 0.0),
            }
        if importance_type not in table:
            raise ValueError(f"importance_type must be one of {sorted(table)}")
        return table[importance_type]

    @property
    def feature_importances_(self) -> np.ndarray:
        """Total gain per feature, normalised to sum to one."""
        imp = self.get_importance("total_gain")
        s = imp.sum()
        return imp / s if s > 0 else imp

    @property
    def n_trees_(self) -> int:
        return sum(len(r) for r in self.trees_)


class GBDTRegressor(_BaseGBDT):
    """Gradient-boosted trees for regression (default loss: squared error)."""

    _default_loss = "squared_error"

    def predict(self, X: np.ndarray, n_iterations: int | None = None) -> np.ndarray:
        return self.loss_.transform(self.predict_raw(X, n_iterations))


class GBDTClassifier(_BaseGBDT):
    """Gradient-boosted trees for binary (logistic) or multiclass (softmax) classification."""

    _default_loss = "auto"

    def _prepare_target(self, y):
        y = np.asarray(y)
        self.classes_, enc = np.unique(y, return_inverse=True)
        self.n_classes_ = self.classes_.shape[0]
        if self.n_classes_ < 2:
            raise ValueError("need at least two classes")
        return enc.astype(np.float64)

    def _encode_target(self, y):
        y = np.asarray(y)
        idx = np.searchsorted(self.classes_, y)
        if np.any(idx >= self.n_classes_) or np.any(self.classes_[np.minimum(idx, self.n_classes_ - 1)] != y):
            raise ValueError("eval_set contains labels not seen in training")
        return idx.astype(np.float64)

    def _make_loss(self) -> Loss:
        if isinstance(self.loss, Loss):
            return self.loss
        if self.loss in ("auto", "logistic", "softmax"):
            return Logistic() if self.n_classes_ == 2 else Softmax(self.n_classes_)
        raise ValueError(f"unknown classification loss {self.loss!r}")

    def predict_proba(self, X: np.ndarray, n_iterations: int | None = None) -> np.ndarray:
        raw = self.predict_raw(X, n_iterations)
        if self.n_classes_ == 2:
            p = self.loss_.transform(raw)
            return np.column_stack([1.0 - p, p])
        return self.loss_.transform(raw)

    def predict(self, X: np.ndarray, n_iterations: int | None = None) -> np.ndarray:
        return self.classes_[self.predict_proba(X, n_iterations).argmax(axis=1)]
