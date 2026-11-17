"""Matched model factories for the benchmark: ours vs XGBoost vs LightGBM vs scikit-learn.

Hyper-parameters are chosen so that all libraries optimise the same objective with
the same capacity: learning rate 0.1, L2 = 1, at most ``2**depth`` leaves per tree,
at most 255 value bins per feature, ``N_THREADS`` threads (default 1).

* ``ours`` / ``xgboost`` grow level by level (depth-wise) up to ``max_depth``.
* ``ours-leafwise`` / ``lightgbm`` / ``sklearn`` grow best-leaf-first up to ``2**depth``
  leaves.
"""

from __future__ import annotations

import os

# Benchmarks run single-threaded by default (GBDT_BENCH_THREADS=1): on a shared machine
# multi-threaded wall-clock times are dominated by scheduling noise, whereas single-thread
# CPU time is reproducible. Set GBDT_BENCH_THREADS=4 to compare multi-threaded wall times.
N_THREADS = int(os.environ.get("GBDT_BENCH_THREADS", "1"))
os.environ["OMP_NUM_THREADS"] = str(N_THREADS)
os.environ["NUMBA_NUM_THREADS"] = str(N_THREADS)

import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402
import xgboost as xgb  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor  # noqa: E402

from gbdt import GBDTClassifier, GBDTRegressor  # noqa: E402

LR = 0.1
LIBS = ["ours", "ours-leafwise", "xgboost", "lightgbm", "sklearn"]
LABELS = {
    "ours": "ours (depth-wise)",
    "ours-leafwise": "ours (leaf-wise)",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "sklearn": "sklearn HistGB",
}


def make_model(lib: str, task: str, n_rounds: int, depth: int, seed: int = 0):
    """Return an unfitted scikit-learn-style estimator."""
    leaves = 2**depth
    clf = task != "regression"
    if lib == "ours":
        cls = GBDTClassifier if clf else GBDTRegressor
        return cls(
            n_estimators=n_rounds, learning_rate=LR, max_depth=depth, reg_lambda=1.0,
            min_child_weight=1.0, max_bins=256, n_threads=N_THREADS, random_state=seed,
        )
    if lib == "ours-leafwise":
        cls = GBDTClassifier if clf else GBDTRegressor
        return cls(
            n_estimators=n_rounds, learning_rate=LR, grow_policy="lossguide", max_depth=0,
            max_leaves=leaves, reg_lambda=1.0, min_child_weight=1.0, min_samples_leaf=20,
            max_bins=256, n_threads=N_THREADS, random_state=seed,
        )
    if lib == "xgboost":
        cls = xgb.XGBClassifier if clf else xgb.XGBRegressor
        return cls(
            n_estimators=n_rounds, learning_rate=LR, max_depth=depth, reg_lambda=1.0,
            min_child_weight=1.0, tree_method="hist", max_bin=255, n_jobs=N_THREADS,
            random_state=seed,
        )
    if lib == "lightgbm":
        cls = lgb.LGBMClassifier if clf else lgb.LGBMRegressor
        return cls(
            n_estimators=n_rounds, learning_rate=LR, num_leaves=leaves, max_depth=-1,
            reg_lambda=1.0, min_child_weight=1.0, min_child_samples=20, max_bin=255,
            n_jobs=N_THREADS, random_state=seed, verbose=-1,
        )
    if lib == "sklearn":
        cls = HistGradientBoostingClassifier if clf else HistGradientBoostingRegressor
        return cls(
            max_iter=n_rounds, learning_rate=LR, max_leaf_nodes=leaves, max_depth=None,
            l2_regularization=1.0, min_samples_leaf=20, max_bins=255, early_stopping=False,
            random_state=seed,
        )
    raise ValueError(lib)


def predict_for_score(model, lib: str, task: str, X: np.ndarray):
    """Predictions in the form ``sklearn.metrics`` expects (values or probabilities)."""
    if task == "regression":
        return model.predict(X)
    return model.predict_proba(X)
