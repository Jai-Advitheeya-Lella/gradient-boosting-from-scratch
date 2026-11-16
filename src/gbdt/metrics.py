"""Evaluation metrics used for logging and early stopping.

Each metric receives the targets and the *transformed* prediction (values for
regression, probabilities for classification) and returns a float.
``HIGHER_IS_BETTER`` tells early stopping which direction counts as progress.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata


def rmse(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(pred).reshape(-1) - y) ** 2)))


def mae(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(pred).reshape(-1) - y)))


def logloss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(np.asarray(p).reshape(-1), 1e-15, 1 - 1e-15)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def error(y: np.ndarray, p: np.ndarray) -> float:
    """Binary classification error at threshold 0.5."""
    return float(np.mean((np.asarray(p).reshape(-1) > 0.5) != (y > 0.5)))


def auc(y: np.ndarray, p: np.ndarray) -> float:
    """ROC AUC via the Mann-Whitney U statistic (ties get average ranks)."""
    p = np.asarray(p).reshape(-1)
    pos = y > 0.5
    n_pos = int(pos.sum())
    n_neg = y.shape[0] - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(p)
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def mlogloss(y: np.ndarray, P: np.ndarray) -> float:
    P = np.clip(P, 1e-15, 1.0)
    return float(-np.mean(np.log(P[np.arange(y.shape[0]), y.astype(np.int64)])))


def merror(y: np.ndarray, P: np.ndarray) -> float:
    return float(np.mean(P.argmax(axis=1) != y))


METRICS = {
    "rmse": rmse,
    "mae": mae,
    "logloss": logloss,
    "error": error,
    "auc": auc,
    "mlogloss": mlogloss,
    "merror": merror,
}
HIGHER_IS_BETTER = {"auc"}
