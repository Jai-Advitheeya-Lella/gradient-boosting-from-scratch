"""Bit-for-bit-level agreement with XGBoost and scikit-learn on exactly-binnable data.

When every feature has fewer distinct values than ``max_bins`` the quantile
sketches of all libraries are exact, so the trees must be identical. Remaining
differences are floating point: XGBoost accumulates in float32 (~1e-6).
"""

import numpy as np
import pytest

from gbdt import GBDTClassifier, GBDTRegressor

xgb = pytest.importorskip("xgboost")
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor  # noqa: E402


def _data(missing: bool, n: int = 1500, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 20, size=(n, 6)).astype(float)
    y = 0.5 * X[:, 0] + np.sin(X[:, 1]) + X[:, 2] * X[:, 3] / 40 + rng.normal(size=n) * 0.3
    if missing:
        X[rng.random(X.shape) < 0.1] = np.nan
    return X, y


@pytest.mark.parametrize("missing", [False, True])
@pytest.mark.parametrize("gamma", [0.0, 0.5])
def test_matches_xgboost_regression(missing, gamma):
    X, y = _data(missing)
    ours = GBDTRegressor(n_estimators=20, learning_rate=0.3, max_depth=4, reg_lambda=1.0, gamma=gamma).fit(X, y)
    # XGBoost's loss_chg omits the 1/2 of the paper's gain, so its gamma is 2x ours
    ref = xgb.XGBRegressor(
        n_estimators=20, learning_rate=0.3, max_depth=4, reg_lambda=1.0, gamma=2 * gamma,
        tree_method="hist", max_bin=256, base_score=float(y.mean()), n_jobs=2,
    ).fit(X, y)
    np.testing.assert_allclose(ours.predict(X), ref.predict(X), atol=2e-5)


@pytest.mark.parametrize("missing", [False, True])
def test_matches_xgboost_logistic(missing):
    X, y = _data(missing)
    yb = (y > y.mean()).astype(int)
    ours = GBDTClassifier(n_estimators=20, learning_rate=0.3, max_depth=4, reg_alpha=0.2).fit(X, yb)
    ref = xgb.XGBClassifier(
        n_estimators=20, learning_rate=0.3, max_depth=4, reg_alpha=0.2, tree_method="hist",
        base_score=float(yb.mean()), n_jobs=2,
    ).fit(X, yb)
    np.testing.assert_allclose(ours.predict_proba(X)[:, 1], ref.predict_proba(X)[:, 1], atol=1e-5)


@pytest.mark.parametrize("missing", [False, True])
def test_matches_sklearn_leafwise_regression(missing):
    X, y = _data(missing)
    ours = GBDTRegressor(
        n_estimators=20, learning_rate=0.3, grow_policy="lossguide", max_depth=0, max_leaves=15,
        min_samples_leaf=20, min_child_weight=1e-3, reg_lambda=0.5,
    ).fit(X, y)
    ref = HistGradientBoostingRegressor(
        max_iter=20, learning_rate=0.3, max_leaf_nodes=15, min_samples_leaf=20, l2_regularization=0.5,
        early_stopping=False,
    ).fit(X, y)
    np.testing.assert_allclose(ours.predict(X), ref.predict(X), atol=1e-7)


@pytest.mark.parametrize("missing", [False, True])
def test_matches_sklearn_softmax(missing):
    X, y = _data(missing)
    yc = np.digitize(y, np.quantile(y, [0.25, 0.5, 0.75]))
    # continuous sample weights break the exact gain ties that discrete
    # first-round gradients produce (tie-breaking order is implementation-defined)
    w = np.random.default_rng(1).uniform(0.5, 1.5, size=y.shape[0])
    ours = GBDTClassifier(
        n_estimators=10, learning_rate=0.3, grow_policy="lossguide", max_depth=0, max_leaves=15,
        min_samples_leaf=20, min_child_weight=1e-3, reg_lambda=0.0,
    ).fit(X, yc, sample_weight=w)
    ref = HistGradientBoostingClassifier(
        max_iter=10, learning_rate=0.3, max_leaf_nodes=15, min_samples_leaf=20, early_stopping=False
    ).fit(X, yc, sample_weight=w)
    np.testing.assert_allclose(ours.predict_proba(X), ref.predict_proba(X), atol=1e-7)
