import numpy as np
import pytest

from gbdt import CustomLoss, GBDTClassifier, GBDTRegressor
from gbdt.metrics import auc, rmse


@pytest.fixture(scope="module")
def reg_data():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(4000, 8))
    y = 2 * X[:, 0] + np.sin(3 * X[:, 1]) + X[:, 2] * X[:, 3] + 0.2 * rng.normal(size=4000)
    return X[:3000], y[:3000], X[3000:], y[3000:]


def test_boosting_reduces_error(reg_data):
    X, y, Xt, yt = reg_data
    m = GBDTRegressor(n_estimators=300, learning_rate=0.1, max_depth=4).fit(X, y, eval_set=[(Xt, yt)])
    curve = m.evals_result_["valid_0"]["rmse"]
    assert curve[-1] < 0.3 * np.std(yt)
    assert curve[-1] < curve[10] < curve[0]
    assert rmse(yt, m.predict(Xt)) == pytest.approx(curve[-1])


def test_early_stopping_uses_best_iteration(reg_data):
    X, y, Xt, yt = reg_data
    m = GBDTRegressor(n_estimators=5000, learning_rate=0.3, max_depth=6, early_stopping_rounds=20)
    m.fit(X, y, eval_set=[(Xt, yt)])
    curve = np.array(m.evals_result_["valid_0"]["rmse"])
    assert m.n_iter_ < 5000
    assert m.best_iteration_ == int(np.argmin(curve))
    assert m.n_iter_ == m.best_iteration_ + 1 + 20
    assert rmse(yt, m.predict(Xt)) == pytest.approx(curve.min())


def test_staged_predictions_match_predict(reg_data):
    X, y, Xt, _ = reg_data
    m = GBDTRegressor(n_estimators=30).fit(X, y)
    stages = list(m.staged_predict_raw(Xt, step=10))
    assert [s for s, _ in stages] == [10, 20, 30]
    np.testing.assert_allclose(stages[0][1][:, 0], m.predict(Xt, n_iterations=10))
    np.testing.assert_allclose(stages[-1][1][:, 0], m.predict(Xt))


def test_custom_loss_equals_builtin(reg_data):
    X, y, Xt, _ = reg_data
    custom = CustomLoss(lambda y, f: (f - y, np.ones_like(f)), init_fn=lambda y: y.mean())
    a = GBDTRegressor(n_estimators=20).fit(X, y).predict(Xt)
    b = GBDTRegressor(loss=custom, n_estimators=20).fit(X, y).predict(Xt)
    np.testing.assert_allclose(a, b, rtol=1e-12)


def test_integer_weights_equal_row_duplication():
    rng = np.random.default_rng(0)
    X = rng.integers(0, 15, size=(500, 3)).astype(float)
    y = X[:, 0] - X[:, 1] + rng.normal(size=500)
    w = rng.integers(1, 4, size=500)
    a = GBDTRegressor(n_estimators=10, max_depth=3).fit(X, y, sample_weight=w).predict(X)
    b = GBDTRegressor(n_estimators=10, max_depth=3).fit(np.repeat(X, w, 0), np.repeat(y, w)).predict(X)
    np.testing.assert_allclose(a, b, rtol=1e-10)


def test_sampling_is_seeded_and_still_learns(reg_data):
    X, y, Xt, yt = reg_data
    kw = dict(n_estimators=100, subsample=0.5, colsample_bytree=0.7, colsample_bynode=0.8)
    a = GBDTRegressor(random_state=1, **kw).fit(X, y).predict(Xt)
    b = GBDTRegressor(random_state=1, **kw).fit(X, y).predict(Xt)
    c = GBDTRegressor(random_state=2, **kw).fit(X, y).predict(Xt)
    np.testing.assert_array_equal(a, b)
    assert not np.allclose(a, c)
    assert rmse(yt, a) < 0.5 * np.std(yt)
    goss = GBDTRegressor(n_estimators=100, sampling="goss").fit(X, y).predict(Xt)
    assert rmse(yt, goss) < 0.5 * np.std(yt)


def test_missing_values_at_predict_time(reg_data):
    X, y, Xt, _ = reg_data
    m = GBDTRegressor(n_estimators=20).fit(X, y)
    Xn = Xt.copy()
    Xn[:, 0] = np.nan
    p = m.predict(Xn)
    assert np.all(np.isfinite(p))


def test_classifier_binary_and_multiclass_labels():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(3000, 5))
    logits = np.stack([X[:, 0], X[:, 1] - X[:, 2], -X[:, 0] + X[:, 3]], axis=1)
    y = np.array(["a", "b", "c"])[logits.argmax(1)]
    clf = GBDTClassifier(n_estimators=60, max_depth=3).fit(X[:2000], y[:2000], eval_set=[(X[2000:], y[2000:])])
    assert clf.n_trees_ == 60 * 3
    P = clf.predict_proba(X[2000:])
    np.testing.assert_allclose(P.sum(1), 1.0)
    assert np.mean(clf.predict(X[2000:]) == y[2000:]) > 0.85
    yb = (X[:, 0] + X[:, 1] > 0).astype(int)
    clf = GBDTClassifier(n_estimators=60, max_depth=3, eval_metric="auc", early_stopping_rounds=10)
    clf.fit(X[:2000], yb[:2000], eval_set=[(X[2000:], yb[2000:])])
    assert auc(yb[2000:], clf.predict_proba(X[2000:])[:, 1]) > 0.97


def test_importances_and_apply(reg_data):
    X, y, Xt, _ = reg_data
    m = GBDTRegressor(n_estimators=30, max_depth=3).fit(X, y)
    imp = m.feature_importances_
    assert imp.sum() == pytest.approx(1.0) and imp.argmax() == 0
    w = m.get_importance("weight")
    assert w.sum() == sum((t.left != -1).sum() for r in m.trees_ for t in r)
    leaves = m.apply(Xt)
    assert leaves.shape == (Xt.shape[0], 30)
    t0 = m.trees_[0][0]
    assert np.all(t0.left[leaves[:, 0]] == -1)
    assert "leaf" in t0.to_text()
