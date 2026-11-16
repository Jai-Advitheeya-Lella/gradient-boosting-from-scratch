"""Split search and tree growth against brute-force NumPy computations."""

import numpy as np
import pytest

from gbdt import _kernels as K
from gbdt.binning import BinMapper
from gbdt.tree import TreeGrower, TreeParams


def _brute_force_best_split(x, g, h, lam, gamma):
    """Enumerate thresholds of one raw feature (no missing values)."""
    G, H = g.sum(), h.sum()
    best = (-np.inf, None)
    for t in np.unique(x)[:-1]:
        m = x <= t
        gl, hl = g[m].sum(), h[m].sum()
        gain = 0.5 * (gl**2 / (hl + lam) + (G - gl) ** 2 / (H - hl + lam) - G**2 / (H + lam)) - gamma
        if gain > best[0]:
            best = (gain, t)
    return best


@pytest.mark.parametrize("seed", range(5))
def test_best_split_matches_brute_force(seed):
    rng = np.random.default_rng(seed)
    n, d = 300, 4
    X = rng.integers(0, 30, size=(n, d)).astype(float)
    g = rng.normal(size=n) + X[:, seed % d] / 10
    h = rng.uniform(0.5, 2.0, size=n)
    lam, gamma = 1.3, 0.1
    bm = BinMapper().fit(X)
    XbT = bm.transform(X)
    rows = np.arange(n, dtype=np.int32)
    feats = np.arange(d)
    hist = K.build_histogram(XbT, rows, g, h, feats, bm.max_bins)
    gain, f, b, dl, gl, hl, cl = K.find_best_split(
        hist, feats, bm.n_bins_, bm.missing_bin, g.sum(), h.sum(), float(n), lam, 0.0, gamma, 0.0, 1
    )
    brute = [_brute_force_best_split(X[:, j], g, h, lam, gamma) for j in range(d)]
    j = int(np.argmax([bg for bg, _ in brute]))
    assert f == j
    assert gain == pytest.approx(brute[j][0], rel=1e-10)
    assert bm.thresholds_[f][b] == pytest.approx(brute[j][1] + 0.5)  # midpoint threshold
    m = X[:, f] <= bm.thresholds_[f][b]
    assert gl == pytest.approx(g[m].sum()) and cl == m.sum()


def test_missing_direction_is_learned():
    """Rows with NaN carry large positive gradients; they must be routed with the positives."""
    rng = np.random.default_rng(0)
    n = 1000
    x = rng.uniform(size=n)
    g = np.where(x > 0.5, 1.0, -1.0)
    miss = rng.random(n) < 0.2
    g[miss] = 1.0
    x[miss] = np.nan
    X = x[:, None]
    bm = BinMapper().fit(X)
    XbT = bm.transform(X)
    rows = np.arange(n, dtype=np.int32)
    hist = K.build_histogram(XbT, rows, g, np.ones(n), np.arange(1), bm.max_bins)
    _, f, b, dl, *_ = K.find_best_split(
        hist, np.arange(1), bm.n_bins_, bm.missing_bin, g.sum(), float(n), float(n), 1.0, 0.0, 0.0, 0.0, 1
    )
    assert bm.thresholds_[0][b] == pytest.approx(0.5, abs=0.02)
    assert dl is False or dl == 0  # NaNs go right together with x > 0.5


def test_parallel_and_serial_histograms_agree():
    rng = np.random.default_rng(0)
    n, d = 20_000, 7
    X = rng.normal(size=(n, d))
    X[rng.random((n, d)) < 0.05] = np.nan
    bm = BinMapper().fit(X)
    XbT = bm.transform(X)
    rows = np.sort(rng.choice(n, 12_000, replace=False)).astype(np.int32)
    g, h = rng.normal(size=n), rng.uniform(size=n)
    feats = np.arange(d)
    a = K.build_histogram_serial(XbT, rows, g, h, feats, 256)
    b = K.build_histogram_parallel(XbT, rows, g, h, feats, 256)
    np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12)
    # every feature's histogram sums to the node totals
    np.testing.assert_allclose(a[:, :, 0].sum(axis=1), g[rows].sum())
    np.testing.assert_allclose(a[:, :, 2].sum(axis=1), rows.shape[0])


def _grow(X, g, h, **kw):
    bm = BinMapper().fit(X)
    XbT = bm.transform(X)
    p = TreeParams(**kw)
    grower = TreeGrower(
        XbT, g, h, np.arange(X.shape[0], dtype=np.int32), np.arange(X.shape[1]), bm.n_bins_,
        bm.thresholds_, bm.missing_bin, p, np.random.default_rng(0),
    )
    return grower.grow(), XbT, bm


def test_leaf_values_are_newton_steps():
    """With lambda=0 and eta=1 every leaf value is -sum(g)/sum(h) over its rows."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(2000, 5))
    g = rng.normal(size=2000) + X[:, 0]
    h = rng.uniform(0.2, 1.0, size=2000)
    tree, XbT, bm = _grow(X, g, h, max_depth=3, reg_lambda=0.0, learning_rate=1.0)
    pred = tree.predict_binned(XbT, bm.missing_bin)
    for v in np.unique(pred):
        m = pred == v
        assert v == pytest.approx(-g[m].sum() / h[m].sum(), rel=1e-9)


def test_tree_shape_limits():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(3000, 6))
    g = rng.normal(size=3000) + np.sin(3 * X[:, 0]) + X[:, 1] * X[:, 2]
    h = np.ones(3000)
    t, *_ = _grow(X, g, h, max_depth=4)
    assert t.depth() == 4 and t.n_leaves == 16
    t, *_ = _grow(X, g, h, max_depth=0, max_leaves=11, grow_policy="lossguide")
    assert t.n_leaves == 11
    t, *_ = _grow(X, g, h, max_depth=6, min_samples_leaf=400)
    assert t.n_samples[t.is_leaf].min() >= 400
    # a huge gamma forbids every split
    t, *_ = _grow(X, g, h, max_depth=6, gamma=1e9)
    assert t.n_nodes == 1
