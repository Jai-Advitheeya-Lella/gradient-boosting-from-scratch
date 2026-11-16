import numpy as np

from gbdt.binning import BinMapper


def test_few_unique_values_get_exact_bins():
    X = np.array([[3.0], [1.0], [2.0], [1.0], [3.0]])
    bm = BinMapper(max_bins=256).fit(X)
    np.testing.assert_allclose(bm.thresholds_[0], [1.5, 2.5])
    assert bm.n_bins_[0] == 3
    np.testing.assert_array_equal(bm.transform(X)[0], [2, 0, 1, 0, 2])


def test_missing_values_go_to_last_bin():
    X = np.array([[0.0], [np.nan], [1.0]])
    bm = BinMapper(max_bins=16).fit(X)
    codes = bm.transform(X)[0]
    assert codes[1] == bm.missing_bin == 15
    assert codes[0] < codes[2] < 15


def test_quantile_bins_are_balanced_and_monotone():
    rng = np.random.default_rng(0)
    x = rng.lognormal(size=(100_000, 1))
    bm = BinMapper(max_bins=33, subsample=None).fit(x)
    codes = bm.transform(x)[0]
    counts = np.bincount(codes, minlength=32)[:32]
    assert bm.n_bins_[0] == 32
    assert counts.min() > 0.9 * counts.mean() and counts.max() < 1.1 * counts.mean()
    # binning is monotone in x
    order = np.argsort(x[:, 0])
    assert np.all(np.diff(codes[order].astype(int)) >= 0)


def test_bin_split_equals_threshold_split():
    """`code <= b` must be exactly `x <= thresholds[b]` for every b."""
    rng = np.random.default_rng(1)
    x = rng.normal(size=(5000, 1))
    bm = BinMapper(max_bins=64).fit(x)
    codes = bm.transform(x)[0]
    for b in range(bm.n_bins_[0] - 1):
        np.testing.assert_array_equal(codes <= b, x[:, 0] <= bm.thresholds_[0][b])
