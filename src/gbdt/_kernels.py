"""Numba kernels for the hot loops: histograms, split search, partitioning, prediction.

Everything here operates on the binned, feature-major matrix ``XbT`` of shape
``(n_features, n_samples)`` (dtype ``uint8``) produced by :class:`gbdt.binning.BinMapper`.

Histogram layout: ``hist[f, b, 0]`` = sum of gradients, ``hist[f, b, 1]`` = sum of
hessians and ``hist[f, b, 2]`` = number of rows that fall in bin ``b`` of feature ``f``.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange

# Below this many rows a node's histogram is built single-threaded: thread
# start-up costs more than the work itself.
PARALLEL_MIN_ROWS = 4096


# --------------------------------------------------------------------------- #
# Histograms
# --------------------------------------------------------------------------- #
@njit(cache=True, nogil=True)
def _gather(rows, g, h):
    n = rows.shape[0]
    gs = np.empty(n, dtype=np.float64)
    hs = np.empty(n, dtype=np.float64)
    for i in range(n):
        r = rows[i]
        gs[i] = g[r]
        hs[i] = h[r]
    return gs, hs


@njit(cache=True, nogil=True)
def build_histogram_serial(XbT, rows, g, h, features, n_bins_total):
    """Gradient/hessian/count histogram of ``rows`` for the given ``features``."""
    d = XbT.shape[0]
    hist = np.zeros((d, n_bins_total, 3), dtype=np.float64)
    gs, hs = _gather(rows, g, h)  # "ordered gradients": sequential reads below
    n = rows.shape[0]
    for fi in range(features.shape[0]):
        f = features[fi]
        col = XbT[f]
        hf = hist[f]
        for i in range(n):
            b = col[rows[i]]
            hf[b, 0] += gs[i]
            hf[b, 1] += hs[i]
            hf[b, 2] += 1.0
    return hist


@njit(cache=True, nogil=True, parallel=True)
def build_histogram_parallel(XbT, rows, g, h, features, n_bins_total):
    """Same as :func:`build_histogram_serial`, parallelised over features."""
    d = XbT.shape[0]
    hist = np.zeros((d, n_bins_total, 3), dtype=np.float64)
    gs, hs = _gather(rows, g, h)
    n = rows.shape[0]
    for fi in prange(features.shape[0]):
        f = features[fi]
        col = XbT[f]
        hf = hist[f]
        for i in range(n):
            b = col[rows[i]]
            hf[b, 0] += gs[i]
            hf[b, 1] += hs[i]
            hf[b, 2] += 1.0
    return hist


def build_histogram(XbT, rows, g, h, features, n_bins_total):
    if rows.shape[0] >= PARALLEL_MIN_ROWS:
        return build_histogram_parallel(XbT, rows, g, h, features, n_bins_total)
    return build_histogram_serial(XbT, rows, g, h, features, n_bins_total)


# --------------------------------------------------------------------------- #
# Split finding
# --------------------------------------------------------------------------- #
@njit(cache=True, inline="always")
def _l1_threshold(G, alpha):
    if G > alpha:
        return G - alpha
    if G < -alpha:
        return G + alpha
    return 0.0


@njit(cache=True, inline="always")
def _score(G, H, lam, alpha):
    """Structure score of a leaf, ``T_alpha(G)^2 / (H + lambda)`` (Chen & Guestrin eq. 6)."""
    t = _l1_threshold(G, alpha)
    return t * t / (H + lam)


@njit(cache=True, inline="always")
def leaf_weight(G, H, lam, alpha):
    """Optimal leaf weight ``w* = -T_alpha(G) / (H + lambda)`` (Chen & Guestrin eq. 5)."""
    return -_l1_threshold(G, alpha) / (H + lam)


@njit(cache=True, nogil=True)
def find_best_split(
    hist,
    features,
    n_bins,
    missing_bin,
    G,
    H,
    C,
    lam,
    alpha,
    gamma,
    min_child_weight,
    min_samples_leaf,
):
    """Scan every candidate threshold of every feature; return the best split.

    Gain of splitting a node into L and R (Chen & Guestrin 2016, eq. 7)::

        gain = 1/2 * [S(G_L, H_L) + S(G_R, H_R) - S(G, H)] - gamma

    For a feature with missing values both default directions are evaluated
    (sparsity-aware split finding, Algorithm 3 of the XGBoost paper), plus the
    split that isolates the missing values from all non-missing ones.

    Returns ``(gain, feature, bin, default_left, G_L, H_L, C_L)``; ``feature``
    is -1 if no split satisfies the constraints.
    """
    parent = _score(G, H, lam, alpha)
    best_gain = -np.inf
    best_f = -1
    best_b = -1
    best_dl = False
    best_gl = 0.0
    best_hl = 0.0
    best_cl = 0.0

    for fi in range(features.shape[0]):
        f = features[fi]
        nb = n_bins[f]
        gm = hist[f, missing_bin, 0]
        hm = hist[f, missing_bin, 1]
        cm = hist[f, missing_bin, 2]
        has_missing = cm > 0.0
        gl = 0.0
        hl = 0.0
        cl = 0.0
        for b in range(nb - 1):  # threshold between bin b and b + 1
            gl += hist[f, b, 0]
            hl += hist[f, b, 1]
            cl += hist[f, b, 2]
            # (1) missing values go right
            gr = G - gl
            hr = H - hl
            cr = C - cl
            if cl >= min_samples_leaf and cr >= min_samples_leaf and hl >= min_child_weight and hr >= min_child_weight:
                gain = 0.5 * (_score(gl, hl, lam, alpha) + _score(gr, hr, lam, alpha) - parent) - gamma
                if gain > best_gain:
                    best_gain = gain
                    best_f = f
                    best_b = b
                    # with no missing values in training, send unseen NaNs to the larger child
                    best_dl = (cl >= cr) if not has_missing else False
                    best_gl = gl
                    best_hl = hl
                    best_cl = cl
            # (2) missing values go left
            if has_missing:
                gl2 = gl + gm
                hl2 = hl + hm
                cl2 = cl + cm
                gr2 = G - gl2
                hr2 = H - hl2
                cr2 = C - cl2
                if (
                    cl2 >= min_samples_leaf
                    and cr2 >= min_samples_leaf
                    and hl2 >= min_child_weight
                    and hr2 >= min_child_weight
                ):
                    gain = 0.5 * (_score(gl2, hl2, lam, alpha) + _score(gr2, hr2, lam, alpha) - parent) - gamma
                    if gain > best_gain:
                        best_gain = gain
                        best_f = f
                        best_b = b
                        best_dl = True
                        best_gl = gl2
                        best_hl = hl2
                        best_cl = cl2
        if has_missing:
            # (3) all non-missing values left, missing values right
            gl = G - gm
            hl = H - hm
            cl = C - cm
            if cl >= min_samples_leaf and cm >= min_samples_leaf and hl >= min_child_weight and hm >= min_child_weight:
                gain = 0.5 * (_score(gl, hl, lam, alpha) + _score(gm, hm, lam, alpha) - parent) - gamma
                if gain > best_gain:
                    best_gain = gain
                    best_f = f
                    best_b = nb - 1
                    best_dl = False
                    best_gl = gl
                    best_hl = hl
                    best_cl = cl
    return best_gain, best_f, best_b, best_dl, best_gl, best_hl, best_cl


# --------------------------------------------------------------------------- #
# Row partitioning
# --------------------------------------------------------------------------- #
@njit(cache=True, nogil=True)
def partition_rows(XbT, rows, feature, bin_threshold, default_left, missing_bin):
    """Split ``rows`` into (left, right) according to a binned split."""
    col = XbT[feature]
    n = rows.shape[0]
    go_left = np.empty(n, dtype=np.bool_)
    n_left = 0
    for i in range(n):
        v = col[rows[i]]
        if v == missing_bin:
            gl = default_left
        else:
            gl = v <= bin_threshold
        go_left[i] = gl
        n_left += gl
    left = np.empty(n_left, dtype=rows.dtype)
    right = np.empty(n - n_left, dtype=rows.dtype)
    il = 0
    ir = 0
    for i in range(n):
        if go_left[i]:
            left[il] = rows[i]
            il += 1
        else:
            right[ir] = rows[i]
            ir += 1
    return left, right


# --------------------------------------------------------------------------- #
# Prediction
# --------------------------------------------------------------------------- #
@njit(cache=True, nogil=True, parallel=True)
def predict_tree_binned(XbT, feature, bin_threshold, default_left, left, right, value, missing_bin, out):
    """``out[i] += tree(x_i)`` for one tree on feature-major binned data."""
    n = XbT.shape[1]
    for i in prange(n):
        node = 0
        while left[node] != -1:
            v = XbT[feature[node], i]
            if v == missing_bin:
                go_left = default_left[node]
            else:
                go_left = v <= bin_threshold[node]
            node = left[node] if go_left else right[node]
        out[i] += value[node]


@njit(cache=True, nogil=True, parallel=True)
def predict_packed(Xb, feature, bin_threshold, default_left, left, right, value, offsets, tree_output, missing_bin, out):
    """Sum the predictions of many trees stored back-to-back in flat arrays.

    ``Xb`` is **row-major** ``(n_samples, n_features)`` so that one row's codes stay
    in cache while all trees are traversed. Tree ``t`` occupies nodes
    ``offsets[t]:offsets[t+1]`` and adds to output column ``tree_output[t]``.
    """
    n = Xb.shape[0]
    n_trees = offsets.shape[0] - 1
    for i in prange(n):
        for t in range(n_trees):
            base = offsets[t]
            node = 0
            while left[base + node] != -1:
                k = base + node
                v = Xb[i, feature[k]]
                if v == missing_bin:
                    go_left = default_left[k]
                else:
                    go_left = v <= bin_threshold[k]
                node = left[k] if go_left else right[k]
            out[i, tree_output[t]] += value[base + node]


@njit(cache=True, nogil=True, parallel=True)
def leaf_index_packed(Xb, feature, bin_threshold, default_left, left, right, offsets, missing_bin, out):
    """Leaf (node) index reached by every row in every tree."""
    n = Xb.shape[0]
    n_trees = offsets.shape[0] - 1
    for i in prange(n):
        for t in range(n_trees):
            base = offsets[t]
            node = 0
            while left[base + node] != -1:
                k = base + node
                v = Xb[i, feature[k]]
                if v == missing_bin:
                    go_left = default_left[k]
                else:
                    go_left = v <= bin_threshold[k]
                node = left[k] if go_left else right[k]
            out[i, t] = node
