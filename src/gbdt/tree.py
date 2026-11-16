"""Second-order regression trees grown on gradient/hessian histograms."""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field

import numpy as np

from . import _kernels as K


@dataclass
class TreeParams:
    """Hyper-parameters of a single tree (see :class:`gbdt.booster.GBDTRegressor`)."""

    max_depth: int = 6
    max_leaves: int = 0  # 0 = unlimited
    grow_policy: str = "depthwise"  # or "lossguide"
    reg_lambda: float = 1.0
    reg_alpha: float = 0.0
    gamma: float = 0.0
    min_child_weight: float = 1.0
    min_samples_leaf: int = 1
    min_split_gain: float = 0.0
    colsample_bynode: float = 1.0
    learning_rate: float = 0.1


@dataclass
class Tree:
    """A fitted tree stored as flat node arrays (node 0 is the root).

    For internal nodes ``left``/``right`` hold child indices; for leaves both are
    -1. ``value`` holds the shrunk leaf weight ``eta * w*`` (also filled for
    internal nodes, which is handy for path-based explanations). ``gain`` and
    ``cover`` (sum of hessians) feed the feature importances.
    """

    feature: np.ndarray
    bin_threshold: np.ndarray
    threshold: np.ndarray
    default_left: np.ndarray
    left: np.ndarray
    right: np.ndarray
    value: np.ndarray
    gain: np.ndarray
    cover: np.ndarray
    n_samples: np.ndarray

    @property
    def n_nodes(self) -> int:
        return int(self.left.shape[0])

    @property
    def n_leaves(self) -> int:
        return int((self.left == -1).sum())

    @property
    def is_leaf(self) -> np.ndarray:
        return self.left == -1

    def depth(self) -> int:
        depths = np.zeros(self.n_nodes, dtype=np.int64)
        for i in range(self.n_nodes):
            if self.left[i] != -1:
                depths[self.left[i]] = depths[self.right[i]] = depths[i] + 1
        return int(depths.max())

    def predict_binned(self, XbT: np.ndarray, missing_bin: int, out: np.ndarray | None = None) -> np.ndarray:
        """Add this tree's prediction for feature-major binned data to ``out``."""
        if out is None:
            out = np.zeros(XbT.shape[1])
        K.predict_tree_binned(
            XbT,
            self.feature,
            self.bin_threshold,
            self.default_left,
            self.left,
            self.right,
            self.value,
            missing_bin,
            out,
        )
        return out

    def to_text(self, feature_names: list[str] | None = None, precision: int = 4) -> str:
        """Human-readable dump of the tree (real-valued thresholds)."""
        lines: list[str] = []

        def name(f: int) -> str:
            return feature_names[f] if feature_names is not None else f"f{f}"

        def rec(node: int, depth: int) -> None:
            pad = "  " * depth
            if self.left[node] == -1:
                lines.append(f"{pad}leaf: {self.value[node]:.{precision}g} (cover={self.cover[node]:.4g})")
                return
            miss = "left" if self.default_left[node] else "right"
            lines.append(
                f"{pad}[{name(self.feature[node])} <= {self.threshold[node]:.{precision}g}] "
                f"gain={self.gain[node]:.4g} cover={self.cover[node]:.4g} missing->{miss}"
            )
            rec(int(self.left[node]), depth + 1)
            rec(int(self.right[node]), depth + 1)

        rec(0, 0)
        return "\n".join(lines)


@dataclass(order=True)
class _Candidate:
    priority: tuple
    node_id: int = field(compare=False)


class TreeGrower:
    """Grow one tree on binned data for given gradients and hessians.

    Two growth policies share one priority-queue loop:

    * ``depthwise`` (XGBoost default): nodes are expanded in breadth-first order,
      i.e. level by level, until ``max_depth``.
    * ``lossguide`` (LightGBM default): the leaf with the largest gain is
      expanded next, until ``max_leaves``.

    Histograms use the **subtraction trick**: only the child with fewer rows
    is histogrammed; its sibling's histogram is ``parent - child``.
    """

    def __init__(
        self,
        XbT: np.ndarray,
        g: np.ndarray,
        h: np.ndarray,
        rows: np.ndarray,
        features: np.ndarray,
        n_bins: np.ndarray,
        bin_thresholds: list[np.ndarray],
        missing_bin: int,
        params: TreeParams,
        rng: np.random.Generator,
    ):
        self.XbT = XbT
        self.g = np.ascontiguousarray(g, dtype=np.float64)
        self.h = np.ascontiguousarray(h, dtype=np.float64)
        self.rows = rows
        self.features = np.ascontiguousarray(features, dtype=np.int64)
        self.n_bins = n_bins
        self.bin_thresholds = bin_thresholds
        self.missing_bin = missing_bin
        self.n_bins_total = missing_bin + 1
        self.p = params
        self.rng = rng
        if params.grow_policy not in ("depthwise", "lossguide"):
            raise ValueError("grow_policy must be 'depthwise' or 'lossguide'")

    # ------------------------------------------------------------------ #
    def _node_features(self) -> np.ndarray:
        frac = self.p.colsample_bynode
        if frac >= 1.0:
            return self.features
        k = max(1, int(round(frac * self.features.shape[0])))
        return np.sort(self.rng.choice(self.features, size=k, replace=False))

    def _evaluate(self, nid: int) -> None:
        """Find the best split of node ``nid`` and queue it if it is worth taking."""
        p = self.p
        node = self._nodes[nid]
        if p.max_depth > 0 and node["depth"] >= p.max_depth:
            return
        if node["C"] < 2 * p.min_samples_leaf or node["H"] < 2 * p.min_child_weight:
            return
        gain, f, b, dl, gl, hl, cl = K.find_best_split(
            node["hist"],
            self._node_features(),
            self.n_bins,
            self.missing_bin,
            node["G"],
            node["H"],
            node["C"],
            p.reg_lambda,
            p.reg_alpha,
            p.gamma,
            p.min_child_weight,
            p.min_samples_leaf,
        )
        if f < 0 or not gain > max(p.min_split_gain, 1e-12):
            return
        node["split"] = (gain, int(f), int(b), bool(dl), gl, hl, cl)
        if p.grow_policy == "depthwise":
            prio = (node["depth"], nid)
        else:
            prio = (-gain, nid)
        heapq.heappush(self._heap, _Candidate(prio, nid))

    def _new_node(self, rows, hist, G, H, C, depth) -> int:
        nid = len(self._nodes)
        self._nodes.append(
            dict(rows=rows, hist=hist, G=G, H=H, C=C, depth=depth, split=None, left=-1, right=-1)
        )
        return nid

    # ------------------------------------------------------------------ #
    def grow(self) -> Tree:
        p = self.p
        self._nodes: list[dict] = []
        self._heap: list[_Candidate] = []

        rows = self.rows
        hist = K.build_histogram(self.XbT, rows, self.g, self.h, self.features, self.n_bins_total)
        G = float(self.g[rows].sum())
        H = float(self.h[rows].sum())
        root = self._new_node(rows, hist, G, H, float(rows.shape[0]), 0)
        self._evaluate(root)
        n_leaves = 1

        while self._heap and (p.max_leaves <= 0 or n_leaves < p.max_leaves):
            nid = heapq.heappop(self._heap).node_id
            node = self._nodes[nid]
            gain, f, b, dl, gl, hl, cl = node["split"]
            left_rows, right_rows = K.partition_rows(self.XbT, node["rows"], f, b, dl, self.missing_bin)
            # histogram subtraction: build the smaller child, derive the larger
            if left_rows.shape[0] <= right_rows.shape[0]:
                hl_ = K.build_histogram(self.XbT, left_rows, self.g, self.h, self.features, self.n_bins_total)
                hr_ = node["hist"] - hl_
            else:
                hr_ = K.build_histogram(self.XbT, right_rows, self.g, self.h, self.features, self.n_bins_total)
                hl_ = node["hist"] - hr_
            node["hist"] = None
            node["rows"] = None
            gr, hr, cr = node["G"] - gl, node["H"] - hl, node["C"] - cl
            li = self._new_node(left_rows, hl_, gl, hl, cl, node["depth"] + 1)
            ri = self._new_node(right_rows, hr_, gr, hr, cr, node["depth"] + 1)
            node["left"], node["right"] = li, ri
            n_leaves += 1
            self._evaluate(li)
            self._evaluate(ri)

        return self._finalize()

    def _finalize(self) -> Tree:
        p = self.p
        n = len(self._nodes)
        feature = np.zeros(n, dtype=np.int64)
        bin_thr = np.zeros(n, dtype=np.int64)
        thr = np.full(n, np.nan)
        dleft = np.zeros(n, dtype=np.bool_)
        left = np.full(n, -1, dtype=np.int64)
        right = np.full(n, -1, dtype=np.int64)
        value = np.zeros(n)
        gain = np.zeros(n)
        cover = np.zeros(n)
        nsamp = np.zeros(n, dtype=np.int64)
        for i, nd in enumerate(self._nodes):
            value[i] = p.learning_rate * K.leaf_weight(nd["G"], nd["H"], p.reg_lambda, p.reg_alpha)
            cover[i] = nd["H"]
            nsamp[i] = int(nd["C"])
            if nd["left"] != -1:
                g_, f, b, dl, *_ = nd["split"]
                feature[i], bin_thr[i], dleft[i], gain[i] = f, b, dl, g_
                t = self.bin_thresholds[f]
                thr[i] = t[b] if b < t.shape[0] else np.inf
                left[i], right[i] = nd["left"], nd["right"]
        self._nodes = []  # release histograms / row buffers
        return Tree(feature, bin_thr, thr, dleft, left, right, value, gain, cover, nsamp)
