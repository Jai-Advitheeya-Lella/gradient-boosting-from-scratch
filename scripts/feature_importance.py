"""Feature importances (gain / cover / weight) on California Housing, ours vs XGBoost.

Both libraries grow the same trees on this data to numerical precision (see
``tests/test_reference.py``), so their importances should agree; this script checks
that with a Spearman rank correlation and draws the figure.
Writes ``results/feature_importance.json`` and ``assets/feature_importance.png``.
"""

from __future__ import annotations

import json
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("NUMBA_NUM_THREADS", "4")
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "benchmarks"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

import datasets as D  # noqa: E402
import style  # noqa: E402
from libs import make_model  # noqa: E402

TYPES = [("total_gain", "total_gain", "Total gain"), ("total_cover", "total_cover", "Total cover (hessian mass)"), ("weight", "weight", "Split count")]


def main() -> None:
    ds = D.california(0)
    ours = make_model("ours", "regression", 200, 6).fit(ds.X_train, ds.y_train)
    ref = make_model("xgboost", "regression", 200, 6).fit(ds.X_train, ds.y_train)
    booster = ref.get_booster()
    booster.feature_names = [f"f{i}" for i in range(ds.X_train.shape[1])]
    names = ds.feature_names
    out: dict = {"features": names}
    style.apply()
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
    order = np.argsort(ours.get_importance("total_gain"))
    for ax, (ours_key, xgb_key, title) in zip(axes, TYPES):
        a = ours.get_importance(ours_key)
        sc = booster.get_score(importance_type=xgb_key)
        b = np.array([sc.get(f"f{i}", 0.0) for i in range(len(names))])
        # the weight of the paper's gain has a 1/2 that XGBoost omits; normalise both to sum 1
        an, bn = a / a.sum(), b / b.sum()
        rho = spearmanr(a, b).statistic
        out[ours_key] = {"ours": an.tolist(), "xgboost": bn.tolist(), "spearman": float(rho),
                         "max_abs_diff": float(np.abs(an - bn).max())}
        y = np.arange(len(names))
        ax.barh(y + 0.19, an[order], height=0.36, color=style.COLORS["ours"], label="ours")
        ax.barh(y - 0.19, bn[order], height=0.36, color=style.COLORS["xgboost"], label="XGBoost")
        ax.set_yticks(y)
        ax.set_yticklabels([names[i] for i in order])
        ax.set_title(f"{title}\nSpearman rho = {rho:.3f}")
        ax.set_xlabel("share of total")
        ax.grid(axis="y", visible=False)
        print(f"{ours_key}: spearman={rho:.4f} max |diff of shares|={out[ours_key]['max_abs_diff']:.4f}")
    axes[0].legend(loc="lower right")
    fig.suptitle("California Housing, 200 depth-6 trees: feature importances", fontsize=11, fontweight="bold", y=1.03)
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "assets", "feature_importance.png"), bbox_inches="tight")
    with open(os.path.join(ROOT, "results", "feature_importance.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
