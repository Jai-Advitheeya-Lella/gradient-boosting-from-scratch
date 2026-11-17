"""Ablations of the from-scratch booster: histogram bins, L2 / gamma, row & column sampling.

* (a) ``max_bins`` sweep on Higgs: test AUC and single-thread training CPU time.
* (b) lambda x gamma grid on California Housing (depth 8): test RMSE at the
  validation-optimal round.
* (c) sampling schemes on Higgs: uniform row subsampling, GOSS, column sampling.

Writes ``results/ablations.json`` and ``assets/ablations.png``.
"""

from __future__ import annotations

import json
import os
import sys
import time

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["NUMBA_NUM_THREADS"] = "1"  # single thread: CPU times are comparable on a shared machine
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "benchmarks"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import accuracy_score, roc_auc_score  # noqa: E402

import datasets as D  # noqa: E402
import style  # noqa: E402
from gbdt import GBDTClassifier, GBDTRegressor  # noqa: E402

BINS = [8, 16, 32, 64, 128, 256]
LAMBDAS = [0.0, 1.0, 10.0, 100.0, 1000.0]
GAMMAS = [0.0, 0.01, 0.1, 0.5, 2.0]
SAMPLING = {
    "full data": {},
    "uniform 50%": dict(subsample=0.5),
    "uniform 30%": dict(subsample=0.3),
    "GOSS (top 20% + 10% of rest)": dict(sampling="goss", goss_top_rate=0.2, goss_other_rate=0.1),
    "colsample_bytree 50%": dict(colsample_bytree=0.5),
    "colsample_bynode 50%": dict(colsample_bynode=0.5),
}


def main() -> None:
    out: dict = {"bins": [], "grid": {}, "sampling": {}}
    hi = D.higgs(0)
    GBDTClassifier(n_estimators=2, n_threads=1).fit(hi.X_train[:2000], hi.y_train[:2000])  # jit warm-up

    # (a) bins
    for b in BINS:
        times, aucs = [], []
        for rep in range(3):
            m = GBDTClassifier(n_estimators=150, learning_rate=0.1, max_depth=6, max_bins=b, n_threads=1)
            t0 = time.process_time()
            m.fit(hi.X_train, hi.y_train)
            times.append(time.process_time() - t0)
        aucs.append(roc_auc_score(hi.y_test, m.predict_proba(hi.X_test)[:, 1]))
        out["bins"].append({"max_bins": b, "auc": aucs[0], "fit_time": min(times)})
        print(f"bins={b:3d}  auc={aucs[0]:.4f}  time={min(times):.2f}s", flush=True)

    # (b) lambda x gamma
    ca = D.california(0)
    grid = np.zeros((len(LAMBDAS), len(GAMMAS)))
    for i, lam in enumerate(LAMBDAS):
        for j, gam in enumerate(GAMMAS):
            m = GBDTRegressor(n_estimators=600, learning_rate=0.1, max_depth=8, reg_lambda=lam, gamma=gam,
                              early_stopping_rounds=30, n_threads=1)
            m.fit(ca.X_train, ca.y_train, eval_set=[(ca.X_val, ca.y_val)])
            grid[i, j] = float(np.sqrt(np.mean((m.predict(ca.X_test) - ca.y_test) ** 2)))
            print(f"lambda={lam:7.1f} gamma={gam:5.1f} rmse={grid[i, j]:.4f} rounds={m.best_iteration_ + 1}", flush=True)
    out["grid"] = {"lambdas": LAMBDAS, "gammas": GAMMAS, "rmse": grid.tolist()}

    # (c) sampling on higgs: three seeds of the sampler
    for name, kw in SAMPLING.items():
        aucs, accs, times = [], [], []
        for seed in range(3):
            m = GBDTClassifier(n_estimators=150, learning_rate=0.1, max_depth=6, random_state=seed, n_threads=1, **kw)
            t0 = time.process_time()
            m.fit(hi.X_train, hi.y_train)
            times.append(time.process_time() - t0)
            p = m.predict_proba(hi.X_test)[:, 1]
            aucs.append(roc_auc_score(hi.y_test, p))
            accs.append(accuracy_score(hi.y_test, p > 0.5))
        out["sampling"][name] = {"auc": aucs, "accuracy": accs, "fit_time": times}
        print(f"{name:32s} auc={np.mean(aucs):.4f}±{np.std(aucs):.4f} time={min(times):.2f}s", flush=True)

    with open(os.path.join(ROOT, "results", "ablations.json"), "w") as f:
        json.dump(out, f, indent=1)

    style.apply()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.2, 3.7), gridspec_kw={"width_ratios": [1, 1.05]})
    bins = [r["max_bins"] for r in out["bins"]]
    a1.plot(bins, [r["auc"] for r in out["bins"]], color="#2a78d6", marker="o", ms=5, mec=style.SURFACE, mew=1)
    a1.set_xscale("log", base=2)
    a1.set_xticks(bins)
    a1.set_xticklabels([str(b) for b in bins])
    a1.set_xlabel("max_bins (incl. missing-value bin)")
    a1.set_ylabel("test AUC", color="#2a78d6")
    a1.set_title("Higgs: histogram resolution")
    for b, r in zip(bins, out["bins"]):
        a1.annotate(f"{r['fit_time']:.1f}s", (b, r["auc"]), textcoords="offset points", xytext=(0, 7), ha="center",
                    fontsize=8, color=style.INK2)
    a1.margins(y=0.15)

    im = a2.imshow(grid, cmap="Blues_r", aspect="auto", origin="lower")
    a2.set_xticks(range(len(GAMMAS)))
    a2.set_xticklabels([f"{g:g}" for g in GAMMAS])
    a2.set_yticks(range(len(LAMBDAS)))
    a2.set_yticklabels([f"{l:g}" for l in LAMBDAS])
    a2.set_xlabel(r"$\gamma$ (min. split gain)")
    a2.set_ylabel(r"$\lambda$ (L2 on leaf weights)")
    a2.set_title("California Housing: test RMSE, depth 8, early stopping")
    a2.grid(False)
    best = np.unravel_index(np.argmin(grid), grid.shape)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            a2.text(j, i, f"{grid[i, j]:.3f}", ha="center", va="center", fontsize=8,
                    color="white" if grid[i, j] < np.median(grid) else style.INK,
                    fontweight="bold" if (i, j) == best else "normal")
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "assets", "ablations.png"), bbox_inches="tight")


if __name__ == "__main__":
    main()
