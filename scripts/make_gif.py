"""Render the hero animation: gradient boosting fitting a curve and a decision boundary.

Left: 1-D regression, the ensemble prediction after ``m`` rounds of depth-2 trees.
Middle: 2-D classification (two moons), predicted probability of class 1.
Right: train / validation log-loss of the classifier as rounds are added.

Usage: ``python scripts/make_gif.py --out assets/boosting.gif``
"""

from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("NUMBA_NUM_THREADS", "4")
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "benchmarks"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from sklearn.datasets import make_moons  # noqa: E402

import style  # noqa: E402
from gbdt import GBDTClassifier, GBDTRegressor  # noqa: E402
from gbdt.metrics import logloss  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "assets", "boosting.gif"))
    ap.add_argument("--rounds", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    style.apply()
    rng = np.random.default_rng(args.seed)

    # ---- 1-D regression --------------------------------------------------
    n = 160
    x = np.sort(rng.uniform(-3, 3, n))
    f = lambda t: np.sin(2 * t) + 0.25 * t  # noqa: E731
    y = f(x) + rng.normal(0, 0.25, n)
    reg = GBDTRegressor(n_estimators=args.rounds, learning_rate=0.2, max_depth=2, reg_lambda=1.0,
                        min_child_weight=3.0, random_state=args.seed).fit(x[:, None], y)
    xg = np.linspace(-3.2, 3.2, 400)[:, None]
    reg_stages = [reg.predict(xg, n_iterations=m) for m in range(args.rounds + 1)]

    # ---- 2-D classification ---------------------------------------------
    Xc, yc = make_moons(500, noise=0.22, random_state=args.seed)
    ntr = 300
    clf = GBDTClassifier(n_estimators=args.rounds, learning_rate=0.2, max_depth=3, reg_lambda=1.0,
                         random_state=args.seed).fit(Xc[:ntr], yc[:ntr])
    gx, gy = np.meshgrid(np.linspace(-1.8, 2.8, 140), np.linspace(-1.4, 1.9, 120))
    grid = np.c_[gx.ravel(), gy.ravel()]
    clf_stages = [clf.predict_proba(grid, n_iterations=m)[:, 1].reshape(gx.shape) for m in range(args.rounds + 1)]
    tr_loss, va_loss = [], []
    for m in range(args.rounds + 1):
        tr_loss.append(logloss(yc[:ntr], clf.predict_proba(Xc[:ntr], n_iterations=m)[:, 1]))
        va_loss.append(logloss(yc[ntr:], clf.predict_proba(Xc[ntr:], n_iterations=m)[:, 1]))

    # ---- animation -------------------------------------------------------
    cmap = LinearSegmentedColormap.from_list("bo", ["#a9c9f2", "#f4f2ec", "#f5b79d"])
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.9), gridspec_kw={"width_ratios": [1.15, 1, 1], "wspace": 0.28})
    ax1, ax2, ax3 = axes
    fig.suptitle("Gradient boosting from scratch: second-order histogram trees fitting step by step",
                 fontsize=12, fontweight="bold", x=0.5, y=0.99)

    ax1.scatter(x, y, s=10, color=style.INK2, alpha=0.55, linewidths=0)
    ax1.plot(xg[:, 0], f(xg[:, 0]), color=style.INK2, lw=1.0, ls=":", label="true function")
    (line,) = ax1.plot(xg[:, 0], reg_stages[0], color="#2a78d6", lw=2.4, label="ensemble")
    ax1.set_xlim(-3.2, 3.2)
    ax1.set_ylim(-1.8, 1.8)
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.legend(loc="upper left")
    t1 = ax1.set_title("regression, 0 trees")

    mesh = ax2.pcolormesh(gx, gy, clf_stages[0], cmap=cmap, vmin=0, vmax=1, shading="auto")
    ax2.scatter(Xc[:ntr][yc[:ntr] == 0, 0], Xc[:ntr][yc[:ntr] == 0, 1], s=18, color="#2a78d6", edgecolor="#1a1a19", linewidth=0.5)
    ax2.scatter(Xc[:ntr][yc[:ntr] == 1, 0], Xc[:ntr][yc[:ntr] == 1, 1], s=18, color="#eb6834", edgecolor="#1a1a19", linewidth=0.5)
    ax2.set_xlabel("x1")
    ax2.set_ylabel("x2")
    ax2.grid(False)
    t2 = ax2.set_title("classification, P(class 1)")

    ms = np.arange(args.rounds + 1)
    ax3.plot(ms, tr_loss, color="#2a78d6", lw=0.8, alpha=0.25)
    ax3.plot(ms, va_loss, color="#eb6834", lw=0.8, alpha=0.25)
    (l_tr,) = ax3.plot([], [], color="#2a78d6", label="train")
    (l_va,) = ax3.plot([], [], color="#eb6834", label="validation")
    ax3.set_xlim(0, args.rounds)
    ax3.set_ylim(min(tr_loss) * 0.9, max(va_loss) * 1.05)
    ax3.set_xlabel("boosting round")
    ax3.set_ylabel("log-loss")
    ax3.legend(loc="upper right")
    ax3.set_title("loss")
    fig.subplots_adjust(top=0.84, bottom=0.15, left=0.05, right=0.985)

    frames = list(range(0, 13)) + list(range(14, args.rounds + 1, 2))
    frames += [args.rounds] * 8  # hold the last frame

    def update(m: int):
        line.set_ydata(reg_stages[m])
        t1.set_text(f"regression, {m} trees")
        mesh.set_array(clf_stages[m].ravel())
        t2.set_text(f"classification, {m} trees (depth 3)")
        l_tr.set_data(ms[: m + 1], tr_loss[: m + 1])
        l_va.set_data(ms[: m + 1], va_loss[: m + 1])
        return line, mesh, l_tr, l_va

    anim = FuncAnimation(fig, update, frames=frames, blit=False)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    anim.save(args.out, writer=PillowWriter(fps=8), dpi=100)
    print("wrote", args.out, f"{os.path.getsize(args.out) / 1e6:.2f} MB, {len(frames)} frames")


if __name__ == "__main__":
    main()
