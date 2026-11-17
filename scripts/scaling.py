"""Single-thread CPU time vs number of rows on synthetic Friedman-1 data (28 features, 50 rounds).

Writes ``results/scaling.json`` and ``assets/scaling.png``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

for _k in ("GBDT_BENCH_THREADS", "OMP_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_k] = "1"  # before numpy / sklearn load their thread pools
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "benchmarks"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import NullFormatter  # noqa: E402
from sklearn.datasets import make_friedman1  # noqa: E402

import style  # noqa: E402
from libs import LABELS, LIBS, make_model  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sizes", nargs="+", type=int, default=[10_000, 30_000, 100_000, 300_000, 1_000_000])
    ap.add_argument("--rounds", type=int, default=50)
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()
    make_model("ours", "regression", 2, 6).fit(*make_friedman1(3000, n_features=28, random_state=0))
    make_model("ours-leafwise", "regression", 2, 6).fit(*make_friedman1(3000, n_features=28, random_state=0))
    out = {lib: [] for lib in LIBS}
    for n in args.sizes:
        X, y = make_friedman1(n, n_features=28, noise=1.0, random_state=0)
        reps = args.repeats if n <= 300_000 else 1
        for lib in LIBS:
            ts = []
            for _ in range(reps):
                m = make_model(lib, "regression", args.rounds, 6)
                t0 = time.process_time()
                m.fit(X, y)
                ts.append(time.process_time() - t0)
            out[lib].append({"n": n, "fit_time": min(ts)})
            print(f"n={n:>8,d} {lib:14s} {min(ts):7.2f}s", flush=True)
        del X, y
    with open(os.path.join(ROOT, "results", "scaling.json"), "w") as f:
        json.dump({"rounds": args.rounds, "depth": 6, "features": 28, "runs": out}, f, indent=1)

    style.apply()
    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    for lib in LIBS:
        pts = out[lib]
        ax.plot([p["n"] for p in pts], [p["fit_time"] for p in pts], color=style.COLORS[lib], ls=style.LINESTYLES[lib],
                marker=style.MARKERS[lib], ms=5, mec=style.SURFACE, mew=1, label=LABELS[lib])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("training rows")
    ax.set_ylabel(f"training CPU time, {args.rounds} rounds (s)")
    ax.set_title("Scaling with data size (28 features, depth 6, 1 thread)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "assets", "scaling.png"), bbox_inches="tight")


if __name__ == "__main__":
    main()
