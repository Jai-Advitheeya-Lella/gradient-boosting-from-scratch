"""Turn ``results/benchmark_*.json`` into figures (``assets/``) and markdown tables.

Usage: ``python scripts/plot_results.py``
"""

from __future__ import annotations

import json
import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "benchmarks"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import NullFormatter, ScalarFormatter  # noqa: E402

import style  # noqa: E402
from libs import LABELS, LIBS  # noqa: E402

DATASETS = ["california", "adult", "higgs", "covertype"]
TITLES = {
    "california": "California Housing (regression)",
    "adult": "Adult (binary)",
    "higgs": "Higgs (binary)",
    "covertype": "Covertype, 200k rows (7 classes)",
}
HEADLINE = {"california": "rmse", "adult": "accuracy", "higgs": "accuracy", "covertype": "accuracy"}
YLABEL = {"rmse": "test RMSE", "accuracy": "test accuracy", "logloss": "log-loss"}


def load() -> dict:
    out = {}
    for name in DATASETS:
        p = os.path.join(ROOT, "results", f"benchmark_{name}.json")
        if os.path.exists(p):
            with open(p) as f:
                out[name] = json.load(f)
    return out


def learning_curves(res: dict) -> None:
    names = [n for n in DATASETS if n in res]
    fig, axes = plt.subplots(1, len(names), figsize=(3.9 * len(names), 3.5))
    axes = np.atleast_1d(axes)
    for ax, name in zip(axes, names):
        r = res[name]
        for lib in LIBS:
            if lib not in r["curves"]:
                continue
            c = r["curves"][lib]
            v = np.asarray(c["val"])
            x = np.arange(1, len(v) + 1)
            ax.plot(x, v, color=style.COLORS[lib], ls=style.LINESTYLES[lib], lw=1.7, label=LABELS[lib])
            ax.plot(c["best_round"], v[c["best_round"] - 1], marker=style.MARKERS[lib], ms=5.5,
                    color=style.COLORS[lib], mec=style.SURFACE, mew=1.2, ls="none")
        lo = min(min(c["val"]) for c in r["curves"].values())
        hi = np.median([c["val"][min(9, len(c["val"]) - 1)] for c in r["curves"].values()])
        ax.set_ylim(lo - 0.02 * (hi - lo), lo + 1.6 * (hi - lo) if name != "california" else lo + 1.2 * (hi - lo))
        ax.set_title(TITLES[name])
        ax.set_xlabel("boosting round")
        ax.set_ylabel("validation " + ("RMSE" if r["info"]["task"] == "regression" else "log-loss"))
    axes[0].legend(loc="upper right")
    fig.suptitle("Validation loss per boosting round (marker = validation-optimal round)", fontsize=11, fontweight="bold", y=1.0)
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "assets", "learning_curves.png"), bbox_inches="tight")
    plt.close(fig)


def accuracy_vs_time(res: dict) -> None:
    names = [n for n in DATASETS if n in res]
    fig, axes = plt.subplots(1, len(names), figsize=(3.9 * len(names), 3.5))
    axes = np.atleast_1d(axes)
    for ax, name in zip(axes, names):
        r = res[name]
        key = HEADLINE[name]
        for lib in LIBS:
            if lib not in r["time_curve"]:
                continue
            pts = r["time_curve"][lib]
            ax.plot([p["cpu_time"] for p in pts], [p[key] for p in pts], color=style.COLORS[lib], ls=style.LINESTYLES[lib],
                    marker=style.MARKERS[lib], ms=4.5, mec=style.SURFACE, mew=1.0, label=LABELS[lib])
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_title(TITLES[name])
        ax.set_xlabel("training CPU time (s, log scale)")
        ax.set_ylabel(YLABEL[key] + (" (lower is better)" if key == "rmse" else " (higher is better)"))
        # zoom: ignore the very first (10-round) checkpoint, which is far from converged
        vals = [p[key] for lib in r["time_curve"] for p in r["time_curve"][lib][2:]]
        span = max(vals) - min(vals)
        ax.set_ylim(min(vals) - 0.25 * span, max(vals) + 0.25 * span)
    axes[0].legend(loc="upper right")
    fig.suptitle("Test quality vs training time at 10 / 25 / 50 / 100 / 200 / full-budget rounds (single thread)",
                 fontsize=11, fontweight="bold", y=1.0)
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "assets", "accuracy_vs_time.png"), bbox_inches="tight")
    plt.close(fig)


def fmt(v: np.ndarray, p: int) -> str:
    return f"{np.mean(v):.{p}f} ± {np.std(v):.{p}f}"


def tables(res: dict) -> str:
    lines = []
    for name in [n for n in DATASETS if n in res]:
        r = res[name]
        info, task = r["info"], r["info"]["task"]
        cfg = r["config"]
        n_seeds = len(next(iter(r["final"].values())))
        lines.append(
            f"**{TITLES[name]}**: {info['n_train']:,} train / {info['n_val']:,} val / {info['n_test']:,} test rows, "
            f"{info['n_features']} features, depth {cfg['depth']} (at most {2 ** cfg['depth']} leaves), "
            f"{cfg['rounds']} rounds, learning rate 0.1, mean ± std over {n_seeds} split seeds.\n"
        )
        if task == "regression":
            head = "| Library | RMSE | MAE | R² | Train CPU time (s) |\n|---|---|---|---|---|"
            keys, prec = ["rmse", "mae", "r2"], [4, 4, 4]
        elif task == "binary":
            head = "| Library | Accuracy | AUC | Log-loss | Train CPU time (s) |\n|---|---|---|---|---|"
            keys, prec = ["accuracy", "auc", "logloss"], [4, 4, 4]
        else:
            head = "| Library | Accuracy | Log-loss | Train CPU time (s) |\n|---|---|---|---|"
            keys, prec = ["accuracy", "logloss"], [4, 4]
        rows = [head]
        for lib in LIBS:
            if lib not in r["final"]:
                continue
            runs = r["final"][lib]
            cells = [fmt([x[k] for x in runs], p) for k, p in zip(keys, prec)]
            t = min(x["cpu_time"] for x in runs)
            rows.append(f"| {LABELS[lib]} | " + " | ".join(cells) + f" | {t:.2f} |")
        lines.append("\n".join(rows) + "\n")
    lines.append("Train CPU time: single thread, best of the runs above, binning included.\n")
    # validation-selected round
    lines.append("\n**Test quality at the validation-optimal round (split seed 0, early-stopping style)**\n")
    for name in [n for n in DATASETS if n in res]:
        r = res[name]
        task = r["info"]["task"]
        keys = ["rmse", "mae"] if task == "regression" else (["accuracy", "auc", "logloss"] if task == "binary" else ["accuracy", "logloss"])
        lines.append(f"\n{TITLES[name]}\n")
        lines.append("| Library | Best round | " + " | ".join(keys) + " |\n|---|---|" + "---|" * len(keys))
        for lib in LIBS:
            if lib not in r["curves"]:
                continue
            c = r["curves"][lib]
            lines.append(f"| {LABELS[lib]} | {c['best_round']} | " + " | ".join(f"{c['test_at_best'][k]:.4f}" for k in keys) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    style.apply()
    os.makedirs(os.path.join(ROOT, "assets"), exist_ok=True)
    res = load()
    learning_curves(res)
    accuracy_vs_time(res)
    md = tables(res)
    with open(os.path.join(ROOT, "results", "benchmark_tables.md"), "w") as f:
        f.write(md)
    print(md)


if __name__ == "__main__":
    main()
