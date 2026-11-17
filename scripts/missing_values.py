"""Native missing-value handling vs imputation on California Housing.

Entries of the feature matrices (train, validation and test) are deleted completely
at random with probability ``p``. Compared: learned default directions (ours, XGBoost,
LightGBM) against median imputation followed by our model. Writes
``results/missing_values.json`` and ``assets/missing_values.png``.
"""

from __future__ import annotations

import argparse
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
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.metrics import mean_squared_error  # noqa: E402

import datasets as D  # noqa: E402
import style  # noqa: E402
from libs import make_model  # noqa: E402

RATES = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7]
METHODS = {
    "ours": ("ours, native NaN", "#2a78d6", "-", "o"),
    "ours+impute": ("ours + median imputation", "#2a78d6", (0, (4, 2)), "o"),
    "xgboost": ("XGBoost, native NaN", "#eb6834", "-", "s"),
    "lightgbm": ("LightGBM, native NaN", "#1baf7a", "-", "^"),
}


def mask(X: np.ndarray, p: float, rng: np.random.Generator) -> np.ndarray:
    X = X.copy()
    X[rng.random(X.shape) < p] = np.nan
    return X


def rmse(model, X, y) -> float:
    return float(np.sqrt(mean_squared_error(y, model.predict(X))))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--rounds", type=int, default=300)
    args = ap.parse_args()
    out: dict = {m: {str(p): [] for p in RATES} for m in METHODS}
    for seed in args.seeds:
        ds = D.california(seed)
        for p in RATES:
            rng = np.random.default_rng(1000 + seed)
            Xtr, Xte = mask(ds.X_train, p, rng), mask(ds.X_test, p, rng)
            for m in METHODS:
                lib = "ours" if m.startswith("ours") else m
                model = make_model(lib, "regression", args.rounds, 6, seed)
                if m == "ours+impute":
                    imp = SimpleImputer(strategy="median").fit(Xtr)
                    model.fit(imp.transform(Xtr), ds.y_train)
                    r = rmse(model, imp.transform(Xte), ds.y_test)
                else:
                    model.fit(Xtr, ds.y_train)
                    r = rmse(model, Xte, ds.y_test)
                out[m][str(p)].append(r)
            print(f"seed {seed} p={p:.1f} " + " ".join(f"{m}={out[m][str(p)][-1]:.4f}" for m in METHODS), flush=True)
    with open(os.path.join(ROOT, "results", "missing_values.json"), "w") as f:
        json.dump(out, f, indent=1)

    style.apply()
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    for m, (label, c, ls, mk) in METHODS.items():
        mean = np.array([np.mean(out[m][str(p)]) for p in RATES])
        std = np.array([np.std(out[m][str(p)]) for p in RATES])
        ax.plot(np.array(RATES) * 100, mean, color=c, ls=ls, marker=mk, ms=5, mec=style.SURFACE, mew=1, label=label)
        ax.fill_between(np.array(RATES) * 100, mean - std, mean + std, color=c, alpha=0.10, lw=0)
    ax.set_xlabel("entries deleted completely at random (%)")
    ax.set_ylabel("test RMSE (lower is better)")
    ax.set_title("California Housing: native missing-value handling")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "assets", "missing_values.png"), bbox_inches="tight")


if __name__ == "__main__":
    main()
