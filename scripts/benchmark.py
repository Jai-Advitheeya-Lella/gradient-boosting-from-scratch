"""Benchmark ours vs XGBoost / LightGBM / scikit-learn HistGradientBoosting.

For every dataset and split seed, every library trains a fixed number of rounds
(no early stopping, so training times are comparable) and is scored on the test
split. Three kinds of results are written to ``results/benchmark_<dataset>.json``:

1. ``final``: test metrics and training time (binning included) after the full budget.
2. ``curves``: per-round train / validation / test loss (seed 0 only) and the test
   metrics of the model truncated at the validation-optimal round.
3. ``time_curve``: test metric and training time for several round budgets (seed 0).

Usage::

    python scripts/benchmark.py --datasets california adult higgs covertype
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings

_THREADS = os.environ.get("GBDT_BENCH_THREADS", "1")
os.environ["GBDT_BENCH_THREADS"] = _THREADS
os.environ["OMP_NUM_THREADS"] = _THREADS
os.environ["NUMBA_NUM_THREADS"] = _THREADS

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "benchmarks"))

import numpy as np  # noqa: E402
from sklearn.metrics import accuracy_score, log_loss, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score  # noqa: E402

import datasets as D  # noqa: E402
from libs import LIBS, make_model, predict_for_score  # noqa: E402

# depth (=> 2**depth leaves) and number of boosting rounds per dataset
CONFIG = {
    "california": dict(depth=6, rounds=400),
    "adult": dict(depth=6, rounds=300),
    "higgs": dict(depth=6, rounds=300),
    "covertype": dict(depth=8, rounds=200),
}
CHECKPOINTS = [10, 25, 50, 100, 200]
warnings.filterwarnings("ignore", message=".*eval_set.*")


def score(task: str, y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    if task == "regression":
        return {
            "rmse": float(np.sqrt(mean_squared_error(y, pred))),
            "mae": float(mean_absolute_error(y, pred)),
            "r2": float(r2_score(y, pred)),
        }
    if task == "binary":
        p = pred[:, 1]
        return {
            "accuracy": float(accuracy_score(y, p > 0.5)),
            "auc": float(roc_auc_score(y, p)),
            "logloss": float(log_loss(y, np.clip(p, 1e-15, 1 - 1e-15))),
        }
    return {
        "accuracy": float(accuracy_score(y, pred.argmax(1))),
        "logloss": float(log_loss(y, np.clip(pred, 1e-15, 1.0), labels=np.arange(pred.shape[1]))),
    }


def primary(task: str) -> str:
    return "rmse" if task == "regression" else "logloss"


def fit_timed(lib, ds, rounds, depth, seed):
    model = make_model(lib, ds.task, rounds, depth, seed)
    w0, c0 = time.perf_counter(), time.process_time()
    model.fit(ds.X_train, ds.y_train)
    fit_time = time.perf_counter() - w0
    cpu_time = time.process_time() - c0
    t0 = time.perf_counter()
    pred = predict_for_score(model, lib, ds.task, ds.X_test)
    pred_time = time.perf_counter() - t0
    return model, pred, (fit_time, cpu_time), pred_time


def loss_curves(lib, ds, rounds, depth, seed=0):
    """Per-round primary metric on train/val/test using each library's own logging."""
    task = ds.task
    sets = {"train": (ds.X_train, ds.y_train), "val": (ds.X_val, ds.y_val), "test": (ds.X_test, ds.y_test)}
    model = make_model(lib, task, rounds, depth, seed)
    if lib.startswith("ours"):
        model.fit(ds.X_train, ds.y_train, eval_set=list(sets.values()))
        res = model.evals_result_
        return {k: res[f"valid_{i}"][model.eval_metric_] for i, k in enumerate(sets)}, model
    if lib == "xgboost":
        metric = "rmse" if task == "regression" else "logloss" if task == "binary" else "mlogloss"
        model.set_params(eval_metric=metric)
        model.fit(ds.X_train, ds.y_train, eval_set=list(sets.values()), verbose=False)
        res = model.evals_result()
        return {k: list(res[f"validation_{i}"][metric]) for i, k in enumerate(sets)}, model
    if lib == "lightgbm":
        metric = "rmse" if task == "regression" else "binary_logloss" if task == "binary" else "multi_logloss"
        model.fit(
            ds.X_train, ds.y_train, eval_set=list(sets.values()), eval_names=list(sets), eval_metric=metric,
        )
        res = model.evals_result_
        return {k: list(res[k][metric]) for k in sets}, model
    # scikit-learn: no logging, replay the stages
    model.fit(ds.X_train, ds.y_train)
    out = {}
    for k, (X, y) in sets.items():
        vals = []
        if task == "regression":
            for p in model.staged_predict(X):
                vals.append(float(np.sqrt(mean_squared_error(y, p))))
        else:
            for p in model.staged_predict_proba(X):
                if task == "binary":
                    vals.append(float(log_loss(y, np.clip(p[:, 1], 1e-15, 1 - 1e-15), labels=[0, 1])))
                else:
                    vals.append(float(log_loss(y, np.clip(p, 1e-15, 1.0), labels=np.arange(p.shape[1]))))
        out[k] = vals
    return out, model


def predict_truncated(model, lib, task, X, n):
    """Predictions of the first ``n`` rounds."""
    if lib.startswith("ours"):
        if task == "regression":
            return model.predict(X, n_iterations=n)
        return model.predict_proba(X, n_iterations=n)
    if lib == "xgboost":
        if task == "regression":
            return model.predict(X, iteration_range=(0, n))
        return model.predict_proba(X, iteration_range=(0, n))
    if lib == "lightgbm":
        if task == "regression":
            return model.predict(X, num_iteration=n)
        return model.predict_proba(X, num_iteration=n)
    stages = model.staged_predict if task == "regression" else model.staged_predict_proba
    for i, p in enumerate(stages(X), start=1):
        if i == n:
            return p
    raise AssertionError


def run_dataset(name: str, seeds: list[int], libs: list[str]) -> dict:
    cfg = CONFIG[name]
    depth, rounds = cfg["depth"], cfg["rounds"]
    result: dict = {"dataset": name, "config": cfg, "threads": int(_THREADS), "final": {}, "curves": {}, "time_curve": {}}
    # compile numba kernels once, outside the timed region
    warm = D.load(name, 0)
    make_model("ours", warm.task, 2, depth).fit(warm.X_train[:3000], warm.y_train[:3000])
    make_model("ours-leafwise", warm.task, 2, depth).fit(warm.X_train[:3000], warm.y_train[:3000])

    for seed in seeds:
        ds = D.load(name, seed)
        if seed == seeds[0]:
            print(ds.describe(), flush=True)
            result["info"] = dict(
                n_train=int(ds.X_train.shape[0]), n_val=int(ds.X_val.shape[0]), n_test=int(ds.X_test.shape[0]),
                n_features=int(ds.X_train.shape[1]), task=ds.task, n_classes=ds.n_classes,
                missing_frac=float(np.isnan(ds.X_train).mean()),
            )
        for lib in libs:
            model, pred, ft, pt = fit_timed(lib, ds, rounds, depth, seed)
            m = score(ds.task, ds.y_test, pred)
            result["final"].setdefault(lib, []).append(
                {"seed": seed, "fit_time": ft[0], "cpu_time": ft[1], "predict_time": pt, **m}
            )
            print(f"  [{name} seed={seed}] {lib:14s} fit {ft[0]:7.2f}s wall {ft[1]:7.2f}s cpu  " + "  ".join(f"{k}={v:.4f}" for k, v in m.items()), flush=True)
            del model

    ds = D.load(name, seeds[0])
    pm = primary(ds.task)
    for lib in libs:
        curves, model = loss_curves(lib, ds, rounds, depth, seeds[0])
        best = int(np.argmin(curves["val"])) + 1
        pred = predict_truncated(model, lib, ds.task, ds.X_test, best)
        result["curves"][lib] = {
            "metric": pm, **curves, "best_round": best, "test_at_best": score(ds.task, ds.y_test, pred),
        }
        print(f"  [{name}] curves {lib:14s} best val round {best}  " + "  ".join(f"{k}={v:.4f}" for k, v in result["curves"][lib]["test_at_best"].items()), flush=True)
        del model

    for lib in libs:
        pts = []
        for r in CHECKPOINTS + [rounds]:
            if r > rounds or (pts and r == pts[-1]["rounds"]):
                continue
            if r == rounds:  # reuse the seed-0 full-budget run
                f0 = result["final"][lib][0]
                pts.append({"rounds": r, "fit_time": f0["fit_time"], "cpu_time": f0["cpu_time"], **{k: v for k, v in f0.items() if k not in ("seed", "fit_time", "cpu_time", "predict_time")}})
                continue
            model, pred, ft, _ = fit_timed(lib, ds, r, depth, seeds[0])
            pts.append({"rounds": r, "fit_time": ft[0], "cpu_time": ft[1], **score(ds.task, ds.y_test, pred)})
            del model
        result["time_curve"][lib] = pts
        print(f"  [{name}] time curve {lib}: " + ", ".join(f"{p['rounds']}r={p['cpu_time']:.2f}s" for p in pts), flush=True)
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=list(CONFIG), choices=list(CONFIG))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2], help="train/val/test split seeds")
    ap.add_argument("--libs", nargs="+", default=LIBS, choices=LIBS)
    ap.add_argument("--out", default=os.path.join(ROOT, "results"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for name in args.datasets:
        t0 = time.time()
        res = run_dataset(name, args.seeds, args.libs)
        res["wall_seconds"] = time.time() - t0
        with open(os.path.join(args.out, f"benchmark_{name}.json"), "w") as f:
            json.dump(res, f, indent=1)
        print(f"== {name} done in {res['wall_seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
