"""Train the from-scratch booster on one of the benchmark datasets.

Example::

    python scripts/train.py --dataset adult --rounds 500 --depth 6 --early-stopping 30
"""

from __future__ import annotations

import argparse
import os
import sys
import time

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "benchmarks"))

import numpy as np  # noqa: E402
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score  # noqa: E402

import datasets as D  # noqa: E402
from gbdt import GBDTClassifier, GBDTRegressor  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=list(D.LOADERS), default="california")
    ap.add_argument("--rounds", type=int, default=300)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--leaves", type=int, default=0, help="max leaves (with --policy lossguide)")
    ap.add_argument("--policy", choices=["depthwise", "lossguide"], default="depthwise")
    ap.add_argument("--reg-lambda", type=float, default=1.0)
    ap.add_argument("--reg-alpha", type=float, default=0.0)
    ap.add_argument("--gamma", type=float, default=0.0)
    ap.add_argument("--min-child-weight", type=float, default=1.0)
    ap.add_argument("--subsample", type=float, default=1.0)
    ap.add_argument("--sampling", choices=["uniform", "goss"], default="uniform")
    ap.add_argument("--colsample-bytree", type=float, default=1.0)
    ap.add_argument("--max-bins", type=int, default=256)
    ap.add_argument("--early-stopping", type=int, default=None, help="patience in rounds (on the validation split)")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ds = D.load(args.dataset, args.seed)
    print(ds.describe())
    cls = GBDTRegressor if ds.task == "regression" else GBDTClassifier
    model = cls(
        n_estimators=args.rounds, learning_rate=args.lr, max_depth=args.depth, max_leaves=args.leaves,
        grow_policy=args.policy, reg_lambda=args.reg_lambda, reg_alpha=args.reg_alpha, gamma=args.gamma,
        min_child_weight=args.min_child_weight, subsample=args.subsample, sampling=args.sampling,
        colsample_bytree=args.colsample_bytree, max_bins=args.max_bins,
        early_stopping_rounds=args.early_stopping, n_threads=args.threads, random_state=args.seed, verbose=25,
    )
    t0 = time.perf_counter()
    model.fit(ds.X_train, ds.y_train, eval_set=[(ds.X_val, ds.y_val)])
    print(f"trained {model.n_iter_} rounds ({model.n_trees_} trees) in {time.perf_counter() - t0:.2f}s"
          + (f", best round {model.best_iteration_ + 1}" if model.best_iteration_ is not None else ""))
    if ds.task == "regression":
        pred = model.predict(ds.X_test)
        print(f"test RMSE {np.sqrt(np.mean((pred - ds.y_test) ** 2)):.4f}")
    else:
        proba = model.predict_proba(ds.X_test)
        acc = accuracy_score(ds.y_test, proba.argmax(1))
        msg = f"test accuracy {acc:.4f}  log-loss {log_loss(ds.y_test, proba):.4f}"
        if ds.task == "binary":
            msg += f"  AUC {roc_auc_score(ds.y_test, proba[:, 1]):.4f}"
        print(msg)
    top = np.argsort(-model.get_importance("total_gain"))[:5]
    print("top features by total gain:", ", ".join(ds.feature_names[i] for i in top))


if __name__ == "__main__":
    main()
