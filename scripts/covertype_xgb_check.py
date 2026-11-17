"""Why does XGBoost trail at a fixed 200-round budget on Covertype?

XGBoost's multiclass softmax objective uses the hessian ``2 p (1 - p)`` (a factor 2 above
the exact diagonal ``p (1 - p)`` used here and in scikit-learn), which halves every Newton
step at equal learning rate. Doubling XGBoost's learning rate to 0.2 should therefore
make it comparable to our depth-wise model at learning rate 0.1. This script runs that check
(split seed 0, same depth-8 configuration as the benchmark). Writes ``results/covertype_xgb_lr.json``.
"""

from __future__ import annotations

import json
import os
import sys
import time

for _k in ("GBDT_BENCH_THREADS", "OMP_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_k] = "1"
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "benchmarks"))

import numpy as np  # noqa: E402
import xgboost as xgb  # noqa: E402
from sklearn.metrics import accuracy_score, log_loss  # noqa: E402

import datasets as D  # noqa: E402

ds = D.load("covertype", 0)
out = {}
for lr in (0.1, 0.2):
    m = xgb.XGBClassifier(n_estimators=200, learning_rate=lr, max_depth=8, reg_lambda=1.0, min_child_weight=1.0,
                          tree_method="hist", max_bin=255, n_jobs=1, random_state=0)
    t0 = time.process_time()
    m.fit(ds.X_train, ds.y_train)
    cpu = time.process_time() - t0
    p = m.predict_proba(ds.X_test)
    out[str(lr)] = {"accuracy": float(accuracy_score(ds.y_test, p.argmax(1))), "logloss": float(log_loss(ds.y_test, p)), "cpu_time": cpu}
    print(f"XGBoost lr={lr}: accuracy {out[str(lr)]['accuracy']:.4f}  log-loss {out[str(lr)]['logloss']:.4f}  ({cpu:.1f}s cpu)")
with open(os.path.join(ROOT, "results", "covertype_xgb_lr.json"), "w") as f:
    json.dump(out, f, indent=1)
