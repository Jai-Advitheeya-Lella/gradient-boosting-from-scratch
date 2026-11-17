"""Custom-loss API: robust regression on data with gross outliers.

Any twice-differentiable loss works if you supply the gradient and hessian of the
per-sample loss with respect to the raw prediction. Here 10% of the training targets
are corrupted; squared error chases them, pseudo-Huber (built in) and a user-defined
log-cosh loss (via ``CustomLoss``) do not.

Run: ``python examples/custom_loss.py``
"""

from __future__ import annotations

import numpy as np

from gbdt import CustomLoss, GBDTRegressor, PseudoHuber

rng = np.random.default_rng(0)
n = 6000
X = rng.uniform(-3, 3, size=(n, 5))
signal = np.sin(2 * X[:, 0]) + 0.5 * X[:, 1] ** 2 - X[:, 2] * X[:, 3]
y = signal + rng.normal(0, 0.3, n)
bad = rng.random(n) < 0.10
y_train = y.copy()
y_train[bad] += rng.choice([-1, 1], bad.sum()) * rng.uniform(8, 20, bad.sum())  # gross outliers in train only
tr, te = slice(0, 4000), slice(4000, None)


def log_cosh(y_true, raw):
    """l = log cosh(F - y): g = tanh(r), h = sech^2(r) = 1 - tanh^2(r)."""
    t = np.tanh(raw - y_true)
    return t, np.maximum(1.0 - t**2, 1e-6)


models = {
    "squared error": GBDTRegressor(),
    "pseudo-Huber (built in, delta=1)": GBDTRegressor(loss=PseudoHuber(delta=1.0)),
    "log-cosh (CustomLoss)": GBDTRegressor(loss=CustomLoss(log_cosh, init_fn=np.median)),
}
print(f"{'loss':36s} {'RMSE vs clean signal':>22s} {'MAE vs clean signal':>22s}")
for name, m in models.items():
    m.n_estimators, m.learning_rate, m.max_depth = 200, 0.1, 4
    m.fit(X[tr], y_train[tr])
    pred = m.predict(X[te])
    err = pred - signal[te]  # error against the noiseless signal
    print(f"{name:36s} {np.sqrt(np.mean(err**2)):22.4f} {np.mean(np.abs(err)):22.4f}")
