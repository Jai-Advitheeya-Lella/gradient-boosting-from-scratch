"""Twice-differentiable losses for second-order (Newton) boosting.

A loss only has to provide gradients and hessians of the per-sample loss with
respect to the raw (margin) prediction ``F(x)``. Everything else in the booster
is loss-agnostic, which is what makes the custom-loss API possible.

All methods work on 2-D raw predictions of shape ``(n_samples, n_outputs)``;
``n_outputs`` is 1 except for the multiclass softmax loss.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

import numpy as np

_EPS = 1e-16


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(0.5 * x))  # numerically stable logistic


def _softmax(x: np.ndarray) -> np.ndarray:
    z = x - x.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class Loss(ABC):
    """Base class for boosting losses.

    Subclasses implement :meth:`grad_hess` (and usually :meth:`init_score`,
    :meth:`transform`, :meth:`loss`). ``name`` doubles as the default eval
    metric if no ``eval_metric`` is given.
    """

    n_outputs: int = 1
    name: str = "loss"
    default_metric: str = "loss"

    @abstractmethod
    def grad_hess(self, y: np.ndarray, raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(g, h)``, each of shape ``(n_samples, n_outputs)``."""

    def init_score(self, y: np.ndarray, sample_weight: np.ndarray | None = None) -> np.ndarray:
        """Constant initial raw prediction ``F_0`` (shape ``(n_outputs,)``)."""
        return np.zeros(self.n_outputs)

    def transform(self, raw: np.ndarray) -> np.ndarray:
        """Map raw margins to the prediction space (identity by default).

        Single-output losses return a 1-D array.
        """
        return raw[:, 0] if self.n_outputs == 1 else raw

    def loss(self, y: np.ndarray, raw: np.ndarray, sample_weight: np.ndarray | None = None) -> float:
        """Mean loss; used as an evaluation metric. Optional."""
        raise NotImplementedError


class SquaredError(Loss):
    r"""$\ell = \frac12 (y - F)^2$, so $g = F - y$ and $h = 1$."""

    name = "squared_error"
    default_metric = "rmse"

    def grad_hess(self, y, raw):
        g = raw - y[:, None]
        return g, np.ones_like(g)

    def init_score(self, y, sample_weight=None):
        return np.array([np.average(y, weights=sample_weight)])

    def loss(self, y, raw, sample_weight=None):
        return float(np.average(0.5 * (raw[:, 0] - y) ** 2, weights=sample_weight))


class PseudoHuber(Loss):
    r"""Pseudo-Huber loss $\delta^2(\sqrt{1 + (r/\delta)^2} - 1)$ with $r = F - y$.

    Robust to outliers like Huber, but smooth, so the hessian is well defined:
    $g = r / \sqrt{1 + (r/\delta)^2}$, $h = (1 + (r/\delta)^2)^{-3/2}$.
    """

    name = "pseudo_huber"
    default_metric = "rmse"

    def __init__(self, delta: float = 1.0):
        self.delta = float(delta)

    def grad_hess(self, y, raw):
        r = raw - y[:, None]
        s = 1.0 + (r / self.delta) ** 2
        sq = np.sqrt(s)
        return r / sq, 1.0 / (s * sq)

    def init_score(self, y, sample_weight=None):
        return np.array([np.median(y)])

    def loss(self, y, raw, sample_weight=None):
        r = raw[:, 0] - y
        val = self.delta**2 * (np.sqrt(1.0 + (r / self.delta) ** 2) - 1.0)
        return float(np.average(val, weights=sample_weight))


class Logistic(Loss):
    r"""Binary log-loss on margins: $p = \sigma(F)$, $g = p - y$, $h = p(1-p)$."""

    name = "logistic"
    default_metric = "logloss"

    def grad_hess(self, y, raw):
        p = _sigmoid(raw[:, 0])
        g = (p - y)[:, None]
        h = np.maximum(p * (1.0 - p), _EPS)[:, None]
        return g, h

    def init_score(self, y, sample_weight=None):
        p = np.clip(np.average(y, weights=sample_weight), 1e-12, 1 - 1e-12)
        return np.array([np.log(p / (1.0 - p))])

    def transform(self, raw):
        return _sigmoid(raw[:, 0])

    def loss(self, y, raw, sample_weight=None):
        f = raw[:, 0]
        # log(1 + exp(f)) - y f, computed stably
        val = np.logaddexp(0.0, f) - y * f
        return float(np.average(val, weights=sample_weight))


class Softmax(Loss):
    r"""Multinomial log-loss with $K$ margins per sample.

    $p_k = \mathrm{softmax}(F)_k$, $g_k = p_k - [y = k]$ and the diagonal of the
    hessian $h_k = p_k (1 - p_k)$. One tree per class is grown per round.
    """

    name = "softmax"
    default_metric = "mlogloss"

    def __init__(self, n_classes: int):
        if n_classes < 2:
            raise ValueError("n_classes must be >= 2")
        self.n_outputs = int(n_classes)

    def _onehot(self, y):
        Y = np.zeros((y.shape[0], self.n_outputs))
        Y[np.arange(y.shape[0]), y.astype(np.int64)] = 1.0
        return Y

    def grad_hess(self, y, raw):
        p = _softmax(raw)
        g = p - self._onehot(y)
        h = np.maximum(p * (1.0 - p), _EPS)
        return g, h

    def init_score(self, y, sample_weight=None):
        prior = np.bincount(y.astype(np.int64), weights=sample_weight, minlength=self.n_outputs)
        prior = np.clip(prior / prior.sum(), 1e-12, None)
        logp = np.log(prior)
        return logp - logp.mean()

    def transform(self, raw):
        return _softmax(raw)

    def loss(self, y, raw, sample_weight=None):
        z = raw - raw.max(axis=1, keepdims=True)
        logp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
        val = -logp[np.arange(y.shape[0]), y.astype(np.int64)]
        return float(np.average(val, weights=sample_weight))


class CustomLoss(Loss):
    """Wrap a user function ``f(y, raw) -> (grad, hess)`` into a :class:`Loss`.

    ``raw`` is passed as a 1-D array when ``n_outputs == 1`` (the common case)
    and as ``(n_samples, n_outputs)`` otherwise; the returned arrays must have
    the same shape.

    Example::

        def quartic(y, f):           # l = (f - y)^4 / 4
            r = f - y
            return r**3, 3 * r**2 + 1e-6

        model = GBDTRegressor(loss=CustomLoss(quartic))
    """

    def __init__(
        self,
        grad_hess_fn: Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]],
        n_outputs: int = 1,
        init_fn: Callable[[np.ndarray], np.ndarray | float] | None = None,
        transform_fn: Callable[[np.ndarray], np.ndarray] | None = None,
        loss_fn: Callable[[np.ndarray, np.ndarray], float] | None = None,
        name: str = "custom",
        default_metric: str = "rmse",
    ):
        self.grad_hess_fn = grad_hess_fn
        self.n_outputs = int(n_outputs)
        self.init_fn = init_fn
        self.transform_fn = transform_fn
        self.loss_fn = loss_fn
        self.name = name
        self.default_metric = default_metric if loss_fn is None else "loss"

    def _squeeze(self, raw):
        return raw[:, 0] if self.n_outputs == 1 else raw

    def grad_hess(self, y, raw):
        g, h = self.grad_hess_fn(y, self._squeeze(raw))
        g = np.asarray(g, dtype=np.float64).reshape(raw.shape)
        h = np.asarray(h, dtype=np.float64).reshape(raw.shape)
        return g, h

    def init_score(self, y, sample_weight=None):
        if self.init_fn is None:
            return np.zeros(self.n_outputs)
        return np.atleast_1d(np.asarray(self.init_fn(y), dtype=np.float64))

    def transform(self, raw):
        return self._squeeze(raw) if self.transform_fn is None else self.transform_fn(self._squeeze(raw))

    def loss(self, y, raw, sample_weight=None):
        if self.loss_fn is None:
            raise NotImplementedError
        return float(self.loss_fn(y, self._squeeze(raw)))


def get_loss(loss: str | Loss, n_classes: int | None = None) -> Loss:
    """Resolve a loss given by name or instance."""
    if isinstance(loss, Loss):
        return loss
    if loss == "squared_error":
        return SquaredError()
    if loss == "pseudo_huber":
        return PseudoHuber()
    if loss == "logistic":
        return Logistic()
    if loss == "softmax":
        if n_classes is None:
            raise ValueError("softmax needs n_classes")
        return Softmax(n_classes)
    raise ValueError(f"unknown loss {loss!r}")
