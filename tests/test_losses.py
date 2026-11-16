"""Finite-difference checks of every loss' gradient and hessian."""

import numpy as np
import pytest

from gbdt.losses import CustomLoss, Logistic, PseudoHuber, Softmax, SquaredError


def _per_sample_loss(loss, y, raw):
    return np.array([loss.loss(y[i : i + 1], raw[i : i + 1]) for i in range(y.shape[0])])


@pytest.mark.parametrize(
    "loss,y",
    [
        (SquaredError(), np.random.default_rng(0).normal(size=50)),
        (PseudoHuber(delta=0.7), np.random.default_rng(1).normal(size=50) * 3),
        (Logistic(), np.random.default_rng(2).integers(0, 2, size=50).astype(float)),
        (Softmax(4), np.random.default_rng(3).integers(0, 4, size=50).astype(float)),
    ],
)
def test_grad_hess_finite_differences(loss, y):
    rng = np.random.default_rng(42)
    raw = rng.normal(size=(y.shape[0], loss.n_outputs))
    g, h = loss.grad_hess(y, raw)
    eps = 1e-5
    for k in range(loss.n_outputs):
        e = np.zeros_like(raw)
        e[:, k] = eps
        lp, l0, lm = (_per_sample_loss(loss, y, raw + s * e) for s in (1, 0, -1))
        num_g = (lp - lm) / (2 * eps)
        num_h = (lp - 2 * l0 + lm) / eps**2
        np.testing.assert_allclose(g[:, k], num_g, rtol=1e-5, atol=1e-7)
        np.testing.assert_allclose(h[:, k], num_h, rtol=1e-3, atol=1e-4)


@pytest.mark.parametrize("loss_cls", [SquaredError, Logistic])
def test_init_score_is_stationary(loss_cls):
    """The constant initial margin must zero the mean gradient."""
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=200).astype(float) if loss_cls is Logistic else rng.normal(size=200)
    loss = loss_cls()
    raw = np.tile(loss.init_score(y), (200, 1))
    g, _ = loss.grad_hess(y, raw)
    assert abs(g.mean()) < 1e-10


def test_softmax_init_matches_priors():
    y = np.array([0, 0, 1, 2, 2, 2], dtype=float)
    loss = Softmax(3)
    p = loss.transform(np.tile(loss.init_score(y), (1, 1)))[0]
    np.testing.assert_allclose(p, [2 / 6, 1 / 6, 3 / 6])


def test_custom_loss_shapes():
    loss = CustomLoss(lambda y, f: (f - y, np.ones_like(f)))
    y = np.arange(5.0)
    g, h = loss.grad_hess(y, np.zeros((5, 1)))
    assert g.shape == (5, 1) and h.shape == (5, 1)
    np.testing.assert_allclose(g[:, 0], -y)
