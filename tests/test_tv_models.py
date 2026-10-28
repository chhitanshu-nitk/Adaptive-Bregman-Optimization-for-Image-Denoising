"""Unit tests for TV operators, DC splitting, and gradients."""

import pytest
import numpy as np

from src.models.tv_models import (
    forward_gradient,
    backward_divergence,
    grad_transpose,
    NonConvexTVModel,
)


@pytest.fixture
def sample_image():
    """16x16 test image."""
    rng = np.random.default_rng(42)
    return rng.uniform(0.0, 1.0, size=(16, 16)).astype(np.float64)


def test_gradient_divergence_adjoint():
    """Verify adjointness: <grad u, (p, q)> == <u, -div(p, q)>."""
    rng = np.random.default_rng(123)
    u = rng.standard_normal((16, 16))
    p = rng.standard_normal((16, 16))
    q = rng.standard_normal((16, 16))

    grad_u_p, grad_u_q = forward_gradient(u)
    inner_grad = np.sum(grad_u_p * p + grad_u_q * q)
    gt = grad_transpose(p, q)
    inner_adjoint = np.sum(u * gt)

    assert inner_grad == pytest.approx(inner_adjoint, rel=1e-10)


@pytest.mark.parametrize("penalty_name", ["atan", "log", "mcp", "scad", "geman", "fractional", "capped_l1"])
def test_dc_decomposition_identity(sample_image, penalty_name):
    """Verify f(u) == g(u) - h(u)."""
    b = np.copy(sample_image)
    model = NonConvexTVModel(mu=0.5, tau=1.0, penalty=penalty_name)

    f_val = model.objective(sample_image, b)
    g_val = model.g(sample_image, b)
    h_val = model.h(sample_image)

    assert f_val == pytest.approx(g_val - h_val, rel=1e-8)


def test_gradient_finite_difference(sample_image):
    """Verify gradient matches numerical directional derivatives."""
    b = sample_image * 0.9
    model = NonConvexTVModel(mu=0.5, tau=1.0, penalty="atan", epsilon=1e-4)
    grad = model.gradient(sample_image, b)

    rng = np.random.default_rng(999)
    v = rng.standard_normal(sample_image.shape)
    v /= np.linalg.norm(v)

    eps = 1e-6
    f_plus = model.objective(sample_image + eps * v, b)
    f_minus = model.objective(sample_image - eps * v, b)
    num_dir_deriv = (f_plus - f_minus) / (2.0 * eps)
    ana_dir_deriv = float(np.sum(grad * v))

    assert ana_dir_deriv == pytest.approx(num_dir_deriv, rel=1e-2, abs=1e-3)


def test_subgradient_h_shape_and_finite(sample_image):
    """Verify subgradient w is finite and correctly shaped."""
    model = NonConvexTVModel(mu=0.5, tau=1.0, penalty="atan")
    w = model.subgradient_h(sample_image)

    assert w.shape == sample_image.shape
    assert np.all(np.isfinite(w))
