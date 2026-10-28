"""Unit tests for penalty functions and derivatives."""

import pytest
import numpy as np

from src.models.penalties import (
    AtanPenalty,
    LogPenalty,
    McpPenalty,
    ScadPenalty,
    GemanPenalty,
    FractionalPenalty,
    CappedL1Penalty,
    get_penalty,
    PENALTY_REGISTRY,
)


@pytest.fixture
def test_points():
    """Positive test points for regularizer validation."""
    return np.array([0.0, 0.05, 0.1, 0.25, 0.5, 0.8, 1.0, 1.5, 2.0, 3.7, 5.0, 10.0], dtype=np.float64)


def test_penalty_registry():
    """Verify all 7 penalties are registered."""
    expected = {"atan", "log", "mcp", "scad", "geman", "fractional", "capped_l1"}
    assert expected.issubset(set(PENALTY_REGISTRY.keys()))

    for name in expected:
        p = get_penalty(name)
        assert p is not None
        assert p.phi_prime_0 == pytest.approx(1.0, abs=1e-5)


def test_atan_exact_paper_formula():
    """Verify AtanPenalty matches Rabelo et al. (2025) exact formula."""
    penalty = AtanPenalty()
    sqrt3 = np.sqrt(3.0)

    expected_phi_0 = np.pi / (12.0 * sqrt3)
    assert penalty.evaluate(0.0) == pytest.approx(expected_phi_0, rel=1e-7)
    assert penalty.derivative(0.0) == pytest.approx(1.0, rel=1e-7)

    for t in [0.1, 0.5, 1.0, 2.5]:
        expected_deriv = 1.0 / (1.0 + 2.0 * t + 4.0 * (t ** 2))
        assert penalty.derivative(t) == pytest.approx(expected_deriv, rel=1e-7)


@pytest.mark.parametrize("name", ["atan", "log", "mcp", "scad", "geman", "fractional"])
def test_penalty_derivatives_finite_difference(name, test_points):
    """Verify analytical derivatives match central finite differences."""
    penalty = get_penalty(name)
    h = 1e-6
    safe_points = test_points[(test_points > 0.05) & (test_points != 1.0) & (test_points != 3.7)]

    for t in safe_points:
        f_plus = penalty.evaluate(t + h)
        f_minus = penalty.evaluate(t - h)
        num_deriv = (f_plus - f_minus) / (2.0 * h)
        ana_deriv = penalty.derivative(t)
        assert ana_deriv == pytest.approx(num_deriv, rel=1e-3, abs=1e-4)


@pytest.mark.parametrize("name", ["atan", "log", "mcp", "scad", "geman", "fractional", "capped_l1"])
def test_penalty_concavity_and_monotonicity(name):
    """Verify each penalty is non-decreasing and concave."""
    penalty = get_penalty(name)
    t = np.linspace(0.0, 5.0, 200, dtype=np.float64)

    phi_vals = penalty.evaluate(t)
    derivs = penalty.derivative(t)

    assert np.all(derivs >= -1e-12)
    assert np.all(np.diff(phi_vals) >= -1e-12)
    assert np.all(np.diff(derivs) <= 1e-6)


def test_capped_l1_subgradient():
    """Verify Capped-L1 values and derivative."""
    theta = 0.4
    p = CappedL1Penalty(theta=theta)
    assert p.evaluate(0.2) == pytest.approx(0.2)
    assert p.evaluate(0.6) == pytest.approx(theta)
    assert p.derivative(0.2) == pytest.approx(1.0)
    assert p.derivative(0.6) == pytest.approx(0.0)
