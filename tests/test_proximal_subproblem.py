"""Unit tests for proximal subproblem solver and optimality conditions."""

import pytest
import numpy as np

from src.optimizers.nmPPMDC import solve_weighted_rof_fgp
from src.models.tv_models import forward_gradient, backward_divergence


def test_weighted_rof_optimality():
    """Verify solve_weighted_rof_fgp satisfies optimality conditions."""
    rng = np.random.default_rng(42)
    b_tilde = rng.uniform(0.1, 0.9, size=(16, 16))
    alpha = rng.uniform(0.5, 2.0, size=(16, 16))
    tau = 0.5

    x_sol, (p_sol, q_sol) = solve_weighted_rof_fgp(
        b_tilde=b_tilde,
        alpha=alpha,
        tau=tau,
        max_inner_iter=50,
        tol=1e-5
    )

    # Dual constraints
    mag_dual = np.sqrt(p_sol ** 2 + q_sol ** 2)
    assert np.all(mag_dual <= 1.0 + 1e-6)

    # Primal-dual relationship
    expected_x = b_tilde + (tau / alpha) * backward_divergence(p_sol, q_sol)
    assert np.allclose(x_sol, expected_x, atol=1e-6)

    # Energy reduction check
    p_x, q_x = forward_gradient(x_sol)
    tv_x = np.sum(np.sqrt(p_x ** 2 + q_x ** 2))
    obj_x = 0.5 * np.sum(alpha * ((x_sol - b_tilde) ** 2)) + tau * tv_x

    p_b, q_b = forward_gradient(b_tilde)
    tv_b = np.sum(np.sqrt(p_b ** 2 + q_b ** 2))
    obj_b = tau * tv_b

    assert obj_x <= obj_b + 1e-6
