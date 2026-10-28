"""Unit tests for nmPPMDC optimizer and convergence."""

import pytest
import numpy as np

from src.optimizers.nmPPMDC import nmPPMDCOptimizer
from src.noise.gaussian import add_gaussian_noise


@pytest.fixture
def test_setup():
    clean = np.zeros((32, 32), dtype=np.float64)
    clean[8:24, 8:24] = 0.8
    clean[12:20, 12:20] = 0.3

    noisy, _ = add_gaussian_noise(clean, sigma=25.0, seed=42)
    return clean, noisy


def test_optimizer_runs_and_denoises(test_setup):
    """Verify nmPPMDC reduces noise."""
    clean, noisy = test_setup

    config = {
        "model": {"mu": 12.0, "tau": 0.35, "penalty": "atan", "beta": 8.0},
        "optimizer": {"max_iter": 30, "tol": 1e-4, "max_inner_iter": 25, "nu": 500.0, "zeta": 0.09, "t0": 0.9},
        "bregman": {"beta_adaptive": 8.0, "epsilon_w": 1e-4}
    }

    opt = nmPPMDCOptimizer(config)
    u_denoised, history = opt.optimize(noisy, mode="adaptive")

    mse_noisy = np.mean((clean - noisy) ** 2)
    mse_denoised = np.mean((clean - u_denoised) ** 2)

    assert mse_denoised < mse_noisy
    assert history["iterations"] > 0
    assert len(history["energy_history"]) > 1


@pytest.mark.parametrize("mode", ["euclidean", "adaptive"])
def test_both_bregman_modes(test_setup, mode):
    """Verify Euclidean and Adaptive modes run properly."""
    _, noisy = test_setup

    config = {
        "model": {"mu": 12.0, "tau": 0.35, "penalty": "atan", "beta": 5.0},
        "optimizer": {"max_iter": 5, "tol": 1e-4, "max_inner_iter": 15},
    }

    opt = nmPPMDCOptimizer(config)
    u, hist = opt.optimize(noisy, mode=mode)

    assert u.shape == noisy.shape
    assert hist["mode"] == mode
    assert len(hist["energy_history"]) >= 2


@pytest.mark.parametrize("penalty", ["atan", "log", "mcp", "scad", "geman", "fractional", "capped_l1"])
def test_all_seven_penalties(test_setup, penalty):
    """Verify all 7 non-convex penalties execute properly."""
    _, noisy = test_setup

    config = {
        "model": {"mu": 12.0, "tau": 0.35, "penalty": penalty},
        "optimizer": {"max_iter": 5, "tol": 1e-4, "max_inner_iter": 15},
    }

    opt = nmPPMDCOptimizer(config)
    u, hist = opt.optimize(noisy, mode="adaptive", penalty=penalty)

    assert u.shape == noisy.shape
    assert hist["penalty"] == penalty
    assert not np.isnan(u).any()
