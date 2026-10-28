"""Unit tests for Euclidean and Adaptive Bregman distances."""

import pytest
import numpy as np

from src.bregman.euclidean import EuclideanBregman
from src.bregman.adaptive import AdaptiveBregman


@pytest.fixture
def test_fields():
    flat = np.ones((16, 16), dtype=np.float64) * 0.5
    edge = np.zeros((16, 16), dtype=np.float64)
    edge[:, 8:] = 1.0
    return flat, edge


def test_euclidean_distance():
    """Verify Euclidean distance."""
    euc = EuclideanBregman()
    x = np.ones((8, 8))
    y = np.zeros((8, 8))

    assert euc.distance(x, x) == pytest.approx(0.0)
    assert euc.distance(x, y) == pytest.approx(0.5 * 64.0)

    grad = euc.grad_first_argument(x, y)
    assert np.allclose(grad, x - y)


def test_adaptive_edge_sensitivity(test_fields):
    """Verify adaptive weights decrease at edges."""
    flat, edge = test_fields
    ada = AdaptiveBregman(beta=10.0, epsilon_w=1e-4)

    w_flat = ada.compute_weights(flat)
    w_edge = ada.compute_weights(edge)

    assert np.allclose(w_flat, 1.0)
    assert np.min(w_edge) < 0.2
    assert np.max(w_edge) == pytest.approx(1.0, rel=1e-3)
    assert np.all(w_edge >= 1e-4)


def test_adaptive_distance_nonnegativity(test_fields):
    """Verify Bregman distance is non-negative and zero iff identical."""
    _, edge = test_fields
    ada = AdaptiveBregman(beta=5.0)

    perturbation = np.random.default_rng(42).standard_normal(edge.shape) * 0.1
    dist = ada.distance(edge + perturbation, edge)

    assert dist > 0.0
    assert ada.distance(edge, edge) == pytest.approx(0.0)
