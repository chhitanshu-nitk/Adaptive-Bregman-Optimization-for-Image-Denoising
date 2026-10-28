"""Gradient-Adaptive Bregman Distance Kernel."""

from typing import Optional
import numpy as np
from src.models.tv_models import forward_gradient


class AdaptiveBregman:
    """Adaptive Bregman distance with edge-preserving weight map."""

    name: str = "adaptive"

    def __init__(self, beta: float = 8.0, epsilon_w: float = 1e-4):
        if beta < 0:
            raise ValueError(f"beta must be non-negative, got {beta}")
        self.beta = float(beta)
        self.epsilon_w = float(epsilon_w)

    def compute_weights(self, v: np.ndarray) -> np.ndarray:
        """Compute weight map w_i = max(1 / (1 + beta * ||grad v||^2), epsilon_w)."""
        p, q = forward_gradient(v)
        grad_sq = p ** 2 + q ** 2
        w = 1.0 / (1.0 + self.beta * grad_sq)
        return np.maximum(w, self.epsilon_w)

    def distance(self, x: np.ndarray, y: np.ndarray, weights: Optional[np.ndarray] = None) -> float:
        """Compute d_q(x, y; W) = (1/2) * sum w_i * (x_i - y_i)^2."""
        if weights is None:
            weights = self.compute_weights(y)
        diff = x - y
        return 0.5 * float(np.sum(weights * (diff ** 2)))

    def grad_first_argument(self, x: np.ndarray, y: np.ndarray, weights: Optional[np.ndarray] = None) -> np.ndarray:
        """Gradient w.r.t first argument: nabla_x d_q(x, y; W) = W * (x - y)."""
        if weights is None:
            weights = self.compute_weights(y)
        return weights * (x - y)