"""Standard Euclidean Bregman Distance (Baseline)."""

from typing import Optional
import numpy as np


class EuclideanBregman:
    """Euclidean Bregman distance d_q(x, y) = (1/2) * ||x - y||^2."""

    name: str = "euclidean"

    def compute_weights(self, v: np.ndarray) -> np.ndarray:
        """Uniform weights w_i = 1.0."""
        return np.ones_like(v, dtype=np.float64)

    def distance(self, x: np.ndarray, y: np.ndarray, weights: Optional[np.ndarray] = None) -> float:
        """Compute Euclidean distance d(x, y) = 0.5 * ||x - y||^2."""
        diff = x - y
        if weights is not None:
            return 0.5 * float(np.sum(weights * (diff ** 2)))
        return 0.5 * float(np.sum(diff ** 2))

    def grad_first_argument(self, x: np.ndarray, y: np.ndarray, weights: Optional[np.ndarray] = None) -> np.ndarray:
        """Gradient w.r.t first argument: nabla_x d_q(x, y) = x - y."""
        diff = x - y
        if weights is not None:
            return weights * diff
        return diff