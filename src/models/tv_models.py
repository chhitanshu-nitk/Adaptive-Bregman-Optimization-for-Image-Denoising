"""Total Variation optimization model with Non-Convex Arctan Regularization."""

from typing import Tuple
import numpy as np
from src.models.atan_penalty import AtanPenalty


def forward_gradient(u: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute 2D spatial forward finite differences with Neumann boundary conditions."""
    m, n = u.shape
    p = np.zeros((m, n), dtype=np.float64)
    q = np.zeros((m, n), dtype=np.float64)

    p[:-1, :] = u[1:, :] - u[:-1, :]
    q[:, :-1] = u[:, 1:] - u[:, :-1]
    return p, q


def backward_divergence(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Exact negative adjoint of forward_gradient (-div = grad^T)."""
    m, n = p.shape
    div = np.zeros((m, n), dtype=np.float64)

    div[0, :] = p[0, :]
    div[1:-1, :] = p[1:-1, :] - p[:-2, :]
    div[-1, :] = -p[-2, :]

    div[:, 0] += q[:, 0]
    div[:, 1:-1] += q[:, 1:-1] - q[:, :-2]
    div[:, -1] += -q[:, -2]

    return div


class NonConvexTVModel:
    """Model for computing objective value and exact gradient."""

    def __init__(self, mu: float = 0.15, alpha: float = 5.0, epsilon: float = 1e-4):
        self.mu = float(mu)
        self.epsilon = float(epsilon)
        self.penalty = AtanPenalty(alpha=alpha)

    def compute_tv_field(self, u: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        p, q = forward_gradient(u)
        mag = np.sqrt(p ** 2 + q ** 2 + self.epsilon ** 2)
        return p, q, mag

    def objective(self, u: np.ndarray, f: np.ndarray) -> float:
        # Mean squared error fidelity + mean TV penalty
        fidelity = 0.5 * float(np.mean((u - f) ** 2))
        _, _, mag = self.compute_tv_field(u)
        reg = self.mu * float(np.mean(self.penalty.evaluate(mag)))
        return fidelity + reg

    def gradient(self, u: np.ndarray, f: np.ndarray) -> np.ndarray:
        # Normalized gradient matching the mean objective
        n_pixels = float(u.size)
        grad_fidelity = (u - f) / n_pixels

        p, q, mag = self.compute_tv_field(u)
        phi_prime = self.penalty.derivative(mag)
        coeff = phi_prime / mag

        div = backward_divergence(coeff * p, coeff * q)
        grad_reg = -self.mu * (div / n_pixels)

        # Scale gradient so updates are numerically significant per pixel
        return (grad_fidelity + grad_reg) * n_pixels