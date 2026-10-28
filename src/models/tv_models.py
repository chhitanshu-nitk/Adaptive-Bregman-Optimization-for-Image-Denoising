"""Non-convex Total Variation model with Difference-of-Convex (DC) splitting."""

from typing import Optional, Tuple, Union
import numpy as np

from src.models.penalties import BasePenalty, get_penalty, AtanPenalty


def forward_gradient(u: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """2D spatial forward differences with Neumann boundary conditions."""
    m, n = u.shape
    p = np.zeros((m, n), dtype=np.float64)
    q = np.zeros((m, n), dtype=np.float64)

    p[:-1, :] = u[1:, :] - u[:-1, :]
    q[:, :-1] = u[:, 1:] - u[:, :-1]
    return p, q


def backward_divergence(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Discrete divergence operator with Dirichlet boundary condition."""
    m, n = p.shape
    div = np.zeros((m, n), dtype=np.float64)

    div[0, :] = p[0, :]
    div[1:-1, :] = p[1:-1, :] - p[:-2, :]
    div[-1, :] = -p[-2, :]

    div[:, 0] += q[:, 0]
    div[:, 1:-1] += q[:, 1:-1] - q[:, :-2]
    div[:, -1] += -q[:, -2]

    return div


def grad_transpose(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Exact adjoint of forward_gradient: grad^T (p, q) = -div(p, q)."""
    return -backward_divergence(p, q)


class NonConvexTVModel:
    """Non-convex Total Variation model with DC decomposition f(u) = g(u) - h(u)."""

    def __init__(
        self,
        mu: float = 0.5,
        tau: float = 1.0,
        penalty: Union[str, BasePenalty] = "atan",
        epsilon: float = 1e-6,
        **penalty_kwargs
    ):
        self.mu = float(mu)
        self.epsilon = float(epsilon)

        if isinstance(penalty, str):
            self.penalty = get_penalty(penalty, **penalty_kwargs)
        elif isinstance(penalty, BasePenalty):
            self.penalty = penalty
        else:
            raise TypeError(f"Invalid penalty type: {type(penalty)}")

        self.tau = float(tau)

    def compute_gradient_field(self, u: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute forward differences and gradient magnitude."""
        p, q = forward_gradient(u)
        mag = np.sqrt(p ** 2 + q ** 2)
        return p, q, mag

    def tv_convex(self, u: np.ndarray) -> float:
        """Convex Total Variation TV(u) = sum ||grad u||_2."""
        _, _, mag = self.compute_gradient_field(u)
        return float(np.sum(mag))

    def tv_phi(self, u: np.ndarray) -> float:
        """Non-convex Total Variation TV_phi(u) = sum phi(||grad u||_2)."""
        _, _, mag = self.compute_gradient_field(u)
        return float(np.sum(self.penalty.evaluate(mag)))

    def objective(self, u: np.ndarray, b: np.ndarray) -> float:
        """Total objective: f(u) = (mu / 2) * ||u - b||_2^2 + TV_phi(u)."""
        fidelity = 0.5 * self.mu * float(np.sum((u - b) ** 2))
        return fidelity + self.tv_phi(u)

    def g(self, u: np.ndarray, b: np.ndarray) -> float:
        """First DC component: g(u) = (mu / 2) * ||u - b||_2^2 + tau * TV(u)."""
        fidelity = 0.5 * self.mu * float(np.sum((u - b) ** 2))
        return fidelity + self.tau * self.tv_convex(u)

    def h(self, u: np.ndarray) -> float:
        """Second DC component: h(u) = tau * TV(u) - TV_phi(u)."""
        return self.tau * self.tv_convex(u) - self.tv_phi(u)

    def subgradient_h(self, u: np.ndarray) -> np.ndarray:
        r"""Compute subgradient w \in \partial h(u) via adjoint chain rule."""
        p, q, mag = self.compute_gradient_field(u)
        mag_safe = np.sqrt(p ** 2 + q ** 2 + self.epsilon ** 2)

        phi_prime = self.penalty.derivative(mag)
        diff = np.maximum(0.0, self.tau - phi_prime)
        coeff = diff / mag_safe

        return grad_transpose(coeff * p, coeff * q)

    def gradient(self, u: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Compute exact gradient of smoothed objective nabla f(u)."""
        p, q, _ = self.compute_gradient_field(u)
        mag_eps = np.sqrt(p ** 2 + q ** 2 + self.epsilon ** 2)

        phi_prime = self.penalty.derivative(mag_eps)
        coeff = phi_prime / mag_eps

        grad_reg = grad_transpose(coeff * p, coeff * q)
        grad_fidelity = self.mu * (u - b)
        return grad_fidelity + grad_reg