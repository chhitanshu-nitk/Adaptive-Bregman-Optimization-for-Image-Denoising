"""Gradient-Adaptive Bregman Distance Module."""

import numpy as np
from src.models.tv_models import forward_gradient


class AdaptiveBregman:
    """Gradient-Adaptive Bregman Distance."""

    def __init__(self, beta: float = 50.0, epsilon: float = 1e-4):
        self.beta = float(beta)
        self.epsilon = float(epsilon)

    def compute_weight_kernel(self, v: np.ndarray) -> np.ndarray:
        r"""Compute edge-preserving weight map:
        w(x) = 1 / (1 + beta * |nabla v|^2)
        Flat areas -> w ~ 1 (strong diffusion)
        Edges      -> w << 1 (protects sharpness)
        """
        p, q = forward_gradient(v)
        grad_sqr = p ** 2 + q ** 2
        weights = 1.0 / (1.0 + self.beta * grad_sqr)
        return weights

    def distance(self, u: np.ndarray, v: np.ndarray) -> float:
        weights = self.compute_weight_kernel(v)
        return 0.5 * float(np.mean(weights * ((u - v) ** 2)))

    def apply_metric_inv(self, g: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Precondition gradient with edge weight map.
        
        Flat regions receive full gradient step for aggressive noise suppression.
        Edge regions receive damped gradient step to preserve sharp boundaries.
        """
        w = self.compute_weight_kernel(v)
        return g * np.clip(w, 0.1, 1.0)