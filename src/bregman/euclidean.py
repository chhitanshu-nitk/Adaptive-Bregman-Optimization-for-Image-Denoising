"""Euclidean Bregman Distance implementation.

Mathematical formulation:
    D_psi(u, v) = psi(u) - psi(v) - <nabla psi(v), u - v>
For quadratic generating kernel psi(u) = (1/2) * ||u||_2^2:
    nabla psi(v) = v
    D_psi(u, v) = (1/2) * ||u - v||_2^2
"""

import numpy as np


class EuclideanBregman:
    """Standard Euclidean Bregman Distance (Baseline)."""

    def distance(self, u: np.ndarray, v: np.ndarray) -> float:
        r"""Compute Euclidean Bregman Distance:

        .. math::
            D(u, v) = \frac{1}{2} \|u - v\|_2^2
        """
        return 0.5 * float(np.sum((u - v) ** 2))

    def grad_second_argument(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        r"""Compute gradient with respect to second argument v:

        .. math::
            \nabla_v D(u, v) = v - u
        """
        return v - u

    def apply_metric_inv(self, g: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Inverse metric operator H(v)^{-1} * g.

        For Euclidean distance, H(v) = Identity.
        """
        return g