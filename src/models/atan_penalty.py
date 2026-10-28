"""Non-convex arctangent penalty function and its derivative.

Equations:
    Penalty:
        phi(t; alpha) = (2 / (pi * alpha)) * arctan(alpha * t), for t >= 0
    Derivative:
        phi'(t; alpha) = 2 / (pi * (1 + (alpha * t)^2))
"""

from typing import Union
import numpy as np


class AtanPenalty:
    r"""Non-convex Arctangent Regularization Penalty.

    The penalty function is defined as:
    .. math::
        \phi(t) = \frac{2}{\pi \alpha} \arctan(\alpha t)

    With first derivative:
    .. math::
        \phi'(t) = \frac{2}{\pi (1 + \alpha^2 t^2)}
    """

    def __init__(self, alpha: float = 10.0):
        if alpha <= 0.0:
            raise ValueError(f"Alpha must be positive. Got: {alpha}")
        self.alpha = float(alpha)
        self._norm_const = 2.0 / (np.pi * self.alpha)
        self._deriv_const = 2.0 / np.pi

    def evaluate(self, t: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Compute the arctan penalty phi(t)."""
        return self._norm_const * np.arctan(self.alpha * t)

    def derivative(self, t: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Compute the first-order derivative phi'(t)."""
        return self._deriv_const / (1.0 + (self.alpha * t) ** 2)