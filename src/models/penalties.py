"""Non-convex penalty functions and derivatives for DC Total Variation models."""

from abc import ABC, abstractmethod
from typing import Dict, Type, Union
import numpy as np


class BasePenalty(ABC):
    """Abstract base class for regularizer penalties."""

    name: str = "base"

    @abstractmethod
    def evaluate(self, t: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Compute phi(t)."""
        pass

    @abstractmethod
    def derivative(self, t: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Compute phi'(t)."""
        pass

    @property
    def phi_prime_0(self) -> float:
        """Right derivative at origin phi'_+(0)."""
        return 1.0


class AtanPenalty(BasePenalty):
    """Paper exact Arctangent penalty: phi(t) = (2*atan((1+4t)/sqrt(3)) - pi/6) / (2*sqrt(3))."""

    name: str = "atan"

    def __init__(self, alpha: float = 1.0):
        self.alpha = float(alpha)
        self._sqrt3 = float(np.sqrt(3.0))

    def evaluate(self, t: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        t_arr = np.asarray(t, dtype=np.float64)
        num = 2.0 * np.arctan((1.0 + 4.0 * t_arr) / self._sqrt3) - (np.pi / 6.0)
        val = num / (2.0 * self._sqrt3)
        return float(val) if np.ndim(t) == 0 else val

    def derivative(self, t: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        t_arr = np.asarray(t, dtype=np.float64)
        val = 1.0 / (1.0 + 2.0 * t_arr + 4.0 * (t_arr ** 2))
        return float(val) if np.ndim(t) == 0 else val


class LogPenalty(BasePenalty):
    """Logarithmic penalty: phi(t) = (1/gamma) * ln(1 + gamma * t)."""

    name: str = "log"

    def __init__(self, gamma: float = 5.0):
        if gamma <= 0:
            raise ValueError(f"gamma must be positive, got {gamma}")
        self.gamma = float(gamma)

    def evaluate(self, t: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        t_arr = np.asarray(t, dtype=np.float64)
        val = (1.0 / self.gamma) * np.log1p(self.gamma * t_arr)
        return float(val) if np.ndim(t) == 0 else val

    def derivative(self, t: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        t_arr = np.asarray(t, dtype=np.float64)
        val = 1.0 / (1.0 + self.gamma * t_arr)
        return float(val) if np.ndim(t) == 0 else val


class McpPenalty(BasePenalty):
    """Minimax Concave Penalty (MCP, Zhang 2010)."""

    name: str = "mcp"

    def __init__(self, gamma: float = 1.0):
        if gamma <= 0:
            raise ValueError(f"gamma must be positive, got {gamma}")
        self.gamma = float(gamma)

    def evaluate(self, t: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        t_arr = np.asarray(t, dtype=np.float64)
        val = np.where(
            t_arr <= self.gamma,
            t_arr - (t_arr ** 2) / (2.0 * self.gamma),
            0.5 * self.gamma
        )
        return float(val) if np.ndim(t) == 0 else val

    def derivative(self, t: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        t_arr = np.asarray(t, dtype=np.float64)
        val = np.maximum(0.0, 1.0 - t_arr / self.gamma)
        return float(val) if np.ndim(t) == 0 else val


class ScadPenalty(BasePenalty):
    """Smoothly Clipped Absolute Deviation (SCAD, Fan & Li 2001)."""

    name: str = "scad"

    def __init__(self, a: float = 3.7):
        if a <= 2.0:
            raise ValueError(f"Parameter 'a' must be > 2.0, got {a}")
        self.a = float(a)

    def evaluate(self, t: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        t_arr = np.asarray(t, dtype=np.float64)
        cond1 = t_arr <= 1.0
        cond2 = (t_arr > 1.0) & (t_arr <= self.a)
        val = np.where(
            cond1,
            t_arr,
            np.where(
                cond2,
                (2.0 * self.a * t_arr - t_arr ** 2 - 1.0) / (2.0 * (self.a - 1.0)),
                (self.a + 1.0) / 2.0
            )
        )
        return float(val) if np.ndim(t) == 0 else val

    def derivative(self, t: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        t_arr = np.asarray(t, dtype=np.float64)
        cond1 = t_arr <= 1.0
        cond2 = (t_arr > 1.0) & (t_arr <= self.a)
        val = np.where(
            cond1,
            1.0,
            np.where(
                cond2,
                (self.a - t_arr) / (self.a - 1.0),
                0.0
            )
        )
        return float(val) if np.ndim(t) == 0 else val


class GemanPenalty(BasePenalty):
    """Geman rational penalty: phi(t) = t / (1 + gamma * t)."""

    name: str = "geman"

    def __init__(self, gamma: float = 2.0):
        if gamma <= 0:
            raise ValueError(f"gamma must be positive, got {gamma}")
        self.gamma = float(gamma)

    def evaluate(self, t: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        t_arr = np.asarray(t, dtype=np.float64)
        val = t_arr / (1.0 + self.gamma * t_arr)
        return float(val) if np.ndim(t) == 0 else val

    def derivative(self, t: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        t_arr = np.asarray(t, dtype=np.float64)
        val = 1.0 / ((1.0 + self.gamma * t_arr) ** 2)
        return float(val) if np.ndim(t) == 0 else val


class FractionalPenalty(BasePenalty):
    """Fractional power penalty: phi(t) = ((1 + t/delta)^p - 1) / (p/delta)."""

    name: str = "fractional"

    def __init__(self, p: float = 0.5, delta: float = 1.0):
        if not (0.0 < p < 1.0):
            raise ValueError(f"p must be in (0, 1), got {p}")
        if delta <= 0.0:
            raise ValueError(f"delta must be positive, got {delta}")
        self.p = float(p)
        self.delta = float(delta)

    def evaluate(self, t: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        t_arr = np.asarray(t, dtype=np.float64)
        val = ((1.0 + t_arr / self.delta) ** self.p - 1.0) / (self.p / self.delta)
        return float(val) if np.ndim(t) == 0 else val

    def derivative(self, t: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        t_arr = np.asarray(t, dtype=np.float64)
        val = (1.0 + t_arr / self.delta) ** (self.p - 1.0)
        return float(val) if np.ndim(t) == 0 else val


class CappedL1Penalty(BasePenalty):
    """Capped L1 penalty: phi(t) = min(t, theta)."""

    name: str = "capped_l1"

    def __init__(self, theta: float = 0.5):
        if theta <= 0:
            raise ValueError(f"theta must be positive, got {theta}")
        self.theta = float(theta)

    def evaluate(self, t: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        t_arr = np.asarray(t, dtype=np.float64)
        val = np.minimum(t_arr, self.theta)
        return float(val) if np.ndim(t) == 0 else val

    def derivative(self, t: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        t_arr = np.asarray(t, dtype=np.float64)
        val = np.where(t_arr < self.theta, 1.0, 0.0)
        return float(val) if np.ndim(t) == 0 else val


PENALTY_REGISTRY: Dict[str, Type[BasePenalty]] = {
    "atan": AtanPenalty,
    "log": LogPenalty,
    "mcp": McpPenalty,
    "scad": ScadPenalty,
    "geman": GemanPenalty,
    "fractional": FractionalPenalty,
    "capped_l1": CappedL1Penalty,
}


def get_penalty(name: str, **kwargs) -> BasePenalty:
    """Factory to instantiate penalty by name."""
    clean_name = name.lower().strip().replace("-", "_")
    if clean_name not in PENALTY_REGISTRY:
        supported = list(PENALTY_REGISTRY.keys())
        raise KeyError(f"Unknown penalty '{name}'. Supported penalties: {supported}")
    cls = PENALTY_REGISTRY[clean_name]
    import inspect
    sig = inspect.signature(cls.__init__)
    valid_params = set(sig.parameters.keys()) - {"self"}
    filtered = {k: v for k, v in kwargs.items() if k in valid_params}
    return cls(**filtered)
