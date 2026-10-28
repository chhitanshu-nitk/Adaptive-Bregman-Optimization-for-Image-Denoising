"""Additive White Gaussian Noise (AWGN) generation."""

from typing import Optional, Tuple
import numpy as np


def add_gaussian_noise(
    image: np.ndarray,
    sigma: float,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Add zero-mean Gaussian noise with sigma scaled for [0, 255]."""
    sigma_norm = float(sigma) / 255.0
    rng = np.random.default_rng(seed)
    noise = rng.normal(loc=0.0, scale=sigma_norm, size=image.shape).astype(np.float64)
    noisy_image = np.clip(image + noise, 0.0, 1.0)
    return noisy_image, noise