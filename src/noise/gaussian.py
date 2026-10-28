"""Module for reproducible zero-mean Additive White Gaussian Noise (AWGN)."""

from typing import Optional, Tuple
import numpy as np


def add_gaussian_noise(
    image: np.ndarray,
    sigma: float,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Add reproducible zero-mean Gaussian noise to an image scaled in [0, 1].

    Args:
        image: Clean 2D float64 image array with range in [0, 1].
        sigma: Noise standard deviation scaled for [0, 255] (e.g., 15, 25, 50).
        seed: Optional integer random seed for reproducibility.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - noisy_image: Image corrupted by Gaussian noise, clipped to [0, 1].
            - noise: The additive noise field scaled to [0, 1] domain.
    """
    # Normalize sigma from [0, 255] to [0, 1]
    sigma_norm = float(sigma) / 255.0

    rng = np.random.default_rng(seed)
    noise = rng.normal(loc=0.0, scale=sigma_norm, size=image.shape).astype(np.float64)

    noisy_image = np.clip(image + noise, 0.0, 1.0)
    return noisy_image, noise