"""Evaluation metrics for image reconstruction: MSE, RMSE, PSNR, and SSIM."""

import numpy as np
from skimage.metrics import structural_similarity as skimage_ssim


def compute_mse(clean: np.ndarray, denoised: np.ndarray) -> float:
    """Calculate Mean Squared Error (MSE) between two [0, 1] images."""
    return float(np.mean((clean - denoised) ** 2))


def compute_rmse(clean: np.ndarray, denoised: np.ndarray) -> float:
    """Calculate Root Mean Squared Error (RMSE) between two [0, 1] images."""
    return float(np.sqrt(compute_mse(clean, denoised)))


def compute_psnr(clean: np.ndarray, denoised: np.ndarray, data_range: float = 1.0) -> float:
    """Calculate Peak Signal-to-Noise Ratio (PSNR) in decibels (dB).

    Args:
        clean: Ground truth image in [0, 1].
        denoised: Reconstructed image in [0, 1].
        data_range: Maximum dynamic range (1.0 for normalized images).

    Returns:
        float: PSNR in dB.
    """
    mse = compute_mse(clean, denoised)
    if mse <= 1e-15:
        return 100.0
    return float(10.0 * np.log10((data_range ** 2) / mse))


def compute_ssim(clean: np.ndarray, denoised: np.ndarray, data_range: float = 1.0) -> float:
    """Calculate Structural Similarity Index Measure (SSIM).

    Args:
        clean: Ground truth image in [0, 1].
        denoised: Reconstructed image in [0, 1].
        data_range: Maximum dynamic range (1.0 for normalized images).

    Returns:
        float: SSIM index in [-1.0, 1.0].
    """
    return float(skimage_ssim(clean, denoised, data_range=data_range))