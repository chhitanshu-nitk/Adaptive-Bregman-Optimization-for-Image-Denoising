"""Evaluation metrics: MSE, RMSE, PSNR, and SSIM."""

import numpy as np
from skimage.metrics import structural_similarity as skimage_ssim


def compute_mse(clean: np.ndarray, denoised: np.ndarray) -> float:
    """Mean Squared Error (MSE)."""
    return float(np.mean((clean - denoised) ** 2))


def compute_rmse(clean: np.ndarray, denoised: np.ndarray) -> float:
    """Root Mean Squared Error (RMSE)."""
    return float(np.sqrt(compute_mse(clean, denoised)))


def compute_psnr(clean: np.ndarray, denoised: np.ndarray, data_range: float = 1.0) -> float:
    """Peak Signal-to-Noise Ratio (PSNR) in dB."""
    mse = compute_mse(clean, denoised)
    if mse <= 1e-15:
        return 100.0
    return float(10.0 * np.log10((data_range ** 2) / mse))


def compute_ssim(clean: np.ndarray, denoised: np.ndarray, data_range: float = 1.0) -> float:
    """Structural Similarity Index Measure (SSIM)."""
    return float(skimage_ssim(clean, denoised, data_range=data_range))