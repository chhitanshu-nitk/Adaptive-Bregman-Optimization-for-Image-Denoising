"""Grayscale image loading, normalization, and saving utilities."""

from pathlib import Path
from typing import Union
import cv2
import numpy as np


def load_grayscale_image(path: Union[str, Path]) -> np.ndarray:
    """Load image from disk and normalize to [0, 1] float64."""
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Image not found at: {file_path}")

    img = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Failed to read image at: {file_path}")

    return img.astype(np.float64) / 255.0


def save_image(image: np.ndarray, path: Union[str, Path]) -> None:
    """Clip and save [0, 1] float image as 8-bit grayscale PNG."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    clipped = np.clip(image, 0.0, 1.0)
    uint8_img = (clipped * 255.0 + 0.5).astype(np.uint8)
    cv2.imwrite(str(out_path), uint8_img)