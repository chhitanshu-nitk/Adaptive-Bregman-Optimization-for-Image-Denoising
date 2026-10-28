"""Module for robust loading, normalization, and saving of grayscale images."""

from pathlib import Path
from typing import Union
import cv2
import numpy as np


def load_grayscale_image(path: Union[str, Path]) -> np.ndarray:
    """Load an image from disk in grayscale mode and normalize it to [0, 1].

    Args:
        path: Path to the image file.

    Returns:
        np.ndarray: 2D float64 array with pixel intensities scaled to [0.0, 1.0].

    Raises:
        FileNotFoundError: If the image file does not exist.
        ValueError: If the image fails to load.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Image not found at: {file_path}")

    img = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Failed to read image at: {file_path}")

    # Convert to standard float64 normalized to [0, 1]
    img_float = img.astype(np.float64) / 255.0
    return img_float


def save_image(image: np.ndarray, path: Union[str, Path]) -> None:
    """Clip and save a [0, 1] normalized float image as an 8-bit grayscale image.

    Args:
        image: 2D float array in range [0, 1].
        path: Destination file path.
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    clipped = np.clip(image, 0.0, 1.0)
    uint8_img = (clipped * 255.0 + 0.5).astype(np.uint8)
    cv2.imwrite(str(out_path), uint8_img)