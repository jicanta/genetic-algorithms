"""
Target image loading and preprocessing.

The target is resized to exactly match the render resolution so that
fitness comparison is pixel-perfect without any interpolation at eval time.
"""

from typing import Optional

import numpy as np
from PIL import Image


def load_target(
    image_path: str,
    cols: int,
    cell_w: int,
    cell_h: int,
    char_aspect: Optional[float],
) -> tuple[np.ndarray, int, int]:
    """
    Load image as grayscale, derive grid dimensions, resize to render resolution.

    char_aspect (cell_w / cell_h) corrects for the fact that character cells
    are taller than wide — without it the ASCII canvas looks vertically squished.
    Computed automatically from font metrics when not provided.

    Returns:
        target: float32 array of shape (rows*cell_h, cols*cell_w)
        rows:   number of character rows in the grid
        cols:   number of character columns (same as input)
    """
    img = Image.open(image_path).convert("L")
    iw, ih = img.size
    aspect = char_aspect if char_aspect is not None else (cell_w / cell_h)
    rows = max(1, round(ih / iw * cols * aspect))
    img = img.resize((cols * cell_w, rows * cell_h), Image.LANCZOS)
    return np.array(img, dtype=np.float32), rows, cols
