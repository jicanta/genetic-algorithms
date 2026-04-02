"""
Fitness function for Exercise 1: ASCII Art

The fitness measures how similar the rendered ASCII grid is to the target image.

Key idea:
- Both images are grayscale numpy arrays of the same size
- We compute Mean Squared Error (MSE) between them
- fitness = 1 / (1 + MSE)  →  ranges (0, 1], higher is better

Why MSE?
- Simple and fast with numpy
- Penalizes large pixel differences more than small ones (squared)
- Normalized by image size so it's independent of resolution
"""

import numpy as np
from .renderer import render_ascii_grid


def compute_mse(rendered: np.ndarray, target: np.ndarray) -> float:
    """Mean Squared Error between two grayscale images (same size)."""
    return float(np.mean((rendered.astype(np.float32) - target.astype(np.float32)) ** 2))


def compute_fitness_normalized(rendered: np.ndarray, target: np.ndarray) -> float:
    """
    Normalized fitness: 1 - (MSE / max_possible_MSE)
    Max possible MSE = 255^2 (every pixel fully wrong)
    Result is in [0, 1] where 1 = perfect match, 0 = total opposite.
    This is more interpretable than 1/(1+MSE).
    """
    mse = compute_mse(rendered, target)
    return 1.0 - (mse / (255.0 ** 2))


def compute_fitness(genome: np.ndarray, grid_n: int, target: np.ndarray, cell_size: int = 8) -> float:
    """
    Full fitness pipeline:
    1. Render genome → PIL image → numpy array
    2. Compute MSE vs target
    3. Return 1 / (1 + MSE)

    Args:
        genome: flat int array, length grid_n * grid_n
        grid_n: ASCII grid size
        target: preprocessed target image (numpy array, same pixel size as render)
        cell_size: pixels per ASCII cell

    Returns:
        float in (0, 1], higher = better approximation
    """
    rendered_pil = render_ascii_grid(genome, grid_n, cell_size)
    rendered = np.array(rendered_pil, dtype=np.float32)
    return compute_fitness_normalized(rendered, target)
