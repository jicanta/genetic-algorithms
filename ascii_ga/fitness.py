"""
Fitness function.

Lower MSE = better candidate. Using MSE (not MAE) penalizes large per-pixel
deviations more heavily, which pushes the GA toward globally coherent shapes
rather than settling for uniformly mediocre ones.
"""

import numpy as np
from .render import render_genome


def compute_fitness(
    genome: np.ndarray,
    glyphs: np.ndarray,
    target: np.ndarray,
    rows: int,
    cols: int,
    cell_h: int,
    cell_w: int,
) -> float:
    """MSE between the rendered genome and the target image. Lower = better."""
    rendered = render_genome(genome, glyphs, rows, cols, cell_h, cell_w).astype(np.float32)
    return float(np.mean((rendered - target) ** 2))
