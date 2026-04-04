"""
Fitness function: MSE between rendered genome and target image.

Lower MSE = better individual. A perfect replica would score 0.
"""

import numpy as np

from .render import render_genome


def compute_fitness(
    genome: np.ndarray,
    target: np.ndarray,
    img_w: int,
    img_h: int,
) -> float:
    """
    Compute the Mean Squared Error between the rendered genome and the target.

    Args:
        genome:  float32 (N_triangles, 10).
        target:  float32 (img_h, img_w, 3), values in [0, 255].
        img_w:   Canvas width.
        img_h:   Canvas height.

    Returns:
        MSE as a float. Lower is better.
    """
    rendered = render_genome(genome, img_w, img_h)
    return float(np.mean((rendered - target) ** 2))
