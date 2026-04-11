"""
Genome representation for the triangle GA.

Each individual is a float32 array of shape (N_triangles, 10):
    [x1, y1, x2, y2, x3, y3, r, g, b, a]

All values are normalized to [0.0, 1.0]. Denormalization happens at render time:
    - x_i *= img_width,  y_i *= img_height
    - r, g, b, a *= 255  (PIL expects uint8)
"""

import numpy as np


GENES_PER_TRIANGLE = 10  # 6 coords + 4 color channels


def random_genome(n_triangles: int, rng: np.random.Generator) -> np.ndarray:
    """Return a random genome with all genes uniformly sampled from [0, 1]."""
    return rng.random((n_triangles, GENES_PER_TRIANGLE)).astype(np.float32)


def color_sampled_genome(
    n_triangles: int,
    rng: np.random.Generator,
    target: np.ndarray,
    img_w: int,
    img_h: int,
) -> np.ndarray:
    """
    Return a genome with random geometry but colors sampled from the target image.

    For each triangle, the centroid of its (random) vertices is computed and the
    target pixel at that location is used as the triangle's color.  This gives the
    GA a head-start: shapes are still random but colors are already plausible,
    which typically yields a much lower initial MSE than fully random initialization.

    Alpha is kept random so the GA still has to learn opacity from scratch.

    Args:
        n_triangles: Number of triangles.
        rng:         NumPy random generator.
        target:      float32 (img_h, img_w, 3) target image, values in [0, 255].
        img_w:       Canvas width in pixels.
        img_h:       Canvas height in pixels.

    Returns:
        float32 array of shape (n_triangles, 10).
    """
    genome = rng.random((n_triangles, GENES_PER_TRIANGLE)).astype(np.float32)

    # Columns: x1 y1 x2 y2 x3 y3 r g b a
    xs = genome[:, [0, 2, 4]]  # (N, 3) — normalized x coords of the 3 vertices
    ys = genome[:, [1, 3, 5]]  # (N, 3) — normalized y coords

    cx = xs.mean(axis=1)  # centroid x, normalized [0, 1]
    cy = ys.mean(axis=1)  # centroid y, normalized [0, 1]

    # Convert to pixel indices, clamped to valid range
    px = np.clip((cx * img_w).astype(int), 0, img_w - 1)
    py = np.clip((cy * img_h).astype(int), 0, img_h - 1)

    sampled_rgb = target[py, px] / 255.0   # (N, 3), normalized to [0, 1]
    genome[:, 6:9] = sampled_rgb.astype(np.float32)

    return genome
