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
