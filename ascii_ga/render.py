"""
Genome rendering.

Converts a (rows, cols) integer genome into a grayscale image by pasting
pre-rendered glyph tiles. Fully vectorized — no Python loops at render time.
"""

import numpy as np


def render_genome(
    genome: np.ndarray,   # (rows, cols) int — indices into charset
    glyphs: np.ndarray,   # (N_chars, cell_h, cell_w) uint8
    rows: int,
    cols: int,
    cell_h: int,
    cell_w: int,
) -> np.ndarray:          # (rows*cell_h, cols*cell_w) uint8
    """
    Build the rendered image by indexing the glyph cache and reshaping.

    glyphs[genome]            → (rows, cols, cell_h, cell_w)
    .transpose(0, 2, 1, 3)   → (rows, cell_h, cols, cell_w)
    .reshape(...)             → (rows*cell_h, cols*cell_w)
    """
    return glyphs[genome].transpose(0, 2, 1, 3).reshape(rows * cell_h, cols * cell_w)
