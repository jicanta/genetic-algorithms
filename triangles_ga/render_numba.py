"""
Numba JIT-compiled triangle rasterizer.

Draws all triangles onto a single pre-allocated canvas with in-place alpha
compositing. No intermediate image allocations — one pass over the genome.

The JIT-compiled core (_rasterize) runs as native machine code after the first
call; subsequent calls skip Python overhead entirely.
"""

import numpy as np
import numba


@numba.njit(cache=True)
def _edge(x0, y0, x1, y1, px, py):
    """Signed edge function: positive if (px,py) is on the left side of edge (x0,y0)→(x1,y1)."""
    return (x1 - x0) * (py - y0) - (y1 - y0) * (px - x0)


@numba.njit(cache=True)
def _rasterize(canvas, genome, img_w, img_h):
    """
    Rasterize all triangles in *genome* onto *canvas* with alpha compositing.

    canvas:  float32 (img_h, img_w, 3), pre-filled with white (255).
    genome:  float32 (N, 10) — [x1,y1,x2,y2,x3,y3,r,g,b,a], all in [0,1].
    """
    n_triangles = genome.shape[0]

    for t in range(n_triangles):
        # Denormalize vertices to pixel coords
        ax = genome[t, 0] * img_w
        ay = genome[t, 1] * img_h
        bx = genome[t, 2] * img_w
        by = genome[t, 3] * img_h
        cx = genome[t, 4] * img_w
        cy = genome[t, 5] * img_h

        # Color (0–255) and alpha (0–1)
        r = genome[t, 6] * 255.0
        g = genome[t, 7] * 255.0
        b = genome[t, 8] * 255.0
        a = genome[t, 9]

        # Bounding box (clamp to canvas)
        min_x = max(0, int(min(ax, bx, cx)))
        max_x = min(img_w - 1, int(max(ax, bx, cx)))
        min_y = max(0, int(min(ay, by, cy)))
        max_y = min(img_h - 1, int(max(ay, by, cy)))

        # Ensure consistent winding (CCW).  If the signed area is negative,
        # swap two vertices so all edge tests use the same sign convention.
        area = _edge(ax, ay, bx, by, cx, cy)
        if area < 0.0:
            ax, bx = bx, ax
            ay, by = by, ay

        inv_a = 1.0 - a

        for py in range(min_y, max_y + 1):
            for px in range(min_x, max_x + 1):
                ppx = px + 0.5
                ppy = py + 0.5
                w0 = _edge(ax, ay, bx, by, ppx, ppy)
                w1 = _edge(bx, by, cx, cy, ppx, ppy)
                w2 = _edge(cx, cy, ax, ay, ppx, ppy)

                if w0 >= 0.0 and w1 >= 0.0 and w2 >= 0.0:
                    canvas[py, px, 0] = canvas[py, px, 0] * inv_a + r * a
                    canvas[py, px, 1] = canvas[py, px, 1] * inv_a + g * a
                    canvas[py, px, 2] = canvas[py, px, 2] * inv_a + b * a


def render_genome_numba(genome: np.ndarray, img_w: int, img_h: int) -> np.ndarray:
    """
    Render a triangle genome using the Numba JIT rasterizer.

    Args:
        genome:  float32 (N, 10).
        img_w:   Canvas width in pixels.
        img_h:   Canvas height in pixels.

    Returns:
        float32 (img_h, img_w, 3), values in [0, 255].
    """
    canvas = np.full((img_h, img_w, 3), 255.0, dtype=np.float32)
    _rasterize(canvas, genome, img_w, img_h)
    return canvas
