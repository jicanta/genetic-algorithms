"""
Genome representation for the shape GA.

Triangle genome — float32 array of shape (N, 10):
    [x1, y1, x2, y2, x3, y3, r, g, b, a]

Oval genome — float32 array of shape (N, 8):
    [cx, cy, rx, ry, r, g, b, a]
    cx, cy: center (normalized)
    rx, ry: horizontal and vertical radii (normalized relative to canvas size)

All values are normalized to [0.0, 1.0]. Denormalization happens at render time.
"""

import numpy as np


GENES_PER_TRIANGLE = 10   # 6 vertex coords + r, g, b, a
GENES_PER_OVAL     = 8    # cx, cy, rx, ry + r, g, b, a
OVAL_MIN_RADIUS    = 0.01
OVAL_RADIUS_SPAN   = 0.49
HIGHLIGHT_FRACTION = 0.35


def genes_per_shape(shape: str) -> int:
    if shape == "oval":
        return GENES_PER_OVAL
    return GENES_PER_TRIANGLE


def random_genome(n_triangles: int, rng: np.random.Generator) -> np.ndarray:
    """Return a random genome with all genes uniformly sampled from [0, 1]."""
    return rng.random((n_triangles, GENES_PER_TRIANGLE)).astype(np.float32)


def _highlight_sample_count(n_shapes: int) -> int:
    return max(1, int(round(n_shapes * HIGHLIGHT_FRACTION)))


def _sample_highlight_pixels(
    target: np.ndarray,
    rng: np.random.Generator,
    n_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample target pixels biased toward bright warm highlights."""
    rgb = target.reshape(-1, 3) / 255.0
    max_c = rgb.max(axis=1)
    min_c = rgb.min(axis=1)
    saturation = max_c - min_c
    warm = np.maximum(0.0, np.minimum(rgb[:, 0], rgb[:, 1]) - rgb[:, 2])
    weights = (max_c ** 1.5) * saturation + 4.0 * warm * max_c

    if not np.isfinite(weights).all() or float(weights.sum()) <= 0.0:
        flat_idx = rng.integers(0, target.shape[0] * target.shape[1], size=n_samples)
    else:
        probs = weights / weights.sum()
        flat_idx = rng.choice(target.shape[0] * target.shape[1], size=n_samples, replace=True, p=probs)

    py = flat_idx // target.shape[1]
    px = flat_idx % target.shape[1]
    return px.astype(np.int32), py.astype(np.int32)


def _seed_triangle_highlights(
    genome: np.ndarray,
    rng: np.random.Generator,
    target: np.ndarray,
    img_w: int,
    img_h: int,
) -> None:
    n = _highlight_sample_count(genome.shape[0])
    px, py = _sample_highlight_pixels(target, rng, n)
    cx = (px + 0.5) / img_w
    cy = (py + 0.5) / img_h

    angles = rng.random((n, 3), dtype=np.float32) * (2.0 * np.pi)
    radii = rng.uniform(0.02, 0.12, size=(n, 3)).astype(np.float32)
    xs = np.clip(cx[:, None] + np.cos(angles) * radii, 0.0, 1.0)
    ys = np.clip(cy[:, None] + np.sin(angles) * radii, 0.0, 1.0)

    genome[:n, [0, 2, 4]] = xs
    genome[:n, [1, 3, 5]] = ys
    genome[:n, 6:9] = (target[py, px] / 255.0).astype(np.float32)
    genome[:n, 9] = rng.uniform(0.55, 1.00, size=n).astype(np.float32)


def _seed_oval_highlights(
    genome: np.ndarray,
    rng: np.random.Generator,
    target: np.ndarray,
    img_w: int,
    img_h: int,
) -> None:
    n = _highlight_sample_count(genome.shape[0])
    px, py = _sample_highlight_pixels(target, rng, n)
    genome[:n, 0] = (px + 0.5) / img_w
    genome[:n, 1] = (py + 0.5) / img_h
    genome[:n, 2:4] = rng.uniform(0.02, 0.12, size=(n, 2)).astype(np.float32)
    genome[:n, 4:7] = (target[py, px] / 255.0).astype(np.float32)
    genome[:n, 7] = rng.uniform(0.55, 1.00, size=n).astype(np.float32)


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
    _seed_triangle_highlights(genome, rng, target, img_w, img_h)

    return genome


# ── Oval genomes ──────────────────────────────────────────────────────────────

def random_oval_genome(n_ovals: int, rng: np.random.Generator) -> np.ndarray:
    """Return a random oval genome.

    Centers/colors/alpha are uniform. Radii are biased toward smaller shapes
    with occasional larger ovals; fully uniform radii often cover the whole
    canvas and make the first generations visually muddy.
    """
    genome = rng.random((n_ovals, GENES_PER_OVAL)).astype(np.float32)
    genome[:, 2:4] = (
        OVAL_MIN_RADIUS + OVAL_RADIUS_SPAN * rng.random((n_ovals, 2), dtype=np.float32) ** 2
    )
    return genome


def color_sampled_oval_genome(
    n_ovals: int,
    rng: np.random.Generator,
    target: np.ndarray,
    img_w: int,
    img_h: int,
) -> np.ndarray:
    """
    Return an oval genome with random geometry but colors sampled from the target.

    Each oval's center (cx, cy) is used directly to look up the target color,
    so the sampled color matches what's actually at the oval's position.

    Returns:
        float32 array of shape (n_ovals, 8).
    """
    genome = random_oval_genome(n_ovals, rng)

    # cx, cy are genes 0 and 1
    cx = genome[:, 0]
    cy = genome[:, 1]

    px = np.clip((cx * img_w).astype(int), 0, img_w - 1)
    py = np.clip((cy * img_h).astype(int), 0, img_h - 1)

    sampled_rgb = target[py, px] / 255.0   # (N, 3), normalized to [0, 1]
    genome[:, 4:7] = sampled_rgb.astype(np.float32)
    _seed_oval_highlights(genome, rng, target, img_w, img_h)

    return genome
