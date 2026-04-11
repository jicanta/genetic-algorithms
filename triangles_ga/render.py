"""
Rendering: genome → RGB image.

Two backends are available and can be chosen explicitly:

  skia  — draws all triangles on a single surface in one pass; no per-triangle
           image allocation, no N-fold compositing overhead.  ~5–15x faster than PIL.
  pil   — creates one RGBA layer per triangle and alpha-composites; pure Python,
           no native dependencies, identical visual output to the Skia backend.
  auto  — (default) use Skia if available, fall back to PIL.

Both produce identical (visually) output: semi-transparent triangles alpha-composited
over a white background, in the order they appear in the genome.
"""

import numpy as np

try:
    import skia as _skia
    _HAVE_SKIA = True
except ImportError:
    _HAVE_SKIA = False

# Module-level backend selection.  Set via set_backend() before rendering.
# Options: "auto" | "skia" | "pil"
_BACKEND: str = "auto"


def set_backend(backend: str) -> None:
    """
    Set the rendering backend for this process.

    Args:
        backend: "skia" | "pil" | "auto"
                 "auto" uses Skia when available, PIL otherwise.
    Raises:
        ValueError: for unknown backend names.
        RuntimeError: when "skia" is requested but the skia package is not installed.
    """
    global _BACKEND
    if backend not in ("auto", "skia", "pil"):
        raise ValueError(f"Unknown renderer {backend!r}. Choose: auto | skia | pil")
    if backend == "skia" and not _HAVE_SKIA:
        raise RuntimeError(
            "Renderer 'skia' requested but the skia-python package is not installed. "
            "Install it with:  pip install skia-python"
        )
    _BACKEND = backend


# ── Skia backend ──────────────────────────────────────────────────────────────

# Per-process surface cache: keyed by (img_w, img_h) so one surface is reused
# across all renders in the same worker, avoiding repeated 600KB allocations.
_skia_surfaces: dict = {}


def _render_skia(genome: np.ndarray, img_w: int, img_h: int) -> np.ndarray:
    """
    Render using Skia: single surface, all triangles drawn in one sweep.

    A surface is created once per (width, height) per process and reused for
    every subsequent render, avoiding repeated large memory allocations.
    Skia handles alpha compositing (SrcOver) internally with no intermediate
    allocations, which is the core speed advantage over PIL.
    """
    key = (img_w, img_h)
    if key not in _skia_surfaces:
        _skia_surfaces[key] = _skia.Surface(img_w, img_h)
    surface = _skia_surfaces[key]

    paint = _skia.Paint(AntiAlias=True, BlendMode=_skia.BlendMode.kSrcOver)

    with surface as canvas:
        canvas.clear(_skia.ColorWHITE)

        for triangle in genome:
            x1, y1, x2, y2, x3, y3, r, g, b, a = triangle

            paint.setARGB(
                int(a * 255),
                int(r * 255),
                int(g * 255),
                int(b * 255),
            )

            path = _skia.Path()
            path.moveTo(x1 * img_w, y1 * img_h)
            path.lineTo(x2 * img_w, y2 * img_h)
            path.lineTo(x3 * img_w, y3 * img_h)
            path.close()
            canvas.drawPath(path, paint)

    # Skia snapshot is BGRA uint8 — convert to RGB float32
    bgra = np.array(surface.makeImageSnapshot())   # (H, W, 4) uint8
    rgb  = bgra[:, :, :3][:, :, ::-1].astype(np.float32)  # BGR→RGB
    return rgb


# ── PIL backend (fallback) ────────────────────────────────────────────────────

def _render_pil(genome: np.ndarray, img_w: int, img_h: int) -> np.ndarray:
    """PIL renderer: creates one RGBA layer per triangle and alpha-composites."""
    from PIL import Image, ImageDraw

    canvas = Image.new("RGBA", (img_w, img_h), (255, 255, 255, 255))

    for triangle in genome:
        x1, y1, x2, y2, x3, y3, r, g, b, a = triangle

        vertices = [
            (float(x1 * img_w), float(y1 * img_h)),
            (float(x2 * img_w), float(y2 * img_h)),
            (float(x3 * img_w), float(y3 * img_h)),
        ]
        color = (int(r * 255), int(g * 255), int(b * 255), int(a * 255))

        layer = Image.new("RGBA", (img_w, img_h), (0, 0, 0, 0))
        ImageDraw.Draw(layer).polygon(vertices, fill=color)
        canvas = Image.alpha_composite(canvas, layer)

    return np.array(canvas.convert("RGB"), dtype=np.float32)


# ── Public API ────────────────────────────────────────────────────────────────

def render_genome(genome: np.ndarray, img_w: int, img_h: int) -> np.ndarray:
    """
    Render a triangle genome to an RGB image.

    The backend is controlled by the module-level ``_BACKEND`` variable, which
    can be set with :func:`set_backend`.  Use ``"skia"`` for maximum speed,
    ``"pil"`` for portability, or ``"auto"`` (default) to let the module choose.

    Args:
        genome:  float32 array of shape (N_triangles, 10).
                 Genes are normalized to [0, 1].
        img_w:   Canvas width in pixels.
        img_h:   Canvas height in pixels.

    Returns:
        float32 array of shape (img_h, img_w, 3), values in [0, 255].
    """
    if _BACKEND == "skia" or (_BACKEND == "auto" and _HAVE_SKIA):
        return _render_skia(genome, img_w, img_h)
    return _render_pil(genome, img_w, img_h)
