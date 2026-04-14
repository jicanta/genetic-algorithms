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

try:
    from .render_numba import render_genome_numba as _render_numba
    _HAVE_NUMBA = True
except (ImportError, ModuleNotFoundError):
    _HAVE_NUMBA = False

# Module-level backend selection.  Set via set_backend() before rendering.
# Options: "auto" | "skia" | "pil"
_BACKEND: str = "auto"
_SHAPE:   str = "triangle"


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
    if backend not in ("auto", "skia", "pil", "numba"):
        raise ValueError(f"Unknown renderer {backend!r}. Choose: auto | skia | pil | numba")
    if backend == "skia" and not _HAVE_SKIA:
        raise RuntimeError(
            "Renderer 'skia' requested but the skia-python package is not installed. "
            "Install it with:  pip install skia-python"
        )
    if backend == "numba" and not _HAVE_NUMBA:
        raise RuntimeError(
            "Renderer 'numba' requested but the numba package is not installed. "
            "Install it with:  pip install numba"
        )
    _BACKEND = backend


def set_shape(shape: str) -> None:
    """
    Set the shape type rendered by this process.

    Args:
        shape: "triangle" | "oval"
    """
    global _SHAPE
    if shape not in ("triangle", "oval"):
        raise ValueError(f"Unknown shape {shape!r}. Choose: triangle | oval")
    _SHAPE = shape


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


# ── Oval renderers ───────────────────────────────────────────────────────────

def _render_skia_ovals(genome: np.ndarray, img_w: int, img_h: int) -> np.ndarray:
    """Render axis-aligned ovals using Skia. genome shape: (N, 8) [cx,cy,rx,ry,r,g,b,a]."""
    key = (img_w, img_h)
    if key not in _skia_surfaces:
        _skia_surfaces[key] = _skia.Surface(img_w, img_h)
    surface = _skia_surfaces[key]

    paint = _skia.Paint(AntiAlias=True, BlendMode=_skia.BlendMode.kSrcOver)

    with surface as canvas:
        canvas.clear(_skia.ColorWHITE)

        for oval in genome:
            cx, cy, rx, ry, r, g, b, a = oval

            paint.setARGB(
                int(a * 255),
                int(r * 255),
                int(g * 255),
                int(b * 255),
            )

            rect = _skia.Rect.MakeLTRB(
                (cx - rx) * img_w, (cy - ry) * img_h,
                (cx + rx) * img_w, (cy + ry) * img_h,
            )
            canvas.drawOval(rect, paint)

    bgra = np.array(surface.makeImageSnapshot())
    rgb  = bgra[:, :, :3][:, :, ::-1].astype(np.float32)
    return rgb


def _render_pil_ovals(genome: np.ndarray, img_w: int, img_h: int) -> np.ndarray:
    """Render axis-aligned ovals using PIL. genome shape: (N, 8) [cx,cy,rx,ry,r,g,b,a]."""
    from PIL import Image, ImageDraw

    canvas = Image.new("RGBA", (img_w, img_h), (255, 255, 255, 255))

    for oval in genome:
        cx, cy, rx, ry, r, g, b, a = oval

        bbox = [
            float((cx - rx) * img_w), float((cy - ry) * img_h),
            float((cx + rx) * img_w), float((cy + ry) * img_h),
        ]
        color = (int(r * 255), int(g * 255), int(b * 255), int(a * 255))

        layer = Image.new("RGBA", (img_w, img_h), (0, 0, 0, 0))
        ImageDraw.Draw(layer).ellipse(bbox, fill=color)
        canvas = Image.alpha_composite(canvas, layer)

    return np.array(canvas.convert("RGB"), dtype=np.float32)


# ── Public API ────────────────────────────────────────────────────────────────

def render_genome(genome: np.ndarray, img_w: int, img_h: int) -> np.ndarray:
    """
    Render a genome (triangles or ovals) to an RGB image.

    The backend is controlled by the module-level ``_BACKEND`` variable, which
    can be set with :func:`set_backend`.  Use ``"skia"`` for maximum speed,
    ``"pil"`` for portability, or ``"auto"`` (default) to let the module choose.

    Args:
        genome:  float32 array of shape (N, 10) for triangles or (N, 8) for ovals.
                 Genes are normalized to [0, 1].
        img_w:   Canvas width in pixels.
        img_h:   Canvas height in pixels.

    Returns:
        float32 array of shape (img_h, img_w, 3), values in [0, 255].
    """
    if _BACKEND == "numba":
        # Numba rasterizer — triangles only (ovals fall through to Skia/PIL)
        if _SHAPE == "triangle":
            return _render_numba(genome, img_w, img_h)
    use_skia = _BACKEND == "skia" or (_BACKEND == "auto" and _HAVE_SKIA)
    if _SHAPE == "oval":
        if use_skia:
            return _render_skia_ovals(genome, img_w, img_h)
        return _render_pil_ovals(genome, img_w, img_h)
    if use_skia:
        return _render_skia(genome, img_w, img_h)
    return _render_pil(genome, img_w, img_h)
