"""
Font loading and glyph cache construction.

The glyph cache pre-renders every character in the charset as a fixed-size
grayscale tile. This is done once at startup so that rendering genomes during
evolution is just a numpy indexing operation (no PIL calls in the hot loop).
"""

from typing import Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont


_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
    "/usr/share/fonts/truetype/freefont/FreeMono.ttf",
    "/usr/share/fonts/truetype/ubuntu/UbuntuMono-R.ttf",
]


def load_font(font_path: Optional[str], font_size: int) -> ImageFont.FreeTypeFont:
    if font_path:
        return ImageFont.truetype(font_path, font_size)
    for path in _FONT_CANDIDATES:
        try:
            return ImageFont.truetype(path, font_size)
        except OSError:
            continue
    raise RuntimeError(
        "No monospace font found. Install DejaVu fonts or pass --font /path/to/Mono.ttf"
    )


def get_cell_size(font: ImageFont.FreeTypeFont) -> tuple[int, int]:
    """Return (cell_w, cell_h) in pixels derived from font metrics."""
    ascent, descent = font.getmetrics()
    cell_h = ascent + descent
    cell_w = max(1, round(font.getlength("M")))
    return cell_w, cell_h


def build_glyph_cache(
    charset: str, font: ImageFont.FreeTypeFont, cell_w: int, cell_h: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Render every character in charset to a (cell_h, cell_w) grayscale tile.

    Each character is drawn black on white, centered in its cell.

    Returns:
        glyphs:   uint8 array (N, cell_h, cell_w)
        darkness: float array (N,) — ink coverage [0=white, 1=black]
    """
    n = len(charset)
    glyphs = np.zeros((n, cell_h, cell_w), dtype=np.uint8)

    for i, char in enumerate(charset):
        tile = Image.new("L", (cell_w, cell_h), color=255)
        draw = ImageDraw.Draw(tile)
        bbox = draw.textbbox((0, 0), char, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        x = (cell_w - tw) // 2 - bbox[0]
        y = (cell_h - th) // 2 - bbox[1]
        draw.text((x, y), char, fill=0, font=font)
        glyphs[i] = np.array(tile)

    darkness = 1.0 - glyphs.mean(axis=(1, 2)) / 255.0
    return glyphs, darkness
