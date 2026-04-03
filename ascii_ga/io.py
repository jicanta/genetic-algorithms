"""
Output helpers: convert genomes to text and save results to disk.
"""

from pathlib import Path

import numpy as np
from PIL import Image

from .render import render_genome


def genome_to_text(genome: np.ndarray, charset: str) -> str:
    """Convert a (rows, cols) genome to a plain-text ASCII art string."""
    rows, cols = genome.shape
    return "\n".join(
        "".join(charset[genome[r, c]] for c in range(cols))
        for r in range(rows)
    )


def save_result(
    genome: np.ndarray,
    charset: str,
    glyphs: np.ndarray,
    rows: int,
    cols: int,
    cell_h: int,
    cell_w: int,
    out_dir: Path,
    prefix: str = "best",
) -> tuple[Path, Path]:
    """
    Save the genome as both a .txt file and a rendered .png.

    The text and the image are always in sync because both are derived from
    the same genome — the text is the character grid, the image is that same
    grid rendered with the font.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    txt_path = out_dir / f"{prefix}.txt"
    png_path = out_dir / f"{prefix}.png"

    txt_path.write_text(genome_to_text(genome, charset), encoding="utf-8")

    rendered = render_genome(genome, glyphs, rows, cols, cell_h, cell_w)
    Image.fromarray(rendered, mode="L").save(png_path)

    return txt_path, png_path
