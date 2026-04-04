"""
Saving results: rendered PNG + genome JSON.

The JSON stores triangles in absolute pixel coordinates and uint8 colors,
so results can be reproduced or inspected without re-running the GA.
"""

import json
from pathlib import Path

import numpy as np
from PIL import Image

from .render import render_genome


def save_result(
    genome: np.ndarray,
    img_w: int,
    img_h: int,
    out_dir: Path,
    prefix: str,
) -> tuple[Path, Path]:
    """
    Save the rendered image as PNG and the genome as JSON.

    Args:
        genome:   float32 (N_triangles, 10).
        img_w:    Canvas width in pixels.
        img_h:    Canvas height in pixels.
        out_dir:  Directory to write files into (created if missing).
        prefix:   Filename prefix, e.g. "best" or "gen_00050".

    Returns:
        (json_path, png_path)
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    png_path = out_dir / f"{prefix}.png"
    rendered = render_genome(genome, img_w, img_h)
    Image.fromarray(rendered.astype(np.uint8), "RGB").save(png_path)

    json_path = out_dir / f"{prefix}.json"
    triangles = []
    for tri in genome:
        x1, y1, x2, y2, x3, y3, r, g, b, a = tri.tolist()
        triangles.append({
            "vertices": [
                [round(x1 * img_w, 2), round(y1 * img_h, 2)],
                [round(x2 * img_w, 2), round(y2 * img_h, 2)],
                [round(x3 * img_w, 2), round(y3 * img_h, 2)],
            ],
            "color": {
                "r": int(r * 255),
                "g": int(g * 255),
                "b": int(b * 255),
                "a": int(a * 255),
            },
        })
    with open(json_path, "w") as f:
        json.dump({"img_w": img_w, "img_h": img_h, "triangles": triangles}, f, indent=2)

    return json_path, png_path
