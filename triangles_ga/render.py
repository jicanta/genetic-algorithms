"""
Rendering: genome → RGB image.

Each triangle is drawn onto a white RGBA canvas using alpha compositing,
so semi-transparent triangles blend naturally with each other and the background.
"""

import numpy as np
from PIL import Image, ImageDraw


def render_genome(genome: np.ndarray, img_w: int, img_h: int) -> np.ndarray:
    """
    Render a triangle genome to an RGB image.

    Args:
        genome:  float32 array of shape (N_triangles, 10).
                 Genes are normalized to [0, 1].
        img_w:   Canvas width in pixels.
        img_h:   Canvas height in pixels.

    Returns:
        float32 array of shape (img_h, img_w, 3), values in [0, 255].
    """
    canvas = Image.new("RGBA", (img_w, img_h), (255, 255, 255, 255))

    for triangle in genome:
        x1, y1, x2, y2, x3, y3, r, g, b, a = triangle

        vertices = [
            (float(x1 * img_w), float(y1 * img_h)),
            (float(x2 * img_w), float(y2 * img_h)),
            (float(x3 * img_w), float(y3 * img_h)),
        ]
        color = (
            int(r * 255),
            int(g * 255),
            int(b * 255),
            int(a * 255),
        )

        layer = Image.new("RGBA", (img_w, img_h), (0, 0, 0, 0))
        ImageDraw.Draw(layer).polygon(vertices, fill=color)
        canvas = Image.alpha_composite(canvas, layer)

    rgb = canvas.convert("RGB")
    return np.array(rgb, dtype=np.float32)
