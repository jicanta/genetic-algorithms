"""
Renderer for Exercise 1: ASCII Art

Converts a NxN grid of ASCII character indices into a PIL image,
so we can compare it pixel-by-pixel against the target image.

The charset is ordered from lightest (space) to darkest (@).
Each character maps to a visual "density" — this ordering is the key
insight that lets the GA evolve toward the right chars.
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Charset ordered light → dark by visual pixel coverage.
# Keeps the structural chars (/ \ | ( ) - _) and fills in gradations
# for enough brightness steps to accurately represent the image.
CHARSET = " .'`,:;-_()|\\/!lIrc+*%#@"
N_CHARS = len(CHARSET)  # 25


def indices_to_chars(genome: np.ndarray, grid_n: int) -> list[list[str]]:
    """Convert flat genome array to 2D grid of characters."""
    indices = genome.reshape(grid_n, grid_n)
    return [[CHARSET[i] for i in row] for row in indices]


def render_ascii_grid(genome: np.ndarray, grid_n: int, cell_size: int = 8) -> Image.Image:
    """
    Render an ASCII grid to a grayscale PIL image.

    Args:
        genome: flat array of ints in [0, N_CHARS-1], length = grid_n * grid_n
        grid_n: number of rows/cols in the grid
        cell_size: pixel size of each cell (square)

    Returns:
        Grayscale PIL Image of size (grid_n * cell_size, grid_n * cell_size)
    """
    img_size = grid_n * cell_size
    img = Image.new("L", (img_size, img_size), color=255)  # white background
    draw = ImageDraw.Draw(img)

    grid = indices_to_chars(genome, grid_n)

    # Map each char to a brightness value (0=black, 255=white)
    # We approximate: space=255 (white), @=0 (black), linear in between
    for row in range(grid_n):
        for col in range(grid_n):
            char = grid[row][col]
            char_idx = CHARSET.index(char)
            # Brightness: index 0 (space) → 255, index 9 (@) → 0
            brightness = int(255 * (1 - char_idx / (N_CHARS - 1)))

            x0 = col * cell_size
            y0 = row * cell_size
            x1 = x0 + cell_size
            y1 = y0 + cell_size
            draw.rectangle([x0, y0, x1, y1], fill=brightness)

    return img


def preprocess_target(image_path: str, grid_n: int, cell_size: int = 8) -> np.ndarray:
    """
    Load a target image and resize it to match the ASCII grid render size.

    Returns a numpy array of shape (img_size, img_size) with values in [0, 255].
    """
    img_size = grid_n * cell_size
    img = Image.open(image_path).convert("L")  # grayscale
    img = img.resize((img_size, img_size), Image.LANCZOS)
    return np.array(img, dtype=np.float32)
