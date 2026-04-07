#!/usr/bin/env python3
"""
ASCII Art via optimal separable grayscale matching — entry point.

Usage:
    python ascii_ga/main_greedy.py input.jpg
    python ascii_ga/main_greedy.py input.jpg --cols 100
    python ascii_ga/main_greedy.py input.jpg --font /path/to/Mono.ttf --font-size 12
"""

import sys
from pathlib import Path

# Allow running as a script from the project root: python ascii_ga/main_greedy.py ...
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse

import numpy as np

from ascii_ga.font import build_glyph_cache, get_cell_size, load_font
from ascii_ga.fitness import compute_fitness
from ascii_ga.image import load_target
from ascii_ga.io import save_result


PRINTABLE_ASCII = "".join(chr(i) for i in range(32, 127))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ASCII Art via optimal separable grayscale matching",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("image", help="Input image path")
    parser.add_argument("--cols", type=int, default=80, help="ASCII columns (default: 80)")
    parser.add_argument("--font", dest="font_path", default=None, help="Path to TTF monospace font")
    parser.add_argument("--font-size", type=int, default=12, help="Font size in points (default: 12)")
    parser.add_argument("--output", default="output/ascii_greedy", help="Output directory (default: output/ascii_greedy)")
    parser.add_argument(
        "--charset",
        default=PRINTABLE_ASCII,
        help="Character set (default: all printable ASCII characters)",
    )
    parser.add_argument("--char-aspect", type=float, default=None, help="cell_w/cell_h override (default: auto)")
    args = parser.parse_args()

    if args.cols < 1:
        raise ValueError(f"cols must be >= 1, got {args.cols}")
    if not args.charset:
        raise ValueError("charset must contain at least one character")

    return args


def pairwise_sse(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    lhs_sq = np.sum(lhs * lhs, axis=1, keepdims=True)
    rhs_sq = np.sum(rhs * rhs, axis=1, keepdims=True).T
    cross = lhs @ rhs.T
    return lhs_sq + rhs_sq - 2.0 * cross


def grayscale_match_genome(
    target: np.ndarray,
    glyphs: np.ndarray,
    rows: int,
    cols: int,
    cell_h: int,
    cell_w: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Choose, for each target block, the glyph tile with the lowest grayscale SSE.

    Under a separable objective where the final render is just a tiling of
    independent glyph cells and total error is the sum of per-cell squared
    errors, this solves the global optimum exactly.
    """
    block_tiles = (
        target.reshape(rows, cell_h, cols, cell_w)
        .transpose(0, 2, 1, 3)
    )
    block_vectors = block_tiles.reshape(rows * cols, cell_h * cell_w).astype(np.float32)
    glyph_vectors = glyphs.reshape(glyphs.shape[0], cell_h * cell_w).astype(np.float32)

    score = pairwise_sse(block_vectors, glyph_vectors)
    best_idx = np.argmin(score, axis=1)
    cell_mse = pairwise_sse(block_vectors, glyph_vectors)[np.arange(rows * cols), best_idx] / float(cell_h * cell_w)

    return (
        best_idx.reshape(rows, cols).astype(np.int32),
        cell_mse.reshape(rows, cols),
    )


def main() -> None:
    args = parse_args()

    print("Loading font...")
    font = load_font(args.font_path, args.font_size)
    cell_w, cell_h = get_cell_size(font)
    print(f"  Cell size: {cell_w}x{cell_h} px")

    print(f"Building glyph cache  n_chars={len(args.charset)}")
    glyphs, _ = build_glyph_cache(args.charset, font, cell_w, cell_h)

    print(f"Loading target: {args.image}")
    target, rows, cols = load_target(args.image, args.cols, cell_w, cell_h, args.char_aspect)
    print(f"  Grid: {rows} rows x {cols} cols  |  Render: {cols * cell_w}x{rows * cell_h} px")

    print("Matching target blocks against all glyph tiles with minimum SSE...")
    genome, cell_mse = grayscale_match_genome(
        target,
        glyphs,
        rows,
        cols,
        cell_h,
        cell_w,
    )

    mse = compute_fitness(genome, glyphs, target, rows, cols, cell_h, cell_w)

    out_dir = Path(args.output)
    txt_path, png_path = save_result(
        genome,
        args.charset,
        glyphs,
        rows,
        cols,
        cell_h,
        cell_w,
        out_dir,
        prefix="best",
    )

    print(f"\nFinal MSE: {mse:.2f}")
    print(f"Per-cell mean MSE: {cell_mse.mean():.2f}")
    print(f"Saved -> {txt_path}")
    print(f"Saved -> {png_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
