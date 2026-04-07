#!/usr/bin/env python3
"""
ASCII Art via hybrid greedy matching — entry point.

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
DIFFUSION_NEIGHBORS = (
    (0, 1, 7.0 / 16.0),
    (1, -1, 3.0 / 16.0),
    (1, 0, 5.0 / 16.0),
    (1, 1, 1.0 / 16.0),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ASCII Art via hybrid greedy matching",
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
    parser.add_argument("--tone-weight", type=float, default=1.0, help="Weight for grayscale block mismatch (default: 1.0)")
    parser.add_argument("--edge-weight", type=float, default=0.20, help="Weight for Sobel edge mismatch (default: 0.20)")
    parser.add_argument(
        "--neighbor-weight",
        type=float,
        default=0.10,
        help="Weight for left/up boundary consistency penalty (default: 0.10)",
    )
    parser.add_argument(
        "--dither-strength",
        type=float,
        default=0.15,
        help="Blockwise error-diffusion strength to future cells (default: 0.15)",
    )
    args = parser.parse_args()

    if args.cols < 1:
        raise ValueError(f"cols must be >= 1, got {args.cols}")
    if not args.charset:
        raise ValueError("charset must contain at least one character")
    if args.tone_weight < 0.0:
        raise ValueError(f"tone_weight must be >= 0, got {args.tone_weight}")
    if args.edge_weight < 0.0:
        raise ValueError(f"edge_weight must be >= 0, got {args.edge_weight}")
    if args.neighbor_weight < 0.0:
        raise ValueError(f"neighbor_weight must be >= 0, got {args.neighbor_weight}")
    if args.dither_strength < 0.0:
        raise ValueError(f"dither_strength must be >= 0, got {args.dither_strength}")

    return args


def pairwise_sse(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    lhs_sq = np.sum(lhs * lhs, axis=1, keepdims=True)
    rhs_sq = np.sum(rhs * rhs, axis=1, keepdims=True).T
    cross = lhs @ rhs.T
    return lhs_sq + rhs_sq - 2.0 * cross


def sobel_features(images: np.ndarray) -> np.ndarray:
    """
    Compute Sobel gx and gy for a batch of grayscale images.
    """
    padded = np.pad(images.astype(np.float32), ((0, 0), (1, 1), (1, 1)), mode="edge")
    gx = (
        -padded[:, :-2, :-2] + padded[:, :-2, 2:]
        - 2.0 * padded[:, 1:-1, :-2] + 2.0 * padded[:, 1:-1, 2:]
        - padded[:, 2:, :-2] + padded[:, 2:, 2:]
    )
    gy = (
        -padded[:, :-2, :-2] - 2.0 * padded[:, :-2, 1:-1] - padded[:, :-2, 2:]
        + padded[:, 2:, :-2] + 2.0 * padded[:, 2:, 1:-1] + padded[:, 2:, 2:]
    )
    gx *= 0.25
    gy *= 0.25
    return np.concatenate(
        [gx.reshape(images.shape[0], -1), gy.reshape(images.shape[0], -1)],
        axis=1,
    )


def normalized_mse(vector: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    """
    MSE between one flattened vector and every row in candidates.
    """
    diff = candidates - vector[None, :]
    return np.mean(diff * diff, axis=1)


def neighbor_penalty(
    r: int,
    c: int,
    genome: np.ndarray,
    glyphs: np.ndarray,
    target_tiles: np.ndarray,
) -> np.ndarray:
    """
    Penalize candidate boundaries that disagree with target discontinuities.

    The penalty is relative to the target jump across the block boundary, so a
    sharp edge in the source is allowed while a smooth region favors smoother
    glyph transitions.
    """
    penalties = np.zeros(glyphs.shape[0], dtype=np.float32)
    n_terms = 0

    if c > 0:
        left_idx = genome[r, c - 1]
        left_glyph = glyphs[left_idx]
        target_jump = target_tiles[r, c, :, 0] - target_tiles[r, c - 1, :, -1]
        candidate_jump = glyphs[:, :, 0] - left_glyph[:, -1][None, :]
        penalties += np.mean((candidate_jump - target_jump[None, :]) ** 2, axis=1)
        n_terms += 1

    if r > 0:
        up_idx = genome[r - 1, c]
        up_glyph = glyphs[up_idx]
        target_jump = target_tiles[r, c, 0, :] - target_tiles[r - 1, c, -1, :]
        candidate_jump = glyphs[:, 0, :] - up_glyph[-1, :][None, :]
        penalties += np.mean((candidate_jump - target_jump[None, :]) ** 2, axis=1)
        n_terms += 1

    if n_terms:
        penalties /= float(n_terms)

    return penalties


def hybrid_greedy_genome(
    target: np.ndarray,
    glyphs: np.ndarray,
    rows: int,
    cols: int,
    cell_h: int,
    cell_w: int,
    tone_weight: float,
    edge_weight: float,
    neighbor_weight: float,
    dither_strength: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Greedy ASCII matching with tone, edge, boundary-consistency, and dithering.
    """
    target_tiles = (
        target.reshape(rows, cell_h, cols, cell_w)
        .transpose(0, 2, 1, 3)
        .astype(np.float32)
    )
    working_tiles = target_tiles.copy()
    genome = np.zeros((rows, cols), dtype=np.int32)
    cell_mse = np.zeros((rows, cols), dtype=np.float32)

    glyphs_f = glyphs.astype(np.float32)
    glyph_vectors = glyphs_f.reshape(glyphs.shape[0], cell_h * cell_w)
    glyph_edges = sobel_features(glyphs_f)
    target_edges = sobel_features(target_tiles.reshape(rows * cols, cell_h, cell_w))
    target_edge_strength = np.sqrt(np.mean(target_edges * target_edges, axis=1))
    edge_norm = max(float(np.percentile(target_edge_strength, 95)), 1.0)
    target_edge_strength = np.clip(target_edge_strength / edge_norm, 0.0, 1.0)

    for r in range(rows):
        for c in range(cols):
            block = working_tiles[r, c]
            block_vec = block.reshape(-1)
            tone_scores = normalized_mse(block_vec, glyph_vectors)
            total_scores = tone_weight * tone_scores

            if edge_weight > 0.0:
                edge_scores = normalized_mse(target_edges[r * cols + c], glyph_edges)
                total_scores += edge_weight * target_edge_strength[r * cols + c] * edge_scores

            if neighbor_weight > 0.0 and (r > 0 or c > 0):
                total_scores += neighbor_weight * neighbor_penalty(r, c, genome, glyphs_f, target_tiles)

            best = int(np.argmin(total_scores))
            genome[r, c] = best

            original_block = target_tiles[r, c]
            residual = block - glyphs_f[best]
            cell_mse[r, c] = float(np.mean((original_block - glyphs_f[best]) ** 2))

            if dither_strength > 0.0:
                for dr, dc, weight in DIFFUSION_NEIGHBORS:
                    nr = r + dr
                    nc = c + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        working_tiles[nr, nc] = np.clip(
                            working_tiles[nr, nc] + dither_strength * weight * residual,
                            0.0,
                            255.0,
                        )

    return genome, cell_mse


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

    print(
        "Hybrid greedy matching "
        f"(tone={args.tone_weight}, edge={args.edge_weight}, "
        f"neighbor={args.neighbor_weight}, dither={args.dither_strength})..."
    )
    genome, cell_mse = hybrid_greedy_genome(
        target,
        glyphs,
        rows,
        cols,
        cell_h,
        cell_w,
        args.tone_weight,
        args.edge_weight,
        args.neighbor_weight,
        args.dither_strength,
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
