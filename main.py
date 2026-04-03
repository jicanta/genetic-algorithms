#!/usr/bin/env python3
"""
ASCII Art via Genetic Algorithm — entry point.

Usage:
    python main.py input.jpg
    python main.py input.jpg --cols 100 --population 80 --generations 1000
    python main.py input.jpg --font /path/to/Mono.ttf --font-size 12 --gif
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

try:
    import imageio
    _HAS_IMAGEIO = True
except ImportError:
    _HAS_IMAGEIO = False

from ascii_ga.config import Config
from ascii_ga.font import load_font, get_cell_size, build_glyph_cache
from ascii_ga.image import load_target
from ascii_ga.render import render_genome
from ascii_ga.io import save_result
from ascii_ga.ga import ASCIIArtGA


def parse_args() -> Config:
    p = argparse.ArgumentParser(
        description="ASCII Art via Genetic Algorithm",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("image",                                              help="Input image path")
    p.add_argument("--cols",         type=int,   default=80,            help="ASCII columns (default: 80)")
    p.add_argument("--population",   type=int,   default=80,            help="Population size (default: 80)")
    p.add_argument("--generations",  type=int,   default=500,           help="Generations (default: 500)")
    p.add_argument("--mutation",     type=float, default=0.02,          help="Per-cell mutation rate (default: 0.02)")
    p.add_argument("--font",         dest="font_path", default=None,    help="Path to TTF monospace font")
    p.add_argument("--font-size",    type=int,   default=12,            help="Font size in points (default: 12)")
    p.add_argument("--save-every",   type=int,   default=50,            help="Snapshot interval in gens (default: 50)")
    p.add_argument("--output",       default="output",                  help="Output directory (default: output)")
    p.add_argument("--elite",        type=int,   default=5,             help="Elite count (default: 5)")
    p.add_argument("--tournament-k", type=int,   default=5,             help="Tournament size (default: 5)")
    p.add_argument("--charset",      default="@%#*+=-:. ",              help="Character set, dark→light")
    p.add_argument("--char-aspect",  type=float, default=None,          help="cell_w/cell_h override (default: auto)")
    p.add_argument("--gif",          action="store_true",               help="Save evolution.gif (needs imageio)")
    p.add_argument("--seed",         type=int,   default=42)
    a = p.parse_args()
    return Config(
        image_path=a.image,
        cols=a.cols,
        population=a.population,
        generations=a.generations,
        mutation=a.mutation,
        font_path=a.font_path,
        font_size=a.font_size,
        save_every=a.save_every,
        output_dir=a.output,
        elite=a.elite,
        tournament_k=a.tournament_k,
        charset=a.charset,
        char_aspect=a.char_aspect,
        gif=a.gif,
        seed=a.seed,
    )


def main():
    cfg = parse_args()

    print("Loading font...")
    font = load_font(cfg.font_path, cfg.font_size)
    cell_w, cell_h = get_cell_size(font)
    print(f"  Cell size: {cell_w}×{cell_h} px")

    print(f"Building glyph cache  charset={repr(cfg.charset)}")
    glyphs, darkness = build_glyph_cache(cfg.charset, font, cell_w, cell_h)
    ordered = "".join(cfg.charset[i] for i in np.argsort(darkness)[::-1])
    print(f"  Measured darkness order (dark→light): {repr(ordered)}")

    print(f"Loading target: {cfg.image_path}")
    target, rows, cols = load_target(cfg.image_path, cfg.cols, cell_w, cell_h, cfg.char_aspect)
    print(f"  Grid: {rows} rows × {cols} cols  |  Render: {cols*cell_w}×{rows*cell_h} px")

    print(f"\nInitializing population ({cfg.population} individuals)...")
    ga = ASCIIArtGA(cfg, glyphs, darkness, target, rows, cols, cell_h, cell_w)
    ga.initialize()
    print(f"  Initial best MSE: {ga.best[1]:.2f}")

    out_dir = Path(cfg.output_dir)
    snap_dir = out_dir / "snapshots"
    gif_frames: list[np.ndarray] = []

    print(f"\nEvolving {cfg.generations} generations...\n")
    for gen in range(cfg.generations):
        best_fit, mean_fit = ga.step()
        print(
            f"  Gen {gen+1:4d}/{cfg.generations}"
            f"  best={best_fit:8.2f}"
            f"  mean={mean_fit:8.2f}",
            flush=True,
        )

        if (gen + 1) % cfg.save_every == 0:
            best_genome, _ = ga.best
            _, png = save_result(
                best_genome, cfg.charset, glyphs, rows, cols, cell_h, cell_w,
                snap_dir, prefix=f"gen_{gen+1:05d}",
            )
            print(f"    → snapshot: {png}")

            if cfg.gif:
                rendered = render_genome(best_genome, glyphs, rows, cols, cell_h, cell_w)
                gif_frames.append(rendered.copy())

    best_genome, best_fitness = ga.best
    txt_path, png_path = save_result(
        best_genome, cfg.charset, glyphs, rows, cols, cell_h, cell_w,
        out_dir, prefix="best",
    )
    print(f"\nFinal best MSE: {best_fitness:.2f}")
    print(f"Saved → {txt_path}")
    print(f"Saved → {png_path}")

    if cfg.gif:
        if not _HAS_IMAGEIO:
            print("Warning: --gif requires imageio  →  pip install imageio")
        elif gif_frames:
            gif_path = out_dir / "evolution.gif"
            imageio.mimsave(
                str(gif_path),
                [Image.fromarray(f, "L").convert("RGB") for f in gif_frames],
                fps=4,
            )
            print(f"Saved → {gif_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
