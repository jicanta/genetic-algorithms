#!/usr/bin/env python3
"""
Triangle Art via Genetic Algorithm — entry point.

Usage:
    python main_triangles.py input.jpg
    python main_triangles.py input.jpg --n-triangles 100 --generations 1000
    python main_triangles.py input.jpg --population 100 --img-size 128
"""

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from triangles_ga.config import Config
from triangles_ga.ga import TriangleGA
from triangles_ga.io import save_result


def load_target(image_path: str, img_size: Optional[int]) -> tuple[np.ndarray, int, int]:
    """
    Load and preprocess the target image.

    Converts to RGB float32. Optionally resizes so the longest side equals img_size,
    keeping aspect ratio — smaller images make the GA dramatically faster.

    Returns:
        (target_array, img_w, img_h)
    """
    img = Image.open(image_path).convert("RGB")

    if img_size is not None:
        w, h = img.size
        scale = img_size / max(w, h)
        new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
        img = img.resize((new_w, new_h), Image.LANCZOS)

    img_w, img_h = img.size
    target = np.array(img, dtype=np.float32)  # (img_h, img_w, 3)
    return target, img_w, img_h


def parse_args() -> Config:
    p = argparse.ArgumentParser(
        description="Triangle Art via Genetic Algorithm",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("image",                                                  help="Input image path")
    p.add_argument("--n-triangles",   type=int,   default=50,               help="Number of triangles (default: 50)")
    p.add_argument("--population",    type=int,   default=80,               help="Population size (default: 80)")
    p.add_argument("--generations",   type=int,   default=500,              help="Generations (default: 500)")
    p.add_argument("--mutation-rate", type=float, default=0.02,             help="Per-gene mutation probability (default: 0.02)")
    p.add_argument("--mutation-sigma",type=float, default=0.05,             help="Mutation noise std (default: 0.05)")
    p.add_argument("--elite",         type=int,   default=5,                help="Elite count (default: 5)")
    p.add_argument("--tournament-k",  type=int,   default=5,                help="Tournament size (default: 5)")
    p.add_argument("--crossover-prob",type=float, default=0.8,              help="Crossover probability (default: 0.8)")
    p.add_argument("--save-every",    type=int,   default=50,               help="Snapshot interval in gens (default: 50)")
    p.add_argument("--output",        default="output_triangles",           help="Output directory (default: output_triangles)")
    p.add_argument("--img-size",      type=int,   default=None,             help="Resize longest side to this (default: keep original)")
    p.add_argument("--seed",          type=int,   default=42)
    a = p.parse_args()

    return Config(
        image_path=a.image,
        n_triangles=a.n_triangles,
        population=a.population,
        generations=a.generations,
        mutation_rate=a.mutation_rate,
        mutation_sigma=a.mutation_sigma,
        elite=a.elite,
        tournament_k=a.tournament_k,
        crossover_prob=a.crossover_prob,
        save_every=a.save_every,
        output_dir=a.output,
        img_size=a.img_size,
        seed=a.seed,
    )


def main() -> None:
    cfg = parse_args()

    print(f"Loading target: {cfg.image_path}")
    target, img_w, img_h = load_target(cfg.image_path, cfg.img_size)
    print(f"  Image size: {img_w}×{img_h} px")

    print(f"\nInitializing population ({cfg.population} individuals, {cfg.n_triangles} triangles each)...")
    ga = TriangleGA(cfg, target, img_w, img_h)
    ga.initialize()
    print(f"  Initial best MSE: {ga.best[1]:.2f}")

    out_dir = Path(cfg.output_dir)
    snap_dir = out_dir / "snapshots"

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
            json_path, png_path = save_result(best_genome, img_w, img_h, snap_dir, f"gen_{gen+1:05d}")
            print(f"    → snapshot: {png_path}")

    best_genome, best_fitness = ga.best
    json_path, png_path = save_result(best_genome, img_w, img_h, out_dir, "best")
    print(f"\nFinal best MSE: {best_fitness:.2f}")
    print(f"Saved → {json_path}")
    print(f"Saved → {png_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
