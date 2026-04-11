#!/usr/bin/env python3
"""
Triangle Art via Genetic Algorithm — entry point.

Usage:
    python triangles_ga/main.py input.jpg
    python triangles_ga/main.py input.jpg --n-triangles 100 --generations 1000
    python triangles_ga/main.py input.jpg --selection boltzmann --crossover annular --mutation non_uniform
    python triangles_ga/main.py input.jpg --survival additive --img-size 128
"""

import sys
import time
from pathlib import Path

# Allow running as a script from the project root: python triangles_ga/main.py ...
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
from typing import Optional

import numpy as np
from PIL import Image

from triangles_ga.config import Config
from triangles_ga.ga import TriangleGA
from triangles_ga.io import save_result
from triangles_ga.plots import export_history_csv, export_run_metadata, save_run_plots
from triangles_ga.render import set_backend, _HAVE_SKIA


def load_target(image_path: str, img_size: Optional[int]) -> tuple[np.ndarray, int, int]:
    """
    Load and preprocess the target image as float32 RGB.

    Optionally resizes so the longest side equals img_size (keeps aspect ratio).
    Smaller images make the GA dramatically faster — recommended for early experiments.

    Returns:
        (target_array float32, img_w, img_h)
    """
    img = Image.open(image_path).convert("RGB")

    if img_size is not None:
        w, h = img.size
        scale = img_size / max(w, h)
        new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
        img = img.resize((new_w, new_h), Image.LANCZOS)

    img_w, img_h = img.size
    target = np.array(img, dtype=np.float32)
    return target, img_w, img_h


def parse_args() -> Config:
    p = argparse.ArgumentParser(
        description="Triangle Art via Genetic Algorithm",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    # Problem
    p.add_argument("image", help="Input image path")
    p.add_argument("--n-triangles", type=int, default=50, help="Number of triangles (default: 50)")
    p.add_argument("--img-size", type=int, default=None, help="Resize longest side to this px (default: keep original)")

    # GA core
    p.add_argument("--population", type=int, default=80, help="Population size (default: 80)")
    p.add_argument("--generations", type=int, default=500, help="Generations (default: 500)")
    p.add_argument("--elite", type=int, default=5, help="Elite count preserved per gen (default: 5)")

    # Selection
    p.add_argument(
        "--selection",
        default="tournament_det",
        choices=["tournament_det", "tournament_prob", "roulette", "universal", "boltzmann", "ranking"],
        help="Selection method (default: tournament_det)",
    )
    p.add_argument("--tournament-k", type=int, default=5, help="Tournament size (default: 5)")
    p.add_argument("--tournament-prob", type=float, default=0.75, help="Win probability for probabilistic tournament (default: 0.75)")
    p.add_argument("--boltzmann-t-init", type=float, default=100.0, help="Initial Boltzmann temperature (default: 100.0)")
    p.add_argument("--boltzmann-t-min", type=float, default=1.0, help="Minimum Boltzmann temperature (default: 1.0)")

    # Crossover
    p.add_argument(
        "--crossover",
        default="uniform",
        choices=["uniform", "one_point", "two_point", "annular"],
        help="Crossover method (default: uniform)",
    )
    p.add_argument("--crossover-prob", type=float, default=0.8, help="Crossover probability (default: 0.8)")

    # Mutation
    p.add_argument(
        "--mutation",
        default="uniform",
        choices=["uniform", "gen", "multigen", "non_uniform"],
        help="Mutation method (default: uniform)",
    )
    p.add_argument("--mutation-rate", type=float, default=0.02, help="Per-gene mutation probability (default: 0.02)")
    p.add_argument("--mutation-sigma", type=float, default=0.05, help="Mutation noise std (default: 0.05)")
    p.add_argument("--multigen-max", type=int, default=5, help="Max genes mutated per call for multigen (default: 5)")

    # Survival
    p.add_argument(
        "--survival",
        default="exclusive",
        choices=["exclusive", "additive"],
        help="Survival strategy (default: exclusive)",
    )

    # Termination
    p.add_argument("--stop-stagnation", action="store_true", help="Stop if no improvement for --stagnation-gens generations")
    p.add_argument("--stagnation-gens", type=int, default=50, help="Stagnation window in generations (default: 50)")
    p.add_argument("--stagnation-delta", type=float, default=0.5, help="Minimum MSE improvement to reset stagnation counter (default: 0.5)")
    p.add_argument("--stop-convergence", action="store_true", help="Stop if population fitness std drops below threshold")
    p.add_argument("--convergence-thr", type=float, default=5.0, help="Fitness std threshold for convergence stop (default: 5.0)")

    # Performance
    p.add_argument("--workers", type=int, default=0,
                   help="Parallel worker processes for fitness evaluation (default: 0 = all CPU cores)")
    p.add_argument("--fitness-sample", type=float, default=1.0,
                   help="Fraction of pixels used for MSE fitness (0.1–1.0, default: 1.0 = all pixels)")
    p.add_argument(
        "--renderer",
        default="auto",
        choices=["auto", "skia", "pil"],
        help="Rendering backend: 'skia' (fast, requires skia-python), 'pil' (pure Python), "
             "'auto' (skia if available, else pil). Default: auto",
    )

    # I/O
    p.add_argument("--save-every", type=int, default=50, help="Snapshot interval in gens (default: 50)")
    p.add_argument("--output", default="output/triangles_ga", help="Output directory (default: output/triangles_ga)")
    p.add_argument("--no-plots", action="store_true", help="Skip PNG charts generation (metrics CSV is still saved)")
    p.add_argument("--graphs-only", action="store_true", help="Only save metrics/metadata/plots, skip best image/JSON and snapshots")
    p.add_argument("--seed", type=int, default=42)

    a = p.parse_args()
    return Config(
        image_path=a.image,
        n_triangles=a.n_triangles,
        img_size=a.img_size,
        population=a.population,
        generations=a.generations,
        elite=a.elite,
        selection_method=a.selection,
        tournament_k=a.tournament_k,
        tournament_prob=a.tournament_prob,
        boltzmann_temp_init=a.boltzmann_t_init,
        boltzmann_temp_min=a.boltzmann_t_min,
        crossover_method=a.crossover,
        crossover_prob=a.crossover_prob,
        mutation_method=a.mutation,
        mutation_rate=a.mutation_rate,
        mutation_sigma=a.mutation_sigma,
        multigen_max_genes=a.multigen_max,
        survival_strategy=a.survival,
        stop_on_stagnation=a.stop_stagnation,
        stagnation_gens=a.stagnation_gens,
        stagnation_delta=a.stagnation_delta,
        stop_on_convergence=a.stop_convergence,
        convergence_threshold=a.convergence_thr,
        save_every=a.save_every,
        output_dir=a.output,
        workers=a.workers,
        fitness_sample=a.fitness_sample,
        renderer=a.renderer,
        no_plots=a.no_plots,
        graphs_only=a.graphs_only,
        seed=a.seed,
    )


def main() -> None:
    cfg = parse_args()
    started_at = time.perf_counter()

    set_backend(cfg.renderer)

    # Report which backend is actually in use
    if cfg.renderer == "auto":
        active_renderer = "skia" if _HAVE_SKIA else "pil"
        renderer_note = f"auto → {active_renderer}"
    else:
        active_renderer = cfg.renderer
        renderer_note = active_renderer
    print(f"Renderer: {renderer_note}")

    print(f"Loading target: {cfg.image_path}")
    target, img_w, img_h = load_target(cfg.image_path, cfg.img_size)
    print(f"  Image size: {img_w}×{img_h} px")
    print(
        f"  Selection: {cfg.selection_method}  |  Crossover: {cfg.crossover_method}"
        f"  |  Mutation: {cfg.mutation_method}  |  Survival: {cfg.survival_strategy}"
    )

    print(f"\nInitializing population ({cfg.population} individuals, {cfg.n_triangles} triangles each)...")
    ga = TriangleGA(cfg, target, img_w, img_h)
    ga.initialize()
    print(f"  Initial best MSE: {ga.best[1]:.2f}")

    out_dir = Path(cfg.output_dir)
    snap_dir = out_dir / "snapshots"

    stop_info = []
    if cfg.stop_on_stagnation:
        stop_info.append(f"stagnation>{cfg.stagnation_gens}gens")
    if cfg.stop_on_convergence:
        stop_info.append(f"convergence<{cfg.convergence_threshold}")
    criteria = " | ".join(stop_info) if stop_info else "max generations only"
    print(f"  Termination: {criteria}")

    print(f"\nEvolving {cfg.generations} generations...\n")
    for gen in range(cfg.generations):
        best_fit, mean_fit = ga.step()
        print(
            f"  Gen {gen+1:4d}/{cfg.generations}"
            f"  best={best_fit:8.2f}"
            f"  mean={mean_fit:8.2f}",
            flush=True,
        )

        if (gen + 1) % cfg.save_every == 0 and not cfg.graphs_only:
            best_genome, _ = ga.best
            json_path, png_path = save_result(best_genome, img_w, img_h, snap_dir, f"gen_{gen+1:05d}")
            print(f"    → snapshot: {png_path}")

        stop, reason = ga.should_stop()
        if stop:
            print(f"\n  Early stop at gen {gen+1}: {reason}")
            break

    ga.shutdown()

    best_genome, best_fitness = ga.best
    print(f"\nFinal best MSE: {best_fitness:.2f}")
    if not cfg.graphs_only:
        json_path, png_path = save_result(best_genome, img_w, img_h, out_dir, "best")
        print(f"Saved → {json_path}")
        print(f"Saved → {png_path}")

    metrics_path = export_history_csv(ga.history, out_dir)
    runtime_seconds = time.perf_counter() - started_at
    metadata_path = export_run_metadata(
        {
            "label": out_dir.name,
            "image_path": cfg.image_path,
            "img_w": img_w,
            "img_h": img_h,
            "population": cfg.population,
            "generations_requested": cfg.generations,
            "generations_completed": len(ga.history) - 1,
            "n_triangles": cfg.n_triangles,
            "selection_method": cfg.selection_method,
            "crossover_method": cfg.crossover_method,
            "mutation_method": cfg.mutation_method,
            "survival_strategy": cfg.survival_strategy,
            "best_mse": float(best_fitness),
            "runtime_seconds": runtime_seconds,
            "seed": cfg.seed,
        },
        out_dir,
    )
    print(f"Saved → {metrics_path}")
    print(f"Saved → {metadata_path}")

    if not cfg.no_plots:
        try:
            plot_paths = save_run_plots(ga.history, out_dir / "graphs")
            for plot_path in plot_paths:
                print(f"Saved → {plot_path}")
        except ImportError:
            print("Warning: plots require matplotlib  →  python3 -m pip install matplotlib")

    print("\nDone.")


if __name__ == "__main__":
    main()
