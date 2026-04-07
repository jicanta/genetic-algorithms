#!/usr/bin/env python3
"""
Run a full experiment suite for one image and generate summary plots.

Examples:
    python3 triangles_ga/run_triangles_suite.py images/argentina_flag.png
    python3 triangles_ga/run_triangles_suite.py images/argentina_flag.png --repeats 5 --output graphs
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


SELECTIONS = ["tournament_det", "tournament_prob", "roulette", "universal", "boltzmann", "ranking"]
CROSSOVERS = ["uniform", "one_point", "two_point", "annular"]
MUTATIONS = ["uniform", "gen", "multigen", "non_uniform"]
SURVIVALS = ["exclusive", "additive"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the triangle GA experiment suite for one image")
    parser.add_argument("image", help="Input image path")
    parser.add_argument("--img-size", type=int, default=64, help="Resize longest side to this value")
    parser.add_argument("--population", type=int, default=65, help="Population size")
    parser.add_argument("--generations", type=int, default=100, help="Number of generations")
    parser.add_argument("--n-triangles", type=int, default=50, help="Triangles per individual")
    parser.add_argument("--save-every", type=int, default=50, help="Snapshot interval")
    parser.add_argument("--repeats", type=int, default=3, help="How many seeds to run per method")
    parser.add_argument("--base-seed", type=int, default=42, help="Seed used for the first repetition")
    parser.add_argument("--output", default="graphs", help="Root output directory")
    parser.add_argument("--with-run-plots", action="store_true", help="Also generate plots inside every single run")
    parser.add_argument("--graphs-only", action="store_true", help="Keep only metrics/metadata/plots for each run")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    image_path = Path(args.image)
    if not image_path.is_absolute():
        image_path = (project_root / image_path).resolve()
    image_name = image_path.stem
    root = (project_root / args.output / image_name).resolve()
    root.mkdir(parents=True, exist_ok=True)

    python = sys.executable
    common = [
        python,
        str(script_dir / "main.py"),
        str(image_path),
        "--img-size",
        str(args.img_size),
        "--population",
        str(args.population),
        "--generations",
        str(args.generations),
        "--n-triangles",
        str(args.n_triangles),
        "--save-every",
        str(args.save_every),
        "--stop-stagnation",
        "--stagnation-gens",
        "30",
        "--stagnation-delta",
        "1.0",
        "--stop-convergence",
        "--convergence-thr",
        "50.0",
    ]
    if not args.with_run_plots:
        common.append("--no-plots")
    if args.graphs_only:
        common.append("--graphs-only")

    suites = [
        ("selection", SELECTIONS, {"crossover": "two_point", "mutation": "uniform", "survival": "exclusive"}),
        ("crossover", CROSSOVERS, {"selection": "tournament_det", "mutation": "uniform", "survival": "exclusive"}),
        ("mutation", MUTATIONS, {"selection": "tournament_det", "crossover": "two_point", "survival": "exclusive"}),
        ("survival", SURVIVALS, {"selection": "tournament_det", "crossover": "two_point", "mutation": "uniform"}),
    ]

    for suite_name, methods, fixed in suites:
        suite_root = root / suite_name
        for method in methods:
            for repeat in range(args.repeats):
                seed = args.base_seed + repeat
                run_output = suite_root / f"{suite_name}_{method}" / f"run_{repeat + 1}"
                cmd = common + ["--output", str(run_output), "--seed", str(seed)]

                if suite_name == "selection":
                    cmd += ["--selection", method]
                elif suite_name == "crossover":
                    cmd += ["--crossover", method]
                elif suite_name == "mutation":
                    cmd += ["--mutation", method]
                elif suite_name == "survival":
                    cmd += ["--survival", method]

                for key, value in fixed.items():
                    cmd += [f"--{key.replace('_', '-')}", value]

                print(f"=== {suite_name}: {method} | run {repeat + 1}/{args.repeats} ===")
                subprocess.run(cmd, check=True)

        subprocess.run([python, str(script_dir / "analyze_triangles_results.py"), str(suite_root)], check=True)

    print(f"\nSuite complete. Results saved under: {root}")


if __name__ == "__main__":
    main()
