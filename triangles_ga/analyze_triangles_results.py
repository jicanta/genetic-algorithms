#!/usr/bin/env python3
"""
Aggregate triangle GA experiment results and generate comparison charts.

Examples:
    python3 triangles_ga/analyze_triangles_results.py output_tests
    python3 triangles_ga/analyze_triangles_results.py output_tests/argentina_flag
"""

import argparse
import sys
from pathlib import Path

# Allow running as a script from the project root: python triangles_ga/analyze_triangles_results.py ...
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from triangles_ga.plots import save_experiment_plots


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate comparison charts from triangle GA outputs")
    parser.add_argument("results_root", help="Root directory containing run_metadata.json files")
    parser.add_argument(
        "--output",
        default=None,
        help="Directory for generated charts (default: <results_root>/graphs)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_root = Path(args.results_root)
    out_dir = Path(args.output) if args.output else results_root / "graphs"

    try:
        plot_paths = save_experiment_plots(results_root, out_dir)
    except ImportError:
        print("matplotlib is required for charts  →  python3 -m pip install matplotlib")
        raise SystemExit(1)
    except ValueError as exc:
        print(exc)
        raise SystemExit(1)

    print(f"Loaded results from: {results_root}")
    for path in plot_paths:
        print(f"Saved → {path}")


if __name__ == "__main__":
    main()
