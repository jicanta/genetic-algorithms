"""
Plotting and metrics export utilities for the triangle GA.

The module is optional at runtime: if matplotlib is not installed, callers can
still export CSV metrics and skip image generation gracefully.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable


def _apply_plot_style(plt) -> None:
    """Shared visual style for all exported charts."""
    plt.style.use("default")
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.edgecolor": "#333333",
        "axes.labelcolor": "#222222",
        "axes.titlesize": 16,
        "axes.titleweight": "regular",
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 10,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "grid.color": "#9aa0a6",
        "grid.alpha": 0.28,
        "grid.linestyle": "-",
        "grid.linewidth": 0.8,
    })


def export_history_csv(history: Iterable[dict[str, float]], out_dir: Path) -> Path:
    """Write per-generation metrics to CSV."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "metrics.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["generation", "best", "mean", "std"])
        writer.writeheader()
        for row in history:
            writer.writerow(row)
    return csv_path


def export_run_metadata(metadata: dict, out_dir: Path) -> Path:
    """Write run metadata to JSON so experiment summaries can aggregate results."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "run_metadata.json"
    with open(json_path, "w") as f:
        json.dump(metadata, f, indent=2)
    return json_path


def save_run_plots(history: list[dict[str, float]], out_dir: Path) -> list[Path]:
    """
    Generate standard charts for a single GA run.

    Returns the list of written PNG paths. Raises ImportError if matplotlib is
    not available so the caller can show a friendly warning.
    """
    import matplotlib.pyplot as plt
    _apply_plot_style(plt)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    generations = [int(row["generation"]) for row in history]
    best = [row["best"] for row in history]
    mean = [row["mean"] for row in history]
    std = [row["std"] for row in history]

    saved: list[Path] = []

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(generations, best, label="Best MSE", linewidth=2.4, color="#005f73")
    ax.plot(generations, mean, label="Mean MSE", linewidth=2.0, color="#ee9b00")
    ax.set_title("Mean best fitness por corrida")
    ax.set_xlabel("Generación")
    ax.set_ylabel("MSE")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    path = out_dir / "fitness_evolution.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    saved.append(path)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(generations, std, linewidth=2.4, color="#9b2226")
    ax.set_title("Diversidad de la población")
    ax.set_xlabel("Generación")
    ax.set_ylabel("Desvío estándar del fitness")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    path = out_dir / "diversity_evolution.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    saved.append(path)

    return saved


def save_experiment_plots(results_root: Path, out_dir: Path) -> list[Path]:
    """
    Aggregate run metadata below results_root and generate comparison charts.

    Expected layout:
        results_root/**/run_metadata.json
    """
    import matplotlib.pyplot as plt
    import numpy as np

    _apply_plot_style(plt)

    results_root = Path(results_root)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for metadata_path in sorted(results_root.rglob("run_metadata.json")):
        with open(metadata_path) as f:
            row = json.load(f)
        metrics_path = metadata_path.with_name("metrics.csv")
        if metrics_path.exists():
            with open(metrics_path, newline="") as f:
                row["metrics"] = [
                    {
                        "generation": int(float(m["generation"])),
                        "best": float(m["best"]),
                        "mean": float(m["mean"]),
                        "std": float(m["std"]),
                    }
                    for m in csv.DictReader(f)
                ]
        rows.append(row)

    if not rows:
        raise ValueError(f"No run_metadata.json files found under {results_root}")

    saved: list[Path] = []

    saved.extend(_save_run_overview_plots(rows, out_dir, plt))

    candidate_groups = [
        ("selection_method", "selection"),
        ("crossover_method", "crossover"),
        ("mutation_method", "mutation"),
        ("survival_strategy", "survival"),
    ]
    for key, slug in candidate_groups:
        unique_values = sorted({row.get(key) for row in rows})
        if len(unique_values) > 1:
            saved.extend(_save_group_comparison(rows, key, slug, out_dir, plt, np))

    return saved


def _save_run_overview_plots(rows: list[dict], out_dir: Path, plt) -> list[Path]:
    saved: list[Path] = []

    labels = [row["label"] for row in rows]
    best_values = [row["best_mse"] for row in rows]

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.8), 5))
    ax.bar(labels, best_values, color="#2c7fb8")
    ax.set_title("Fitness final por corrida")
    ax.set_xlabel("Corrida")
    ax.set_ylabel("Fitness final (MSE)")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(True, axis="y")
    fig.tight_layout()
    path = out_dir / "best_mse_by_run.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    saved.append(path)

    return saved


def _save_group_comparison(
    rows: list[dict],
    group_key: str,
    slug: str,
    out_dir: Path,
    plt,
    np,
) -> list[Path]:
    saved: list[Path] = []
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(str(row[group_key]), []).append(row)

    labels = sorted(grouped)
    means = [float(np.mean([row["best_mse"] for row in grouped[label]])) for label in labels]
    stds = [float(np.std([row["best_mse"] for row in grouped[label]])) for label in labels]
    time_means = [float(np.mean([row["runtime_seconds"] for row in grouped[label]])) for label in labels]
    time_stds = [float(np.std([row["runtime_seconds"] for row in grouped[label]])) for label in labels]
    ns = [len(grouped[label]) for label in labels]

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.15), 5.5))
    ax.bar(labels, means, yerr=stds, capsize=7, color="#2c7fb8", ecolor="#111111")
    ax.set_title(f"Fitness final por {slug} (n={min(ns)})")
    ax.set_xlabel(_group_label(slug))
    ax.set_ylabel("Fitness final (best_fitness)")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(True, axis="y")
    fig.tight_layout()
    path = out_dir / f"{slug}_final_fitness_bar.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    saved.append(path)

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.15), 5.5))
    ax.bar(labels, time_means, yerr=time_stds, capsize=7, color="#2c7fb8", ecolor="#111111")
    ax.set_title(f"Tiempo total por {slug} (n={min(ns)})")
    ax.set_xlabel(_group_label(slug))
    ax.set_ylabel("Tiempo total (s)")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(True, axis="y")
    fig.tight_layout()
    path = out_dir / f"{slug}_runtime_bar.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    saved.append(path)

    with_metrics = [label for label in labels if all("metrics" in row for row in grouped[label])]
    if with_metrics:
        fig, ax = plt.subplots(figsize=(10.5, 6.2))
        palette = plt.get_cmap("tab10")
        for idx, label in enumerate(with_metrics):
            runs = grouped[label]
            min_len = min(len(row["metrics"]) for row in runs)
            generations = [runs[0]["metrics"][i]["generation"] for i in range(min_len)]
            matrix = np.array([[row["metrics"][i]["best"] for i in range(min_len)] for row in runs], dtype=float)
            mean_curve = matrix.mean(axis=0)
            std_curve = matrix.std(axis=0)
            color = palette(idx % 10)
            ax.plot(generations, mean_curve, linewidth=2.3, color=color, label=f"{label} (n={len(runs)})")
            ax.fill_between(generations, mean_curve - std_curve, mean_curve + std_curve, color=color, alpha=0.14)
        ax.set_title(f"Mean best_fitness por {slug}")
        ax.set_xlabel("Generación")
        ax.set_ylabel("best_fitness (media por método)")
        ax.legend(loc="upper left", framealpha=0.9)
        ax.grid(True)
        fig.tight_layout()
        path = out_dir / f"{slug}_mean_curve.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        saved.append(path)

    return saved


def _group_label(slug: str) -> str:
    labels = {
        "selection": "Método",
        "crossover": "Crossover",
        "mutation": "Mutación",
        "survival": "Supervivencia",
    }
    return labels.get(slug, slug)
