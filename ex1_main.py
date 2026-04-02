"""
Exercise 1 entry point: ASCII Art via Genetic Algorithms

Usage:
    python3 ex1_main.py                          # auto-generates a test image
    python3 ex1_main.py --image images/flag.png --n 32
    python3 ex1_main.py --image images/smile.png --n 20 --gens 500 --no-plot

The script will:
1. Load (or generate) a target image
2. Show the greedy baseline (best char per cell, no GA)
3. Run the GA with a live pygame window updating each generation
4. Save the final ASCII art and fitness plot to output/
"""

import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))

from src.ex1.renderer import preprocess_target, render_ascii_grid, indices_to_chars, CHARSET
from src.ex1.individual import greedy_individual
from src.ex1.ga import run_ga_interruptible


def make_test_image(path: str, size: int = 128):
    """Generate a simple smiley face test image."""
    img = Image.new("L", (size, size), color=255)
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    m = size // 8
    draw.ellipse([m, m, size - m, size - m], outline=0, width=3)
    r = size // 12
    draw.ellipse([size // 3 - r, size // 3 - r, size // 3 + r, size // 3 + r], fill=0)
    draw.ellipse([2 * size // 3 - r, size // 3 - r, 2 * size // 3 + r, size // 3 + r], fill=0)
    draw.arc([size // 4, size // 2, 3 * size // 4, 3 * size // 4], start=0, end=180, fill=0, width=3)
    img.save(path)
    print(f"Generated test image: {path}")


def print_ascii_art(genome: np.ndarray, grid_n: int):
    grid = indices_to_chars(genome, grid_n)
    print("\n" + "─" * (grid_n + 2))
    for row in grid:
        print("|" + "".join(row) + "|")
    print("─" * (grid_n + 2) + "\n")


def main():
    parser = argparse.ArgumentParser(description="ASCII Art via Genetic Algorithms")
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--n", type=int, default=24, help="Grid size NxN (default: 24)")
    parser.add_argument("--cell", type=int, default=8, help="Pixels per cell (default: 8)")
    parser.add_argument("--pop", type=int, default=60, help="Population size (default: 60)")
    parser.add_argument("--gens", type=int, default=200, help="Generations (default: 200)")
    parser.add_argument("--mut", type=float, default=0.02, help="Mutation rate (default: 0.02)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-plot", action="store_true", help="Disable live pygame window")
    args = parser.parse_args()

    Path("images").mkdir(exist_ok=True)
    Path("output").mkdir(exist_ok=True)

    # ── Load / generate image ──
    image_path = args.image
    if image_path is None:
        image_path = "images/test_smiley.png"
        make_test_image(image_path)

    print(f"Target:      {image_path}")
    print(f"Grid:        {args.n}x{args.n}  |  Pop: {args.pop}  |  Gens: {args.gens}  |  Mut: {args.mut}")

    target = preprocess_target(image_path, args.n, args.cell)
    target_pil = Image.fromarray(target.astype(np.uint8)).convert("RGB")

    # ── Greedy baseline ──
    greedy = greedy_individual(target, args.n, args.cell)
    print("\nGreedy baseline (best char per cell, not GA):")
    print_ascii_art(greedy, args.n)

    # ── Setup pygame visualizer ──
    visualizer = None
    if not args.no_plot:
        from src.ex1.visualizer import AsciiGAVisualizer
        visualizer = AsciiGAVisualizer(target_pil, title=f"ASCII Art GA — {Path(image_path).name}")

    history: list[tuple] = []
    stopped_early = False

    def on_generation(gen, best_genome, best_fitness, avg_fitness):
        nonlocal stopped_early
        history.append((best_fitness, avg_fitness))

        if gen % 10 == 0:
            print(f"  Gen {gen:4d} | best={best_fitness:.5f} | avg={avg_fitness:.5f}")

        if visualizer is not None:
            rendered_pil = render_ascii_grid(best_genome, args.n, args.cell)
            rendered_rgb = rendered_pil.convert("RGB")
            keep_going = visualizer.update(gen, rendered_rgb, best_fitness, avg_fitness)
            if not keep_going:
                stopped_early = True
                raise StopIteration  # signal the GA loop to stop

    # ── Run GA ──
    print("\nRunning GA...")
    best_genome, history = run_ga_interruptible(
        target=target,
        grid_n=args.n,
        cell_size=args.cell,
        pop_size=args.pop,
        n_generations=args.gens,
        mutation_rate=args.mut,
        seed=args.seed,
        callback=on_generation,
    )
    if stopped_early:
        print("\nStopped early by user.")

    if visualizer is not None:
        visualizer.close()

    # ── Recover best genome if GA was interrupted before any gen ran ──
    if best_genome is None:
        best_genome = greedy

    final_fitness = history[-1][0] if history else 0.0
    print(f"\nFinal best fitness: {final_fitness:.5f}")
    print("Final ASCII art:")
    print_ascii_art(best_genome, args.n)

    # ── Save outputs ──
    rendered = render_ascii_grid(best_genome, args.n, args.cell)
    rendered.save("output/ex1_result.png")

    grid = indices_to_chars(best_genome, args.n)
    with open("output/ex1_result.txt", "w") as f:
        for row in grid:
            f.write("".join(row) + "\n")

    print("Saved → output/ex1_result.png")
    print("Saved → output/ex1_result.txt")

    # ── Fitness plot (matplotlib, post-run) ──
    if history:
        fig, ax = plt.subplots(figsize=(9, 4))
        bests = [h[0] for h in history]
        avgs  = [h[1] for h in history]
        ax.plot(bests, label="Best", color="#5090ff", linewidth=2)
        ax.plot(avgs,  label="Avg",  color="#ffaa30", linewidth=1.5, alpha=0.8)
        ax.set_xlabel("Generation")
        ax.set_ylabel("Fitness  (1 − MSE/255²)")
        ax.set_title(f"ASCII Art GA — {Path(image_path).name}  (N={args.n})")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("output/ex1_fitness.png", dpi=120)
        print("Saved → output/ex1_fitness.png")
        plt.show()


if __name__ == "__main__":
    main()
