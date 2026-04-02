"""
Exercise 1 entry point: ASCII Art via Genetic Algorithms

Usage:
    python ex1_main.py                          # uses built-in test image
    python ex1_main.py --image images/smile.png --n 32
    python ex1_main.py --image images/flag.png --n 16 --pop 100 --gens 300

The script will:
1. Load (or generate) a target image
2. Run the GA
3. Show a live matplotlib window updating each generation
4. Save the final ASCII art to output/ and plot fitness history
"""

import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.ex1.renderer import preprocess_target, render_ascii_grid, indices_to_chars, CHARSET
from src.ex1.individual import greedy_individual
from src.ex1.ga import run_ga


def make_test_image(path: str, size: int = 64):
    """Generate a simple test image (smiley face) if no image is provided."""
    from PIL import Image, ImageDraw
    img = Image.new("L", (size, size), color=255)
    draw = ImageDraw.Draw(img)
    # Face outline
    margin = size // 8
    draw.ellipse([margin, margin, size - margin, size - margin], outline=0, width=2)
    # Eyes
    eye_r = size // 12
    draw.ellipse([size // 3 - eye_r, size // 3 - eye_r,
                  size // 3 + eye_r, size // 3 + eye_r], fill=0)
    draw.ellipse([2 * size // 3 - eye_r, size // 3 - eye_r,
                  2 * size // 3 + eye_r, size // 3 + eye_r], fill=0)
    # Smile
    draw.arc([size // 4, size // 2, 3 * size // 4, 3 * size // 4], start=0, end=180, fill=0, width=2)
    img.save(path)
    print(f"Generated test image: {path}")


def print_ascii_art(genome: np.ndarray, grid_n: int):
    """Print the ASCII art to terminal."""
    grid = indices_to_chars(genome, grid_n)
    print("\n" + "─" * (grid_n + 2))
    for row in grid:
        print("|" + "".join(row) + "|")
    print("─" * (grid_n + 2) + "\n")


def main():
    parser = argparse.ArgumentParser(description="ASCII Art via Genetic Algorithms")
    parser.add_argument("--image", type=str, default=None, help="Path to input image")
    parser.add_argument("--n", type=int, default=24, help="Grid size NxN (default: 24)")
    parser.add_argument("--cell", type=int, default=8, help="Pixels per cell (default: 8)")
    parser.add_argument("--pop", type=int, default=60, help="Population size (default: 60)")
    parser.add_argument("--gens", type=int, default=200, help="Number of generations (default: 200)")
    parser.add_argument("--mut", type=float, default=0.02, help="Mutation rate (default: 0.02)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no-plot", action="store_true", help="Disable live plot")
    args = parser.parse_args()

    # ── Prepare image ──
    Path("images").mkdir(exist_ok=True)
    Path("output").mkdir(exist_ok=True)

    image_path = args.image
    if image_path is None:
        image_path = "images/test_smiley.png"
        make_test_image(image_path, size=128)

    print(f"Target image:   {image_path}")
    print(f"Grid size:      {args.n}x{args.n}")
    print(f"Population:     {args.pop}")
    print(f"Generations:    {args.gens}")
    print(f"Mutation rate:  {args.mut}")

    target = preprocess_target(image_path, args.n, args.cell)

    # ── Greedy baseline ──
    greedy = greedy_individual(target, args.n, args.cell)
    print(f"\nGreedy baseline (not GA, just best char per cell):")
    print_ascii_art(greedy, args.n)

    # ── Setup live plot ──
    if not args.no_plot:
        plt.ion()
        fig = plt.figure(figsize=(14, 5))
        gs = gridspec.GridSpec(1, 3, figure=fig)
        ax_target = fig.add_subplot(gs[0])
        ax_render = fig.add_subplot(gs[1])
        ax_fitness = fig.add_subplot(gs[2])

        ax_target.imshow(target, cmap="gray", vmin=0, vmax=255)
        ax_target.set_title("Target Image")
        ax_target.axis("off")

        ax_render.set_title("Best Individual (Gen 0)")
        ax_render.axis("off")

        ax_fitness.set_title("Fitness over Generations")
        ax_fitness.set_xlabel("Generation")
        ax_fitness.set_ylabel("Fitness")

        best_line, = ax_fitness.plot([], [], label="Best", color="blue")
        avg_line, = ax_fitness.plot([], [], label="Avg", color="orange", alpha=0.7)
        ax_fitness.legend()

        best_fitnesses = []
        avg_fitnesses = []

        plt.tight_layout()
        plt.show()

    # ── Callback: update plot every 10 generations ──
    def on_generation(gen, best_genome, best_fitness, avg_fitness):
        if args.no_plot:
            if gen % 20 == 0:
                print(f"Gen {gen:4d} | best={best_fitness:.5f} | avg={avg_fitness:.5f}")
            return

        best_fitnesses.append(best_fitness)
        avg_fitnesses.append(avg_fitness)

        if gen % 10 == 0:
            rendered = render_ascii_grid(best_genome, args.n, args.cell)
            ax_render.clear()
            ax_render.imshow(rendered, cmap="gray", vmin=0, vmax=255)
            ax_render.set_title(f"Best Individual (Gen {gen}) fitness={best_fitness:.4f}")
            ax_render.axis("off")

            xs = list(range(len(best_fitnesses)))
            best_line.set_data(xs, best_fitnesses)
            avg_line.set_data(xs, avg_fitnesses)
            ax_fitness.relim()
            ax_fitness.autoscale_view()

            fig.canvas.draw()
            fig.canvas.flush_events()

            if gen % 20 == 0:
                print(f"Gen {gen:4d} | best={best_fitness:.5f} | avg={avg_fitness:.5f}")

    # ── Run GA ──
    print("\nRunning Genetic Algorithm...")
    best_genome, history = run_ga(
        target=target,
        grid_n=args.n,
        cell_size=args.cell,
        pop_size=args.pop,
        n_generations=args.gens,
        mutation_rate=args.mut,
        seed=args.seed,
        callback=on_generation,
    )

    # ── Results ──
    final_fitness = history[-1][0]
    print(f"\nDone! Final best fitness: {final_fitness:.5f}")
    print("\nFinal ASCII art:")
    print_ascii_art(best_genome, args.n)

    # Save outputs
    rendered = render_ascii_grid(best_genome, args.n, args.cell)
    rendered.save("output/ex1_result.png")
    print("Saved rendered image → output/ex1_result.png")

    # Save ASCII text
    grid = indices_to_chars(best_genome, args.n)
    with open("output/ex1_result.txt", "w") as f:
        for row in grid:
            f.write("".join(row) + "\n")
    print("Saved ASCII art text → output/ex1_result.txt")

    # Final plot
    if not args.no_plot:
        plt.ioff()
        fig2, ax = plt.subplots(figsize=(8, 4))
        best_f = [h[0] for h in history]
        avg_f = [h[1] for h in history]
        ax.plot(best_f, label="Best fitness", color="blue")
        ax.plot(avg_f, label="Avg fitness", color="orange", alpha=0.7)
        ax.set_xlabel("Generation")
        ax.set_ylabel("Fitness")
        ax.set_title("ASCII Art GA - Fitness History")
        ax.legend()
        plt.tight_layout()
        plt.savefig("output/ex1_fitness.png")
        print("Saved fitness plot → output/ex1_fitness.png")
        plt.show()


if __name__ == "__main__":
    main()
