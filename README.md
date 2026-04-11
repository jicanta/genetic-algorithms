# TP2 - Genetic Algorithms

**Sistemas de Inteligencia Artificial — ITBA**

Two genetic algorithm implementations that approximate images using different representations:

- **Part 1 — ASCII Art GA**: evolves a grid of characters to match a grayscale image
- **Part 2 — Triangle Art GA**: evolves a set of colored semi-transparent triangles to match an RGB image

---

## Installation

```bash
pip install pillow numpy

# Triangle Art: fast Skia renderer (~5–15x faster than PIL)
pip install skia-python

# ASCII Art: GIF export
pip install imageio
```

---

## Project Structure

```
genetic-algorithms/
├── ascii_ga/
│   ├── main.py            # GA entry point
│   ├── main_greedy.py     # Greedy (non-evolutionary) baseline
│   ├── config.py
│   ├── font.py
│   ├── image.py
│   ├── render.py
│   ├── fitness.py
│   ├── operators.py
│   ├── ga.py
│   └── io.py
├── triangles_ga/
│   ├── main.py            # GA entry point
│   ├── config.py
│   ├── genome.py
│   ├── render.py
│   ├── fitness.py
│   ├── operators.py
│   ├── ga.py
│   └── io.py
├── docs/
│   ├── ascii_ga.md
│   └── ascii_greedy.md
├── images/                # Input images
└── output/                # Generated results (created on first run)
```

---

## Part 1 — ASCII Art GA

### How it works

Each individual is a `rows × cols` grid of character indices. At every generation the grid is rendered by tiling actual font glyphs (via Pillow), producing a grayscale image that is compared to the target via MSE.

Half the initial population comes from a greedy brightness-matching heuristic; the rest is random. This warm start gives evolution a strong baseline while preserving diversity.

### Genetic operators

| | Method | Details |
|---|---|---|
| **Selection** | Deterministic tournament | Best of `k` random individuals always wins |
| **Crossover** | Uniform or block | Per-cell coin flip, or random rectangular region swap — alternated randomly |
| **Mutation** | Darkness-guided | 70% step ±1–2 ranks in measured darkness order; 30% random character |
| **Elitism** | Top-N | Preserved unchanged every generation |

### Running

```bash
python ascii_ga/main.py images/photo.jpg
python ascii_ga/main.py images/photo.jpg --cols 120 --generations 2000
python ascii_ga/main.py images/photo.jpg --save-every 25 --gif
```

### Parameters

| Parameter | Default | Description |
|---|---|---|
| `--cols` | `80` | ASCII grid columns |
| `--population` | `80` | Population size |
| `--generations` | `500` | Max generations |
| `--elite` | `5` | Elite individuals preserved per generation |
| `--mutation` | `0.02` | Per-cell mutation rate |
| `--tournament-k` | `5` | Tournament size |
| `--crossover-prob` | `0.8` | Crossover probability |
| `--charset` | `@%#*+=-:. ` | Character set (dark → light) |
| `--font` | *(auto)* | Path to TTF monospace font |
| `--font-size` | `12` | Font size in points |
| `--char-aspect` | *(auto)* | `cell_w/cell_h` override to fix proportions |
| `--save-every` | `50` | Snapshot interval in generations |
| `--output` | `output/ascii_ga/` | Output directory |
| `--gif` | *(flag)* | Export `evolution.gif` (requires imageio) |
| `--seed` | `42` | Random seed |
| `--stop-stagnation` | *(flag)* | Stop if no improvement for `--stagnation-gens` generations |
| `--stagnation-gens` | `50` | Stagnation window |
| `--stagnation-delta` | `0.5` | Minimum MSE improvement to reset the counter |
| `--stop-convergence` | *(flag)* | Stop if population fitness std drops below threshold |
| `--convergence-thr` | `5.0` | Convergence threshold |
| `--no-plots` | *(flag)* | Skip chart generation |
| `--graphs-only` | *(flag)* | Save only metrics/plots, skip images and snapshots |

### Output

```
output/ascii_ga/
├── best.txt
├── best.png
├── metrics.csv
├── run_metadata.json
├── graphs/
└── snapshots/
    ├── gen_00050.txt
    └── gen_00050.png
```

---

## Part 1 (bonus) — Greedy ASCII Matching

A fast non-evolutionary baseline. For each cell it scores every glyph using a weighted combination of grayscale mismatch, Sobel edge mismatch, and boundary consistency with already-placed neighbors. Error is diffused to future cells in a Floyd-Steinberg pattern.

```bash
python ascii_ga/main_greedy.py images/photo.jpg
python ascii_ga/main_greedy.py images/photo.jpg --cols 120
# Pure per-block SSE (no heuristics):
python ascii_ga/main_greedy.py images/photo.jpg --edge-weight 0 --neighbor-weight 0 --dither-strength 0
```

| Parameter | Default | Description |
|---|---|---|
| `--cols` | `80` | ASCII grid columns |
| `--font` | *(auto)* | Path to TTF monospace font |
| `--font-size` | `12` | Font size in points |
| `--charset` | all printable ASCII | Allowed characters |
| `--char-aspect` | *(auto)* | `cell_w/cell_h` override |
| `--tone-weight` | `1.0` | Weight for grayscale block mismatch |
| `--edge-weight` | `0.20` | Weight for Sobel edge mismatch |
| `--neighbor-weight` | `0.10` | Weight for left/up boundary consistency |
| `--dither-strength` | `0.15` | Error-diffusion strength |
| `--output` | `output/ascii_greedy/` | Output directory |

---

## Part 2 — Triangle Art GA

### How it works

Each individual is a float32 array of shape `(N_triangles, 10)` — one row per triangle: `[x1, y1, x2, y2, x3, y3, r, g, b, a]`, all values in `[0, 1]`. Triangles are drawn in order onto a white canvas with alpha compositing to produce an RGB image. Fitness is MSE against the target.

### Initial population strategies (`--init`)

| Strategy | Flag | Description |
|---|---|---|
| Random | `--init random` | All genes (positions, colors, alpha) sampled uniformly from `[0, 1]`. *(default)* |
| Color sample | `--init color_sample` | Vertex positions and alpha are random; each triangle's RGB is sampled from the target image at the centroid of its vertices. Gives a much lower initial MSE — the GA starts with plausible colors and focuses evolution on shape and placement. |

### Rendering backends (`--renderer`)

| Backend | Flag | Description |
|---|---|---|
| Auto | `--renderer auto` | Uses Skia if installed, PIL otherwise. *(default)* |
| Skia | `--renderer skia` | Single-surface draw, anti-aliased, ~5–15x faster than PIL. Requires `pip install skia-python`. |
| PIL | `--renderer pil` | One RGBA layer per triangle, pure Python, no native deps. |

### Genetic operators

**Selection** (`--selection`):

| Method | Description |
|---|---|
| `tournament_det` | Best of `k` random individuals always wins *(default)* |
| `tournament_prob` | Best of `k` wins with probability `p`; others with geometrically declining probabilities |
| `roulette` | Fitness-proportional (inverted MSE) |
| `universal` | Stochastic universal sampling |
| `boltzmann` | Temperature-annealed: `p_i ∝ exp(-fitness/T)`, T decays linearly over generations |
| `ranking` | Rank-proportional; avoids domination by a single elite individual |

**Crossover** (`--crossover`):

| Method | Description |
|---|---|
| `uniform` | Per-triangle coin flip between parents *(default)* |
| `one_point` | Single cut point |
| `two_point` | Two cut points, exchange middle segment |
| `annular` | Circular two-point; segment can wrap around |

**Mutation** (`--mutation`):

| Method | Description |
|---|---|
| `uniform` | Per-gene Gaussian noise with typed geometry/color/alpha sigma *(default)* |
| `gen` | Exactly one gene mutated per call, using its gene-type sigma |
| `multigen` | 1 to `--multigen-max` randomly selected genes mutated |
| `non_uniform` | Typed Gaussian noise with sigma decaying over generations |

Triangle draw order is also mutable via `--layer-mutation-rate`, because alpha compositing makes earlier/later triangles produce different images even with the same shapes and colors.

**Survival** (`--survival`):

| Strategy | Description |
|---|---|
| `exclusive` | New generation fully replaces the old one (generational) *(default)* |
| `additive` | Best individuals from the combined parent + offspring pool survive |

### Running

```bash
# Basic run
python triangles_ga/main.py images/photo.jpg

# Color-sampled init + Skia renderer
python triangles_ga/main.py images/photo.jpg --init color_sample --renderer skia

# More triangles, larger population, smaller image for speed
python triangles_ga/main.py images/photo.jpg --n-triangles 100 --population 120 --img-size 128

# Custom operators
python triangles_ga/main.py images/photo.jpg \
    --selection boltzmann \
    --crossover annular \
    --mutation non_uniform \
    --survival additive

# Early stopping
python triangles_ga/main.py images/photo.jpg --stop-stagnation --stagnation-gens 100
```

### Parameters

| Parameter | Default | Description |
|---|---|---|
| `--n-triangles` | `50` | Triangles per individual |
| `--img-size` | *(original)* | Resize longest side to N px before running |
| `--init` | `random` | Initial population: `random` \| `color_sample` |
| `--renderer` | `auto` | Rendering backend: `auto` \| `skia` \| `pil` |
| `--population` | `80` | Population size |
| `--generations` | `500` | Max generations |
| `--elite` | `5` | Elite individuals preserved per generation |
| `--selection` | `tournament_det` | Selection method (see table above) |
| `--tournament-k` | `5` | Tournament size |
| `--tournament-prob` | `0.75` | Win probability for probabilistic tournament |
| `--boltzmann-t-init` | `100.0` | Initial Boltzmann temperature |
| `--boltzmann-t-min` | `1.0` | Minimum Boltzmann temperature |
| `--crossover` | `uniform` | Crossover method (see table above) |
| `--crossover-prob` | `0.8` | Crossover probability |
| `--mutation` | `uniform` | Mutation method (see table above) |
| `--mutation-rate` | `0.02` | Per-gene mutation probability |
| `--mutation-sigma` | `0.05` | Mutation noise std |
| `--multigen-max` | `5` | Max genes for multigen mutation |
| `--geometry-mutation-scale` | `1.0` | Sigma multiplier for triangle vertex genes |
| `--color-mutation-scale` | `0.5` | Sigma multiplier for RGB genes |
| `--alpha-mutation-scale` | `0.5` | Sigma multiplier for opacity genes |
| `--layer-mutation-rate` | `0.02` | Per-individual chance to mutate triangle draw order |
| `--layer-mutation-max-shift` | `8` | Max positions for move-order mutation |
| `--survival` | `exclusive` | Survival strategy (see table above) |
| `--workers` | `0` | Parallel processes for fitness eval; `0` = all CPU cores |
| `--fitness-sample` | `1.0` | Fraction of pixels used for MSE (e.g. `0.5` for 2x speedup) |
| `--save-every` | `50` | Snapshot interval in generations |
| `--output` | `output/triangles_ga/` | Output directory |
| `--seed` | `42` | Random seed |
| `--stop-stagnation` | *(flag)* | Stop if no improvement for `--stagnation-gens` generations |
| `--stagnation-gens` | `50` | Stagnation window |
| `--stagnation-delta` | `0.5` | Minimum MSE improvement to reset the counter |
| `--stop-convergence` | *(flag)* | Stop if population fitness std drops below threshold |
| `--convergence-thr` | `5.0` | Convergence threshold |
| `--no-plots` | *(flag)* | Skip chart generation |
| `--graphs-only` | *(flag)* | Save only metrics/plots, skip images and snapshots |

### Output

```
output/triangles_ga/
├── best.png
├── best.json          # Triangle data: absolute pixel coords + uint8 RGBA colors
├── metrics.csv
├── run_metadata.json
├── graphs/
└── snapshots/
    ├── gen_00050.png
    └── gen_00050.json
```

---

## Algorithm overview

```
Initialize population
        ↓
Evaluate fitness (MSE vs target)
        ↓
┌──────────────────────────────────────┐
│  Select parents                      │
│  Apply crossover (prob = p_c)        │
│  Apply mutation                      │
│  Evaluate offspring                  │
│  Apply survival strategy             │
│  Preserve elite individuals          │
│  Check termination criteria          │
└──────────────────────────────────────┘
        ↓
Save best individual
```

Both implementations minimize MSE between the rendered genome and the target image. Lower MSE = visually closer result.
