# TP2 - Genetic Algorithms

**Sistemas de Inteligencia Artificial — ITBA**

Two genetic algorithm implementations that approximate images using different representations:

1. **Part 1 — ASCII Art GA**: evolves a grid of characters to match a grayscale image
2. **Part 2 — Triangle Art GA**: evolves a set of colored semi-transparent triangles to match an RGB image

---

## Project Structure

```
genetic-algorithms/
├── ascii_ga/                  # Part 1: ASCII Art
│   ├── main.py                # GA entry point
│   ├── main_greedy.py         # Greedy (non-evolutionary) baseline
│   ├── config.py              # Hyperparameter dataclass
│   ├── font.py                # Font loading and glyph cache
│   ├── image.py               # Target image preprocessing
│   ├── render.py              # Genome → image (vectorized NumPy)
│   ├── fitness.py             # MSE fitness function
│   ├── operators.py           # Selection, crossover, mutation
│   ├── ga.py                  # ASCIIArtGA engine
│   └── io.py                  # Save .txt and .png results
├── triangles_ga/              # Part 2: Triangle Art
│   ├── main.py                # GA entry point
│   ├── config.py              # Hyperparameter dataclass
│   ├── genome.py              # Genome representation
│   ├── render.py              # Genome → RGB image (alpha compositing)
│   ├── fitness.py             # MSE fitness function
│   ├── operators.py           # 6 selections, 4 crossovers, 4 mutations
│   ├── ga.py                  # TriangleGA engine
│   └── io.py                  # Save .png and .json results
├── docs/
│   ├── ascii_ga.md            # In-depth ASCII GA algorithm documentation
│   └── ascii_greedy.md        # Greedy matching approach documentation
├── images/                    # Input images
└── output/                    # Generated results (created on first run)
    ├── ascii_ga/
    ├── ascii_greedy/
    └── triangles_ga/
```

---

## Dependencies

```bash
pip install pillow numpy
# Optional — GIF export for ASCII GA:
pip install imageio
```

---

## Part 1 — ASCII Art

### How it works

Unlike a classic ASCII converter (which maps local brightness → character), this algorithm performs global search:

1. **Representation**: each individual is a `rows × cols` grid of character indices into a charset.
2. **Rendering**: the grid is converted to an image by tiling the actual font glyphs (via Pillow), producing a grayscale image with the same resolution as the target.
3. **Fitness**: MSE between the rendered image and the target (lower = better).
4. **Warm start**: half the initial population comes from a greedy brightness-matching heuristic; the other half is random. This gives evolution a strong starting point while maintaining diversity.
5. **Evolution**: tournament selection → crossover → mutation → elite preservation.

### Genetic operators

| Operator | Details |
|----------|---------|
| **Selection** | Deterministic tournament of size `k` |
| **Crossover** | Uniform (per-cell coin flip) or block (random rectangular region swap), alternated randomly |
| **Mutation** | 70% darkness-guided neighbor step (±1/±2 ranks in measured darkness order) + 30% random character |
| **Elitism** | Top-N individuals pass unchanged to the next generation |

### Running

```bash
# Basic run
python ascii_ga/main.py images/photo.jpg

# Higher resolution, more generations
python ascii_ga/main.py images/photo.jpg --cols 120 --generations 2000 --population 100

# Export evolution as GIF
python ascii_ga/main.py images/photo.jpg --save-every 25 --gif

# Custom font
python ascii_ga/main.py images/photo.jpg --font /usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf --font-size 10

# Early stopping
python ascii_ga/main.py images/photo.jpg --stop-stagnation --stagnation-gens 100
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--cols` | `80` | ASCII grid columns |
| `--population` | `80` | Population size |
| `--generations` | `500` | Max generations |
| `--mutation` | `0.02` | Per-cell mutation rate |
| `--font` | *(auto)* | Path to TTF monospace font |
| `--font-size` | `12` | Font size in points |
| `--elite` | `5` | Elite individuals preserved per generation |
| `--tournament-k` | `5` | Tournament size |
| `--crossover-prob` | `0.8` | Crossover probability |
| `--charset` | `@%#*+=-:. ` | Character set (dark → light) |
| `--char-aspect` | *(auto)* | `cell_w/cell_h` override to fix proportions |
| `--save-every` | `50` | Snapshot interval in generations |
| `--output` | `output/ascii_ga/` | Output directory |
| `--gif` | *(flag)* | Export `evolution.gif` (requires imageio) |
| `--seed` | `42` | Random seed |
| `--stop-stagnation` | *(flag)* | Stop if no improvement for `--stagnation-gens` gens |
| `--stagnation-gens` | `50` | Stagnation window |
| `--stagnation-delta` | `0.5` | Minimum MSE improvement to reset counter |
| `--stop-convergence` | *(flag)* | Stop if population fitness std drops below threshold |
| `--convergence-thr` | `5.0` | Convergence threshold |

### Output

```
output/ascii_ga/
├── best.txt            # ASCII art as plain text
├── best.png            # Rendered PNG of the best individual
├── evolution.gif       # Evolution animation (with --gif)
└── snapshots/
    ├── gen_00050.txt
    ├── gen_00050.png
    └── ...
```

---

## Part 1 (bonus) — Greedy ASCII Matching

A fast non-evolutionary alternative that solves each cell independently and exactly.

For each cell in the target grid, it selects the character whose rendered glyph minimizes pixel-level SSE against the corresponding image block. Because the objective is separable (total MSE = sum of per-cell MSEs), this finds the global optimum for this formulation in a single pass.

```bash
python ascii_ga/main_greedy.py images/photo.jpg
python ascii_ga/main_greedy.py images/photo.jpg --cols 120 --charset "@%#*+=-:. "
```

Output goes to `output/ascii_greedy/` by default. See `docs/ascii_greedy.md` for details.

---

## Part 2 — Triangle Art

### How it works

1. **Representation**: each individual is an array of shape `(N_triangles, 10)` where each row encodes one triangle as `[x1, y1, x2, y2, x3, y3, r, g, b, a]` — all values normalized to `[0, 1]`.
2. **Rendering**: triangles are drawn in order onto a white canvas using alpha compositing, producing an RGB image.
3. **Fitness**: MSE between the rendered image and the RGB target.
4. **Evolution**: fully configurable operators (see below).

### Genetic operators

**Selection** (6 methods, `--selection`):

| Method | Description |
|--------|-------------|
| `tournament_det` | Best of k random always wins |
| `tournament_prob` | Best of k wins with probability p; others with geometrically declining probabilities |
| `roulette` | Fitness-proportional (inverted MSE) |
| `universal` | Stochastic universal sampling |
| `boltzmann` | Temperature-annealed: `p_i ∝ exp(-fitness/T)`, T decays over generations |
| `ranking` | Rank-proportional, avoids domination by a single elite individual |

**Crossover** (4 methods, `--crossover`):

| Method | Description |
|--------|-------------|
| `uniform` | Per-triangle coin flip between parents |
| `one_point` | Single cut point |
| `two_point` | Two cut points, exchange middle segment |
| `annular` | Circular two-point; segment can wrap around |

**Mutation** (4 methods, `--mutation`):

| Method | Description |
|--------|-------------|
| `uniform` | Per-gene Gaussian noise with fixed sigma |
| `gen` | Exactly one gene mutated per call |
| `multigen` | 1 to `--multigen-max` randomly selected genes mutated |
| `non_uniform` | Gaussian noise with sigma decaying over generations |

**Survival** (2 strategies, `--survival`):

| Strategy | Description |
|----------|-------------|
| `exclusive` | New generation fully replaces the old one (generational) |
| `additive` | Best individuals from combined parent + offspring pool survive |

### Running

```bash
# Basic run
python triangles_ga/main.py images/photo.jpg

# More triangles, larger population
python triangles_ga/main.py images/photo.jpg --n-triangles 100 --population 120 --generations 1000

# Resize image for faster iteration
python triangles_ga/main.py images/photo.jpg --img-size 128

# Custom operators
python triangles_ga/main.py images/photo.jpg \
    --selection boltzmann \
    --crossover annular \
    --mutation non_uniform \
    --survival additive

# Probabilistic tournament
python triangles_ga/main.py images/photo.jpg --selection tournament_prob --tournament-prob 0.8
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--n-triangles` | `50` | Number of triangles per individual |
| `--img-size` | *(original)* | Resize longest side to N px (speeds up fitness) |
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
| `--survival` | `exclusive` | Survival strategy |
| `--save-every` | `50` | Snapshot interval in generations |
| `--output` | `output/triangles_ga/` | Output directory |
| `--seed` | `42` | Random seed |
| `--stop-stagnation` | *(flag)* | Early stop on stagnation |
| `--stagnation-gens` | `50` | Stagnation window |
| `--stop-convergence` | *(flag)* | Early stop on convergence |
| `--convergence-thr` | `5.0` | Convergence threshold |

### Output

```
output/triangles_ga/
├── best.png            # Rendered PNG of the best individual
├── best.json           # Triangle data: absolute pixel coords + uint8 RGBA colors
└── snapshots/
    ├── gen_00050.png
    ├── gen_00050.json
    └── ...
```

---

## Algorithm overview (common to both parts)

```
Initialize population
    ↓
Evaluate fitness for each individual
    ↓
┌─────────────────────────────────────────┐
│  Select parents                         │
│  Apply crossover (with probability p_c) │
│  Apply mutation                         │
│  Evaluate offspring fitness             │
│  Apply survival strategy                │
│  Preserve elite individuals             │
│  Check termination criteria             │
└─────────────────────────────────────────┘
    ↓
Save best individual
```

Both implementations minimize MSE (mean squared error) between the rendered genome and the target image. Lower MSE = visually closer result.
