#!/usr/bin/env python3
"""
ASCII Art via Genetic Algorithm

Evolves a full grid of ASCII characters so that the rendered image (monospace
font, cell-by-cell) minimizes mean squared error against a target image.
This is the "render → compare → evolve" loop described by Nicassio's JSGenetic
work, reimplemented in Python with numpy and Pillow.

Usage:
    python ascii_ga.py input.jpg
    python ascii_ga.py input.jpg --cols 100 --population 80 --generations 1000
    python ascii_ga.py input.jpg --cols 80 --font /path/to/Mono.ttf --font-size 12
    python ascii_ga.py input.jpg --save-every 50 --gif
"""

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    import imageio
    _HAS_IMAGEIO = True
except ImportError:
    _HAS_IMAGEIO = False


# ─── Config ───────────────────────────────────────────────────────────────────

@dataclass
class Config:
    image_path: str
    cols: int = 80
    population: int = 80
    generations: int = 500
    mutation: float = 0.02
    font_path: Optional[str] = None
    font_size: int = 12
    save_every: int = 50
    output_dir: str = "output"
    elite: int = 5
    tournament_k: int = 5
    crossover_prob: float = 0.8
    charset: str = "@%#*+=-:. "
    char_aspect: Optional[float] = None   # None = auto from font metrics
    gif: bool = False
    seed: int = 42


# ─── Font & glyph cache ───────────────────────────────────────────────────────

_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
    "/usr/share/fonts/truetype/freefont/FreeMono.ttf",
    "/usr/share/fonts/truetype/ubuntu/UbuntuMono-R.ttf",
]


def load_font(font_path: Optional[str], font_size: int) -> ImageFont.FreeTypeFont:
    if font_path:
        return ImageFont.truetype(font_path, font_size)
    for path in _FONT_CANDIDATES:
        try:
            return ImageFont.truetype(path, font_size)
        except OSError:
            continue
    raise RuntimeError(
        "No monospace font found on system. "
        "Install DejaVu fonts or pass --font /path/to/Mono.ttf"
    )


def get_cell_size(font: ImageFont.FreeTypeFont) -> tuple[int, int]:
    """Return (cell_w, cell_h) in pixels for a monospace font."""
    ascent, descent = font.getmetrics()
    cell_h = ascent + descent
    cell_w = max(1, round(font.getlength("M")))
    return cell_w, cell_h


def build_glyph_cache(
    charset: str, font: ImageFont.FreeTypeFont, cell_w: int, cell_h: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Render every character in charset to a (cell_h, cell_w) grayscale tile.

    Returns:
        glyphs:   uint8 array (N, cell_h, cell_w) — white background, black ink
        darkness: float array (N,) — ink coverage in [0=white, 1=black]
    """
    n = len(charset)
    glyphs = np.zeros((n, cell_h, cell_w), dtype=np.uint8)

    for i, char in enumerate(charset):
        tile = Image.new("L", (cell_w, cell_h), color=255)
        draw = ImageDraw.Draw(tile)
        bbox = draw.textbbox((0, 0), char, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        x = (cell_w - tw) // 2 - bbox[0]
        y = (cell_h - th) // 2 - bbox[1]
        draw.text((x, y), char, fill=0, font=font)
        glyphs[i] = np.array(tile)

    darkness = 1.0 - glyphs.mean(axis=(1, 2)) / 255.0
    return glyphs, darkness


# ─── Image loading ────────────────────────────────────────────────────────────

def load_target(
    image_path: str,
    cols: int,
    cell_w: int,
    cell_h: int,
    char_aspect: Optional[float],
) -> tuple[np.ndarray, int, int]:
    """
    Load target image, derive grid dimensions, resize to exact render resolution.

    char_aspect = cell_w / cell_h corrects for the fact that character cells
    are taller than wide — without it, the ASCII canvas looks vertically squished.

    Returns (target_array float32, rows, cols).
    """
    img = Image.open(image_path).convert("L")
    iw, ih = img.size
    aspect = char_aspect if char_aspect is not None else (cell_w / cell_h)
    rows = max(1, round(ih / iw * cols * aspect))
    img = img.resize((cols * cell_w, rows * cell_h), Image.LANCZOS)
    return np.array(img, dtype=np.float32), rows, cols


# ─── Genome rendering ─────────────────────────────────────────────────────────

def render_genome(
    genome: np.ndarray,          # (rows, cols) int
    glyphs: np.ndarray,          # (N_chars, cell_h, cell_w) uint8
    rows: int,
    cols: int,
    cell_h: int,
    cell_w: int,
) -> np.ndarray:                 # (rows*cell_h, cols*cell_w) uint8
    """
    Build rendered image by pasting glyph tiles.

    glyphs[genome] → (rows, cols, cell_h, cell_w)
    transpose(0,2,1,3) → (rows, cell_h, cols, cell_w)
    reshape → (rows*cell_h, cols*cell_w)
    """
    return glyphs[genome].transpose(0, 2, 1, 3).reshape(rows * cell_h, cols * cell_w)


# ─── Fitness ──────────────────────────────────────────────────────────────────

def compute_fitness(
    genome: np.ndarray,
    glyphs: np.ndarray,
    target: np.ndarray,
    rows: int,
    cols: int,
    cell_h: int,
    cell_w: int,
) -> float:
    """MSE between rendered genome and target. Lower = better."""
    rendered = render_genome(genome, glyphs, rows, cols, cell_h, cell_w).astype(np.float32)
    return float(np.mean((rendered - target) ** 2))


# ─── GA operators ─────────────────────────────────────────────────────────────

def greedy_genome(
    target: np.ndarray,
    rows: int,
    cols: int,
    cell_h: int,
    cell_w: int,
    darkness: np.ndarray,
) -> np.ndarray:
    """
    Build a genome by mapping each cell's average brightness to the nearest
    character by measured darkness. Used for warm-start initialization.
    """
    sorted_dark_first = np.argsort(darkness)[::-1]   # darkest char index first
    genome = np.zeros((rows, cols), dtype=np.int32)
    n = len(darkness)

    for r in range(rows):
        for c in range(cols):
            y0, x0 = r * cell_h, c * cell_w
            brightness = float(target[y0:y0 + cell_h, x0:x0 + cell_w].mean())
            t = brightness / 255.0                    # 0=black, 1=white
            rank = round(t * (n - 1))                 # 0=darkest, n-1=lightest
            genome[r, c] = sorted_dark_first[rank]

    return genome


def tournament_select(
    population: list[np.ndarray],
    fitnesses: np.ndarray,
    k: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Deterministic tournament: pick k random contestants, return the one with lowest MSE."""
    contestants = rng.integers(0, len(population), size=k)
    winner = contestants[np.argmin(fitnesses[contestants])]
    return population[winner]


def crossover_uniform(
    p1: np.ndarray, p2: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Uniform crossover at cell level."""
    mask = rng.random(p1.shape) < 0.5
    return np.where(mask, p1, p2), np.where(mask, p2, p1)


def crossover_block(
    p1: np.ndarray, p2: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Swap a random rectangular sub-grid between two parents."""
    rows, cols = p1.shape
    r1, r2 = sorted(rng.integers(0, rows + 1, size=2))
    c1, c2 = sorted(rng.integers(0, cols + 1, size=2))
    ch1, ch2 = p1.copy(), p2.copy()
    ch1[r1:r2, c1:c2] = p2[r1:r2, c1:c2]
    ch2[r1:r2, c1:c2] = p1[r1:r2, c1:c2]
    return ch1, ch2


def mutate(
    genome: np.ndarray,
    mutation_rate: float,
    n_chars: int,
    darkness: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Per-cell mutation.
    70% of mutations step ±1-2 in darkness rank (local search).
    30% jump to a random character (exploration).
    """
    child = genome.copy()
    mask = rng.random(child.shape) < mutation_rate
    n_mut = int(mask.sum())
    if n_mut == 0:
        return child

    sorted_dark = np.argsort(darkness)           # index i → char index at darkness rank i
    rank_of = np.argsort(sorted_dark)             # rank_of[char_idx] = darkness rank

    current = child[mask]
    use_neighbor = rng.random(n_mut) < 0.7
    new_vals = np.empty(n_mut, dtype=np.int32)

    # Random mutations
    if (~use_neighbor).any():
        new_vals[~use_neighbor] = rng.integers(0, n_chars, size=(~use_neighbor).sum())

    # Neighbor mutations: step ±1 or ±2 in darkness ordering
    if use_neighbor.any():
        cur_ranks = rank_of[current[use_neighbor]]
        steps = rng.integers(-2, 3, size=use_neighbor.sum())   # -2, -1, 0, +1, +2
        new_ranks = np.clip(cur_ranks + steps, 0, n_chars - 1)
        new_vals[use_neighbor] = sorted_dark[new_ranks]

    child[mask] = new_vals
    return child


# ─── ASCIIArtGA class ─────────────────────────────────────────────────────────

class ASCIIArtGA:
    def __init__(
        self,
        config: Config,
        glyphs: np.ndarray,
        darkness: np.ndarray,
        target: np.ndarray,
        rows: int,
        cols: int,
        cell_h: int,
        cell_w: int,
    ):
        self.config = config
        self.glyphs = glyphs
        self.darkness = darkness
        self.target = target
        self.rows = rows
        self.cols = cols
        self.cell_h = cell_h
        self.cell_w = cell_w
        self.n_chars = len(config.charset)
        self.rng = np.random.default_rng(config.seed)

        self.population: list[np.ndarray] = []
        self.fitnesses: np.ndarray = np.empty(0)
        self._best_genome: Optional[np.ndarray] = None
        self._best_fitness: float = float("inf")

    @property
    def best(self) -> tuple[np.ndarray, float]:
        return self._best_genome, self._best_fitness

    def _eval(self, genome: np.ndarray) -> float:
        return compute_fitness(
            genome, self.glyphs, self.target,
            self.rows, self.cols, self.cell_h, self.cell_w,
        )

    def _sync_best(self):
        idx = int(np.argmin(self.fitnesses))
        if self.fitnesses[idx] < self._best_fitness:
            self._best_fitness = float(self.fitnesses[idx])
            self._best_genome = self.population[idx].copy()

    def initialize(self):
        """Half warm-start (greedy + noise at increasing rates), half random."""
        cfg = self.config
        n_greedy = cfg.population // 2
        base = greedy_genome(
            self.target, self.rows, self.cols,
            self.cell_h, self.cell_w, self.darkness,
        )
        pop: list[np.ndarray] = []
        for i in range(n_greedy):
            noise = 0.05 + 0.45 * (i / max(n_greedy - 1, 1))
            pop.append(mutate(base, noise, self.n_chars, self.darkness, self.rng))
        for _ in range(cfg.population - n_greedy):
            pop.append(self.rng.integers(0, self.n_chars, size=(self.rows, self.cols)))

        self.population = pop
        self.fitnesses = np.array([self._eval(ind) for ind in pop])
        self._sync_best()

    def step(self) -> tuple[float, float]:
        """Run one generation. Returns (best_fitness, mean_fitness)."""
        cfg = self.config

        # Elitism
        elite_idx = np.argsort(self.fitnesses)[: cfg.elite]
        offspring: list[np.ndarray] = [self.population[i].copy() for i in elite_idx]
        off_fits: list[float] = list(self.fitnesses[elite_idx])

        while len(offspring) < cfg.population:
            p1 = tournament_select(self.population, self.fitnesses, cfg.tournament_k, self.rng)
            p2 = tournament_select(self.population, self.fitnesses, cfg.tournament_k, self.rng)

            if self.rng.random() < cfg.crossover_prob:
                cx = crossover_uniform if self.rng.random() < 0.5 else crossover_block
                c1, c2 = cx(p1, p2, self.rng)
            else:
                c1, c2 = p1.copy(), p2.copy()

            for child in (c1, c2):
                if len(offspring) >= cfg.population:
                    break
                child = mutate(child, cfg.mutation, self.n_chars, self.darkness, self.rng)
                offspring.append(child)
                off_fits.append(self._eval(child))

        self.population = offspring[: cfg.population]
        self.fitnesses = np.array(off_fits[: cfg.population])
        self._sync_best()

        return self._best_fitness, float(self.fitnesses.mean())


# ─── Output helpers ───────────────────────────────────────────────────────────

def genome_to_text(genome: np.ndarray, charset: str) -> str:
    rows, cols = genome.shape
    return "\n".join(
        "".join(charset[genome[r, c]] for c in range(cols))
        for r in range(rows)
    )


def save_result(
    genome: np.ndarray,
    charset: str,
    glyphs: np.ndarray,
    rows: int,
    cols: int,
    cell_h: int,
    cell_w: int,
    out_dir: Path,
    prefix: str = "best",
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    txt_path = out_dir / f"{prefix}.txt"
    png_path = out_dir / f"{prefix}.png"

    txt_path.write_text(genome_to_text(genome, charset), encoding="utf-8")

    rendered = render_genome(genome, glyphs, rows, cols, cell_h, cell_w)
    Image.fromarray(rendered, mode="L").save(png_path)

    return txt_path, png_path


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> Config:
    p = argparse.ArgumentParser(
        description="ASCII Art via Genetic Algorithm",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("image", help="Input image path")
    p.add_argument("--cols",        type=int,   default=80,        help="ASCII columns (default: 80)")
    p.add_argument("--population",  type=int,   default=80,        help="Population size (default: 80)")
    p.add_argument("--generations", type=int,   default=500,       help="Generations (default: 500)")
    p.add_argument("--mutation",    type=float, default=0.02,      help="Per-cell mutation rate (default: 0.02)")
    p.add_argument("--font",        dest="font_path", default=None, help="Path to TTF monospace font")
    p.add_argument("--font-size",   type=int,   default=12,        help="Font size in points (default: 12)")
    p.add_argument("--save-every",  type=int,   default=50,        help="Snapshot interval in gens (default: 50)")
    p.add_argument("--output",      default="output",              help="Output directory (default: output)")
    p.add_argument("--elite",       type=int,   default=5,         help="Elite count (default: 5)")
    p.add_argument("--tournament-k",type=int,   default=5,         help="Tournament size (default: 5)")
    p.add_argument("--charset",     default="@%#*+=-:. ",          help="Character set, dark→light")
    p.add_argument("--char-aspect", type=float, default=None,      help="cell_w/cell_h override (default: auto)")
    p.add_argument("--gif",         action="store_true",           help="Save evolution.gif (needs imageio)")
    p.add_argument("--seed",        type=int,   default=42)
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

    # ── Font & glyph cache ──
    print("Loading font...")
    font = load_font(cfg.font_path, cfg.font_size)
    cell_w, cell_h = get_cell_size(font)
    print(f"  Cell size: {cell_w}×{cell_h} px")

    print(f"Building glyph cache  charset={repr(cfg.charset)}")
    glyphs, darkness = build_glyph_cache(cfg.charset, font, cell_w, cell_h)

    sorted_dark_first = np.argsort(darkness)[::-1]
    ordered = "".join(cfg.charset[i] for i in sorted_dark_first)
    print(f"  Measured darkness order (dark→light): {repr(ordered)}")

    # ── Target image ──
    print(f"Loading target: {cfg.image_path}")
    target, rows, cols = load_target(
        cfg.image_path, cfg.cols, cell_w, cell_h, cfg.char_aspect
    )
    print(f"  Grid: {rows} rows × {cols} cols  |  Render: {cols*cell_w}×{rows*cell_h} px")

    # ── Initialize ──
    print(f"\nInitializing population ({cfg.population} individuals)...")
    ga = ASCIIArtGA(cfg, glyphs, darkness, target, rows, cols, cell_h, cell_w)
    ga.initialize()
    print(f"  Initial best MSE: {ga.best[1]:.2f}")

    out_dir = Path(cfg.output_dir)
    snap_dir = out_dir / "snapshots"
    gif_frames: list[np.ndarray] = []

    # ── Evolution loop ──
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
            print(f"    → snapshot saved: {png}")

            if cfg.gif:
                rendered = render_genome(best_genome, glyphs, rows, cols, cell_h, cell_w)
                gif_frames.append(rendered.copy())

    # ── Final output ──
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
