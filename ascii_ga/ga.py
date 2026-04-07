"""
ASCIIArtGA — the main GA engine.

Owns the population, fitness array, and best-so-far tracking.
The caller drives the loop by calling step() once per generation.
"""

from typing import Optional

import numpy as np

from .config import Config
from .fitness import compute_fitness
from .operators import greedy_genome, tournament_select, crossover_uniform, crossover_block, mutate


class ASCIIArtGA:
    def __init__(
        self,
        config: Config,
        glyphs: np.ndarray,    # (N_chars, cell_h, cell_w) — precomputed glyph tiles
        darkness: np.ndarray,  # (N_chars,) — measured ink coverage per char
        target: np.ndarray,    # (render_h, render_w) float32 — preprocessed target image
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

        # Termination tracking
        self._stagnation_counter: int = 0
        self._last_improved_fitness: float = float("inf")

    @property
    def best(self) -> tuple[np.ndarray, float]:
        """Current best genome and its fitness (MSE)."""
        return self._best_genome, self._best_fitness

    def initialize(self):
        """
        Seed the population with a mix of warm-start and random individuals.

        Half the population starts from a greedy brightness-mapped genome with
        increasing amounts of noise — giving the GA a head start while keeping
        diversity. The other half is fully random for broader exploration.
        """
        cfg = self.config
        n_greedy = cfg.population // 2
        base = greedy_genome(
            self.target, self.rows, self.cols, self.cell_h, self.cell_w, self.darkness
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
        """
        Run one generation.

        Pipeline:
          1. Copy elite individuals unchanged.
          2. Fill the rest via tournament selection → crossover → mutation.
          3. Replace old population with offspring.
          4. Update best-so-far tracker.

        Returns (best_fitness, mean_fitness) for this generation.
        """
        cfg = self.config

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

    def should_stop(self) -> tuple[bool, str]:
        """
        Check optional termination criteria (beyond max generations).

        Returns:
            (stop: bool, reason: str)

        Stagnation (content): best fitness hasn't improved by at least
        `stagnation_delta` for `stagnation_gens` consecutive generations.

        Convergence (structure): std of population fitnesses drops below
        `convergence_threshold`, meaning the population has collapsed.
        """
        cfg = self.config

        if cfg.stop_on_stagnation:
            if self._stagnation_counter >= cfg.stagnation_gens:
                return True, (
                    f"stagnation: no improvement > {cfg.stagnation_delta} "
                    f"for {cfg.stagnation_gens} generations"
                )

        if cfg.stop_on_convergence:
            diversity = float(self.fitnesses.std())
            if diversity < cfg.convergence_threshold:
                return True, (
                    f"convergence: fitness std={diversity:.2f} "
                    f"< threshold={cfg.convergence_threshold}"
                )

        return False, ""

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _eval(self, genome: np.ndarray) -> float:
        return compute_fitness(
            genome, self.glyphs, self.target,
            self.rows, self.cols, self.cell_h, self.cell_w,
        )

    def _sync_best(self) -> None:
        idx = int(np.argmin(self.fitnesses))
        current = float(self.fitnesses[idx])
        if current < self._best_fitness:
            self._best_fitness = current
            self._best_genome = self.population[idx].copy()

        if self._last_improved_fitness - self._best_fitness >= self.config.stagnation_delta:
            self._stagnation_counter = 0
            self._last_improved_fitness = self._best_fitness
        else:
            self._stagnation_counter += 1
