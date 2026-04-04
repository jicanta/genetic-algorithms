"""
TriangleGA — the main GA engine for triangle-based image approximation.

Owns the population and fitness array. The caller drives evolution by
calling step() once per generation.

Survival strategy: exclusive replacement (full generational replacement
with elitism). Additional strategies will be added in the next step.
"""

from typing import Optional

import numpy as np

from .config import Config
from .fitness import compute_fitness
from .genome import random_genome
from .operators import tournament_select, crossover_uniform, mutate


class TriangleGA:
    def __init__(
        self,
        config: Config,
        target: np.ndarray,
        img_w: int,
        img_h: int,
    ):
        """
        Args:
            config:  Hyperparameters.
            target:  float32 (img_h, img_w, 3) — preprocessed target image.
            img_w:   Canvas width in pixels.
            img_h:   Canvas height in pixels.
        """
        self.config = config
        self.target = target
        self.img_w = img_w
        self.img_h = img_h
        self.rng = np.random.default_rng(config.seed)

        self.population: list[np.ndarray] = []
        self.fitnesses: np.ndarray = np.empty(0)
        self._best_genome: Optional[np.ndarray] = None
        self._best_fitness: float = float("inf")

    @property
    def best(self) -> tuple[np.ndarray, float]:
        """Current best genome and its MSE fitness."""
        return self._best_genome, self._best_fitness

    def initialize(self) -> None:
        """Create a fully random initial population and evaluate it."""
        cfg = self.config
        self.population = [
            random_genome(cfg.n_triangles, self.rng)
            for _ in range(cfg.population)
        ]
        self.fitnesses = np.array([self._eval(ind) for ind in self.population])
        self._sync_best()

    def step(self) -> tuple[float, float]:
        """
        Run one generation.

        Pipeline:
          1. Copy elite individuals unchanged.
          2. Fill the rest: tournament selection → crossover → mutation.
          3. Replace old population with offspring (exclusive replacement).
          4. Update best-so-far tracker.

        Returns:
            (best_fitness, mean_fitness) after this generation.
        """
        cfg = self.config

        elite_idx = np.argsort(self.fitnesses)[: cfg.elite]
        offspring: list[np.ndarray] = [self.population[i].copy() for i in elite_idx]
        off_fits: list[float] = list(self.fitnesses[elite_idx])

        while len(offspring) < cfg.population:
            p1 = tournament_select(self.population, self.fitnesses, cfg.tournament_k, self.rng)
            p2 = tournament_select(self.population, self.fitnesses, cfg.tournament_k, self.rng)

            if self.rng.random() < cfg.crossover_prob:
                c1, c2 = crossover_uniform(p1, p2, self.rng)
            else:
                c1, c2 = p1.copy(), p2.copy()

            for child in (c1, c2):
                if len(offspring) >= cfg.population:
                    break
                child = mutate(child, cfg.mutation_rate, cfg.mutation_sigma, self.rng)
                offspring.append(child)
                off_fits.append(self._eval(child))

        self.population = offspring[: cfg.population]
        self.fitnesses = np.array(off_fits[: cfg.population])
        self._sync_best()

        return self._best_fitness, float(self.fitnesses.mean())

    def _eval(self, genome: np.ndarray) -> float:
        return compute_fitness(genome, self.target, self.img_w, self.img_h)

    def _sync_best(self) -> None:
        idx = int(np.argmin(self.fitnesses))
        if self.fitnesses[idx] < self._best_fitness:
            self._best_fitness = float(self.fitnesses[idx])
            self._best_genome = self.population[idx].copy()
