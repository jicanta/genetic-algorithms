"""
TriangleGA — GA engine with configurable operators and survival strategies.

Supported selection methods : tournament_det | tournament_prob | roulette |
                               universal | boltzmann | ranking
Supported crossover methods : uniform | one_point | two_point | annular
Supported mutation methods  : uniform | gen | multigen | non_uniform
Survival strategies         : exclusive | additive
"""

from typing import Optional

import numpy as np

from .config import Config
from .fitness import compute_fitness
from .genome import random_genome
from .operators import (
    # selection
    tournament_det, tournament_prob, roulette, universal, boltzmann, ranking,
    # crossover
    crossover_uniform, crossover_one_point, crossover_two_point, crossover_annular,
    # mutation
    mutate_uniform, mutate_gen, mutate_multigen, mutate_non_uniform,
)

_SELECTION = {
    "tournament_det":  tournament_det,
    "tournament_prob": tournament_prob,
    "roulette":        roulette,
    "universal":       universal,
    "boltzmann":       boltzmann,
    "ranking":         ranking,
}

_CROSSOVER = {
    "uniform":    crossover_uniform,
    "one_point":  crossover_one_point,
    "two_point":  crossover_two_point,
    "annular":    crossover_annular,
}

_MUTATION = {
    "uniform":     mutate_uniform,
    "gen":         mutate_gen,
    "multigen":    mutate_multigen,
    "non_uniform": mutate_non_uniform,
}


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
            config:  All hyperparameters and strategy choices.
            target:  float32 (img_h, img_w, 3) — preprocessed target image.
            img_w:   Canvas width in pixels.
            img_h:   Canvas height in pixels.
        """
        self.config = config
        self.target = target
        self.img_w = img_w
        self.img_h = img_h
        self.rng = np.random.default_rng(config.seed)
        self._generation = 0

        if config.selection_method not in _SELECTION:
            raise ValueError(f"Unknown selection method: {config.selection_method!r}. "
                             f"Choose from: {list(_SELECTION)}")
        if config.crossover_method not in _CROSSOVER:
            raise ValueError(f"Unknown crossover method: {config.crossover_method!r}. "
                             f"Choose from: {list(_CROSSOVER)}")
        if config.mutation_method not in _MUTATION:
            raise ValueError(f"Unknown mutation method: {config.mutation_method!r}. "
                             f"Choose from: {list(_MUTATION)}")
        if config.survival_strategy not in ("exclusive", "additive"):
            raise ValueError(f"Unknown survival strategy: {config.survival_strategy!r}. "
                             f"Choose from: exclusive | additive")

        self._select_fn  = _SELECTION[config.selection_method]
        self._cross_fn   = _CROSSOVER[config.crossover_method]
        self._mutate_fn  = _MUTATION[config.mutation_method]

        self.population: list[np.ndarray] = []
        self.fitnesses: np.ndarray = np.empty(0)
        self._best_genome: Optional[np.ndarray] = None
        self._best_fitness: float = float("inf")
        self.history: list[dict[str, float]] = []

        # Termination tracking
        self._stagnation_counter: int = 0
        self._last_improved_fitness: float = float("inf")

    @property
    def best(self) -> tuple[np.ndarray, float]:
        """Current best genome and its MSE fitness."""
        return self._best_genome, self._best_fitness

    def initialize(self) -> None:
        """Create a fully random initial population and evaluate fitness."""
        cfg = self.config
        self.population = [
            random_genome(cfg.n_triangles, self.rng)
            for _ in range(cfg.population)
        ]
        self.fitnesses = np.array([self._eval(ind) for ind in self.population])
        self._sync_best()
        self._record_history(generation=0)

    def step(self) -> tuple[float, float]:
        """
        Run one generation.

        Returns:
            (best_fitness, mean_fitness) after this generation.
        """
        cfg = self.config
        gen = self._generation

        # --- Always preserve elite individuals ---
        elite_idx = np.argsort(self.fitnesses)[: cfg.elite]
        elite = [self.population[i].copy() for i in elite_idx]
        elite_fits = list(self.fitnesses[elite_idx])

        # --- Generate offspring ---
        offspring, off_fits = self._generate_offspring(cfg.population - cfg.elite)

        # --- Survival strategy ---
        if cfg.survival_strategy == "exclusive":
            # Full replacement: new generation = elite + offspring only
            self.population = elite + offspring
            self.fitnesses = np.array(elite_fits + off_fits)

        else:  # additive
            # Pool parents and offspring, keep best N
            combined = self.population + offspring
            combined_fits = np.concatenate([self.fitnesses, np.array(off_fits)])
            survivors = np.argsort(combined_fits)[: cfg.population]
            self.population = [combined[i] for i in survivors]
            self.fitnesses = combined_fits[survivors]

        self._sync_best()
        self._generation += 1
        self._record_history(generation=self._generation)
        return self._best_fitness, float(self.fitnesses.mean())

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _select(self) -> np.ndarray:
        """Select one parent using the configured selection method."""
        cfg = self.config
        temp = self._boltzmann_temperature()
        return self._select_fn(
            population=self.population,
            fitnesses=self.fitnesses,
            k=cfg.tournament_k,
            prob=cfg.tournament_prob,
            temperature=temp,
            rng=self.rng,
        )

    def _mutate(self, genome: np.ndarray) -> np.ndarray:
        """Mutate a genome using the configured mutation method."""
        cfg = self.config
        return self._mutate_fn(
            genome=genome,
            rng=self.rng,
            mutation_rate=cfg.mutation_rate,
            mutation_sigma=cfg.mutation_sigma,
            max_genes=cfg.multigen_max_genes,
            generation=self._generation,
            max_generations=cfg.generations,
        )

    def _generate_offspring(self, n: int) -> tuple[list[np.ndarray], list[float]]:
        """Produce n offspring via selection → crossover → mutation."""
        cfg = self.config
        offspring: list[np.ndarray] = []
        fits: list[float] = []

        while len(offspring) < n:
            p1 = self._select()
            p2 = self._select()

            if self.rng.random() < cfg.crossover_prob:
                c1, c2 = self._cross_fn(p1, p2, self.rng)
            else:
                c1, c2 = p1.copy(), p2.copy()

            for child in (c1, c2):
                if len(offspring) >= n:
                    break
                child = self._mutate(child)
                offspring.append(child)
                fits.append(self._eval(child))

        return offspring, fits

    def _boltzmann_temperature(self) -> float:
        """
        Linear temperature decay for Boltzmann selection.

        T decreases from boltzmann_temp_init to boltzmann_temp_min over all generations.
        """
        cfg = self.config
        progress = self._generation / max(cfg.generations - 1, 1)
        return cfg.boltzmann_temp_init + progress * (cfg.boltzmann_temp_min - cfg.boltzmann_temp_init)

    def should_stop(self) -> tuple[bool, str]:
        """
        Check termination criteria other than max generations.

        Returns:
            (stop: bool, reason: str)

        Stagnation (content): the best fitness hasn't improved by at least
        `stagnation_delta` in the last `stagnation_gens` consecutive generations.
        Tracks real progress, not just noise — small oscillations don't reset the counter.

        Convergence (structure): the standard deviation of the population's fitnesses
        drops below `convergence_threshold`. When all individuals score similarly,
        the population has collapsed and further evolution yields diminishing returns.
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

    def _eval(self, genome: np.ndarray) -> float:
        return compute_fitness(genome, self.target, self.img_w, self.img_h)

    def _sync_best(self) -> None:
        idx = int(np.argmin(self.fitnesses))
        current = float(self.fitnesses[idx])
        if current < self._best_fitness:
            self._best_fitness = current
            self._best_genome = self.population[idx].copy()

        # Update stagnation counter
        if self._last_improved_fitness - self._best_fitness >= self.config.stagnation_delta:
            self._stagnation_counter = 0
            self._last_improved_fitness = self._best_fitness
        else:
            self._stagnation_counter += 1

    def _record_history(self, generation: int) -> None:
        """Store per-generation metrics for later plotting/export."""
        self.history.append({
            "generation": float(generation),
            "best": float(self._best_fitness),
            "mean": float(self.fitnesses.mean()),
            "std": float(self.fitnesses.std()),
        })
