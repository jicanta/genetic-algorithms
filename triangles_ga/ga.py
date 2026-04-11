"""
TriangleGA — GA engine with configurable operators and survival strategies.

Supported selection methods : tournament_det | tournament_prob | roulette |
                               universal | boltzmann | ranking
Supported crossover methods : uniform | one_point | two_point | annular
Supported mutation methods  : uniform | gen | multigen | non_uniform
Survival strategies         : exclusive | additive
"""

import os
import sys
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from typing import Optional

import numpy as np

# ── Worker-side globals ───────────────────────────────────────────────────────
# The target image and image dimensions are stored once per worker process via
# the pool initializer so they never have to be pickled for every map() call.
_W_target: Optional[np.ndarray] = None
_W_img_w:  int = 0
_W_img_h:  int = 0
_W_sample_mask: Optional[np.ndarray] = None  # flat boolean mask for pixel subsampling


def _worker_init(target: np.ndarray, img_w: int, img_h: int,
                 sample_mask: Optional[np.ndarray],
                 renderer: str = "auto") -> None:
    global _W_target, _W_img_w, _W_img_h, _W_sample_mask
    _W_target = target
    _W_img_w  = img_w
    _W_img_h  = img_h
    _W_sample_mask = sample_mask
    from .render import set_backend
    set_backend(renderer)


def _eval_genome(genome: np.ndarray) -> float:
    """Evaluate a genome using worker-local target (no per-call pickling)."""
    from .render import render_genome
    rendered = render_genome(genome, _W_img_w, _W_img_h)
    if _W_sample_mask is not None:
        diff = rendered.reshape(-1, 3)[_W_sample_mask] - _W_target.reshape(-1, 3)[_W_sample_mask]
    else:
        diff = rendered - _W_target
    return float(np.mean(diff ** 2))

from .config import Config
from .fitness import compute_fitness
from .genome import random_genome, color_sampled_genome

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
        if config.renderer not in ("auto", "skia", "pil"):
            raise ValueError(f"Unknown renderer: {config.renderer!r}. "
                             f"Choose from: auto | skia | pil")

        self._select_fn  = _SELECTION[config.selection_method]
        self._cross_fn   = _CROSSOVER[config.crossover_method]
        self._mutate_fn  = _MUTATION[config.mutation_method]

        # Number of parallel workers for fitness evaluation.
        # 0 (default) → use all CPU cores; 1 → disable parallelism.
        self._workers: int = config.workers if config.workers > 0 else (os.cpu_count() or 1)

        # Pixel subsampling mask — evaluated once and shared with workers.
        n_pixels = img_h * img_w
        if 0.0 < config.fitness_sample < 1.0:
            rng_mask = np.random.default_rng(config.seed + 9999)
            sample_mask: Optional[np.ndarray] = rng_mask.random(n_pixels) < config.fitness_sample
        else:
            sample_mask = None
        self._sample_mask = sample_mask

        # Set worker globals for the main-process (single-worker) path.
        _worker_init(target, img_w, img_h, sample_mask, config.renderer)

        # Persistent process pool — created once, reused every generation.
        #
        # Strategy depends on the renderer:
        #   Skia  → always use 'spawn': Skia's internal GPU context doesn't survive
        #            fork(), so forked workers silently fall back to slow paths.
        #            The target is sent once via the pool initializer.
        #
        #   PIL   → prefer 'fork' on Linux: child processes inherit the target
        #            array via COW with zero serialisation cost (no IPC at all).
        #            On macOS/Windows, fall back to spawn+initializer.
        if self._workers > 1:
            from .render import _HAVE_SKIA
            use_skia = (config.renderer == "skia") or (config.renderer == "auto" and _HAVE_SKIA)
            if use_skia or sys.platform != "linux":
                # spawn: fresh worker, Skia initialises correctly
                self._pool: Optional[ProcessPoolExecutor] = ProcessPoolExecutor(
                    max_workers=self._workers,
                    initializer=_worker_init,
                    initargs=(target, img_w, img_h, sample_mask, config.renderer),
                )
            else:
                # fork on Linux with PIL: zero-cost COW target sharing
                ctx = multiprocessing.get_context("fork")
                self._pool = ProcessPoolExecutor(
                    max_workers=self._workers, mp_context=ctx
                )
        else:
            self._pool = None

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
        """Create the initial population and evaluate fitness.

        With init_method='random' (default) all genes are uniformly random.
        With init_method='color_sample' vertex positions and alpha are still
        random, but each triangle's RGB is sampled from the target at the
        centroid of its vertices — giving the GA a much better color starting
        point without biasing the geometry search.
        """
        cfg = self.config
        if cfg.init_method == "color_sample":
            self.population = [
                color_sampled_genome(cfg.n_triangles, self.rng,
                                     self.target, self.img_w, self.img_h)
                for _ in range(cfg.population)
            ]
        else:
            self.population = [
                random_genome(cfg.n_triangles, self.rng)
                for _ in range(cfg.population)
            ]
        self.fitnesses = np.array(self._eval_batch(self.population))
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
        """Produce n offspring via selection → crossover → mutation, then evaluate in parallel."""
        cfg = self.config
        offspring: list[np.ndarray] = []

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
                offspring.append(self._mutate(child))

        fits = self._eval_batch(offspring)
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
        """Single-genome eval (in-process, no pool overhead)."""
        return self._eval_batch([genome])[0]

    def _eval_batch(self, genomes: list[np.ndarray]) -> list[float]:
        """Evaluate a batch of genomes. Uses the persistent pool when available."""
        if self._pool is None or len(genomes) <= 1:
            # In-process path: call the worker function directly with local target
            _worker_init(self.target, self.img_w, self.img_h, self._sample_mask)
            return [_eval_genome(g) for g in genomes]
        chunksize = max(1, len(genomes) // self._workers)
        return list(self._pool.map(_eval_genome, genomes, chunksize=chunksize))

    def shutdown(self) -> None:
        """Shut down the worker pool. Call when the GA is done."""
        if self._pool is not None:
            self._pool.shutdown(wait=False)
            self._pool = None

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
