from dataclasses import dataclass
from typing import Optional


@dataclass
class Config:
    image_path: str

    def __post_init__(self) -> None:
        if self.population < 2:
            raise ValueError(f"population must be >= 2, got {self.population}")
        if self.elite < 0 or self.elite >= self.population:
            raise ValueError(f"elite must be in [0, population), got {self.elite}")
        if self.generations < 1:
            raise ValueError(f"generations must be >= 1, got {self.generations}")
        if not 0.0 <= self.mutation_rate <= 1.0:
            raise ValueError(f"mutation_rate must be in [0, 1], got {self.mutation_rate}")
        if self.mutation_sigma < 0.0:
            raise ValueError(f"mutation_sigma must be >= 0, got {self.mutation_sigma}")
        if not 0.0 <= self.crossover_prob <= 1.0:
            raise ValueError(f"crossover_prob must be in [0, 1], got {self.crossover_prob}")
        if self.n_triangles < 1:
            raise ValueError(f"n_triangles must be >= 1, got {self.n_triangles}")
        if self.n_triangles < 2 and self.crossover_method in ("one_point", "annular"):
            raise ValueError(
                f"crossover_method={self.crossover_method!r} requires at least 2 triangles"
            )
        if self.init_method not in ("random", "color_sample", "mixed"):
            raise ValueError(
                f"init_method must be 'random', 'color_sample', or 'mixed', got {self.init_method!r}"
            )
        if self.shape not in ("triangle", "oval"):
            raise ValueError(f"shape must be 'triangle' or 'oval', got {self.shape!r}")
        if self.target_mse is not None and self.target_mse < 0.0:
            raise ValueError(f"target_mse must be >= 0 when provided, got {self.target_mse}")
        if self.tournament_k < 1:
            raise ValueError(f"tournament_k must be >= 1, got {self.tournament_k}")
        if not 0.0 < self.tournament_prob <= 1.0:
            raise ValueError(f"tournament_prob must be in (0, 1], got {self.tournament_prob}")
        if self.boltzmann_temp_init <= 0.0 or self.boltzmann_temp_min <= 0.0:
            raise ValueError("Boltzmann temperatures must be > 0")
        if self.multigen_max_genes < 1:
            raise ValueError(f"multigen_max_genes must be >= 1, got {self.multigen_max_genes}")
        if self.geometry_mutation_scale < 0.0:
            raise ValueError(f"geometry_mutation_scale must be >= 0, got {self.geometry_mutation_scale}")
        if self.color_mutation_scale < 0.0:
            raise ValueError(f"color_mutation_scale must be >= 0, got {self.color_mutation_scale}")
        if self.alpha_mutation_scale < 0.0:
            raise ValueError(f"alpha_mutation_scale must be >= 0, got {self.alpha_mutation_scale}")
        if not 0.0 <= self.layer_mutation_rate <= 1.0:
            raise ValueError(f"layer_mutation_rate must be in [0, 1], got {self.layer_mutation_rate}")
        if self.layer_mutation_max_shift < 1:
            raise ValueError(f"layer_mutation_max_shift must be >= 1, got {self.layer_mutation_max_shift}")
        if self.img_size is not None and self.img_size < 1:
            raise ValueError(f"img_size must be >= 1 when provided, got {self.img_size}")
        if self.stagnation_gens < 1:
            raise ValueError(f"stagnation_gens must be >= 1, got {self.stagnation_gens}")
        if self.convergence_threshold <= 0.0:
            raise ValueError(f"convergence_threshold must be > 0, got {self.convergence_threshold}")
        if self.workers < 0:
            raise ValueError(f"workers must be >= 0, got {self.workers}")
        if not 0.0 < self.fitness_sample <= 1.0:
            raise ValueError(f"fitness_sample must be in (0, 1], got {self.fitness_sample}")
        if self.saliency_weight < 0.0:
            raise ValueError(f"saliency_weight must be >= 0, got {self.saliency_weight}")
        if self.save_every < 1:
            raise ValueError(f"save_every must be >= 1, got {self.save_every}")

    # --- Problem parameters ---
    n_triangles: int = 50
    img_size: Optional[int] = None       # resize longest side to this (None = keep original)
    init_method: str = "mixed"           # initial population: random | color_sample | mixed
    shape: str = "triangle"             # shape primitive: triangle | oval

    # --- GA core ---
    population: int = 80
    generations: int = 500
    elite: int = 5                       # always preserved regardless of strategy

    # --- Selection ---
    selection_method: str = "tournament_det"
    # Options: tournament_det | tournament_prob | roulette | universal | boltzmann | ranking
    tournament_k: int = 5                # tournament size (both tournament variants)
    tournament_prob: float = 0.75        # win probability for probabilistic tournament
    boltzmann_temp_init: float = 100.0   # starting temperature for Boltzmann
    boltzmann_temp_min: float = 1.0      # minimum temperature (never reaches 0)

    # --- Crossover ---
    crossover_method: str = "uniform"
    # Options: uniform | one_point | two_point | annular
    crossover_prob: float = 0.8

    # --- Mutation ---
    mutation_method: str = "uniform"
    # Options: uniform | gen | multigen | non_uniform
    mutation_rate: float = 0.02          # probability per gene (uniform / non_uniform / multigen)
    mutation_sigma: float = 0.05         # noise std (uniform / non_uniform / multigen)
    multigen_max_genes: int = 5          # max genes mutated per individual (multigen)
    geometry_mutation_scale: float = 1.0 # sigma multiplier for vertex coordinates
    color_mutation_scale: float = 1.0    # sigma multiplier for RGB genes
    alpha_mutation_scale: float = 1.0    # sigma multiplier for opacity genes
    layer_mutation_rate: float = 0.02    # per-individual chance to mutate draw order
    layer_mutation_max_shift: int = 8    # max positions for move-order mutation

    # --- Survival strategy ---
    survival_strategy: str = "exclusive"
    # Options: exclusive | additive

    # --- Termination criteria ---
    # Max generations is always active (the `generations` field above).
    # The two below are opt-in.
    stop_on_stagnation: bool = False     # content: stop if no improvement for N gens
    stagnation_gens: int = 50            # how many gens without improvement trigger stop
    stagnation_delta: float = 0.5        # minimum MSE improvement to reset the counter

    stop_on_convergence: bool = False    # structure: stop if population diversity collapses
    convergence_threshold: float = 5.0  # std of fitnesses below this → converged

    target_mse: Optional[float] = None  # stop when best MSE reaches this value (None = disabled)

    # --- Performance ---
    workers: int = 1        # parallel processes for fitness eval; 1 = single-threaded (default), 0 = all CPU cores
    fitness_sample: float = 1.0  # fraction of pixels used for MSE (1.0 = all pixels)
    saliency_weight: float = 0.0 # extra weight for bright/saturated target pixels
    renderer: str = "auto"  # rendering backend: auto | skia | pil | numba
    fast_fitness: bool = False  # use Numba JIT MSE (requires numba)

    # --- I/O ---
    save_every: int = 50
    output_dir: str = "output_triangles"
    no_plots: bool = False
    graphs_only: bool = False
    seed: int = 42
