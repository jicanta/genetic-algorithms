from dataclasses import dataclass, field
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
        if not 0.0 <= self.crossover_prob <= 1.0:
            raise ValueError(f"crossover_prob must be in [0, 1], got {self.crossover_prob}")
        if self.n_triangles < 1:
            raise ValueError(f"n_triangles must be >= 1, got {self.n_triangles}")
        if self.stagnation_gens < 1:
            raise ValueError(f"stagnation_gens must be >= 1, got {self.stagnation_gens}")
        if self.convergence_threshold <= 0.0:
            raise ValueError(f"convergence_threshold must be > 0, got {self.convergence_threshold}")

    # --- Problem parameters ---
    n_triangles: int = 50
    img_size: Optional[int] = None       # resize longest side to this (None = keep original)

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

    # --- I/O ---
    save_every: int = 50
    output_dir: str = "output_triangles"
    no_plots: bool = False
    graphs_only: bool = False
    seed: int = 42
