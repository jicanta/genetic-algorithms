from dataclasses import dataclass
from typing import Optional


@dataclass
class Config:
    image_path: str
    cols: int = 80
    population: int = 80
    generations: int = 500
    mutation: float = 0.02
    crossover_prob: float = 0.8
    font_path: Optional[str] = None
    font_size: int = 12
    save_every: int = 50
    output_dir: str = "output"
    elite: int = 5
    tournament_k: int = 5
    charset: str = "@%#*+=-:. "
    char_aspect: Optional[float] = None  # None = auto from font metrics
    gif: bool = False
    seed: int = 42

    # --- Termination criteria (opt-in) ---
    stop_on_stagnation: bool = False
    stagnation_gens: int = 50         # consecutive gens without sufficient improvement
    stagnation_delta: float = 0.5     # minimum MSE improvement to reset counter

    stop_on_convergence: bool = False
    convergence_threshold: float = 5.0  # std of fitnesses below this → converged

    def __post_init__(self) -> None:
        if self.population < 2:
            raise ValueError(f"population must be >= 2, got {self.population}")
        if self.elite < 0 or self.elite >= self.population:
            raise ValueError(f"elite must be in [0, population), got {self.elite}")
        if self.tournament_k < 1:
            raise ValueError(f"tournament_k must be >= 1, got {self.tournament_k}")
        if self.generations < 1:
            raise ValueError(f"generations must be >= 1, got {self.generations}")
        if not 0.0 <= self.mutation <= 1.0:
            raise ValueError(f"mutation must be in [0, 1], got {self.mutation}")
        if not 0.0 <= self.crossover_prob <= 1.0:
            raise ValueError(f"crossover_prob must be in [0, 1], got {self.crossover_prob}")
        if self.cols < 1:
            raise ValueError(f"cols must be >= 1, got {self.cols}")
        if not self.charset:
            raise ValueError("charset must contain at least one character")
        if self.stagnation_gens < 1:
            raise ValueError(f"stagnation_gens must be >= 1, got {self.stagnation_gens}")
        if self.convergence_threshold <= 0.0:
            raise ValueError(f"convergence_threshold must be > 0, got {self.convergence_threshold}")
