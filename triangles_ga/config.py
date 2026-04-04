from dataclasses import dataclass
from typing import Optional


@dataclass
class Config:
    image_path: str
    n_triangles: int = 50
    population: int = 80
    generations: int = 500
    mutation_rate: float = 0.02    # probability per gene of being mutated
    mutation_sigma: float = 0.05   # std of Gaussian noise applied on mutation
    elite: int = 5
    tournament_k: int = 5
    crossover_prob: float = 0.8
    save_every: int = 50
    output_dir: str = "output_triangles"
    seed: int = 42
    img_size: Optional[int] = None  # resize longest side to this (None = keep original)
