from dataclasses import dataclass
from typing import Optional


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
    char_aspect: Optional[float] = None  # None = auto from font metrics
    gif: bool = False
    seed: int = 42
