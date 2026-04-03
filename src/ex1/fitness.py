"""
Fitness function for Exercise 1: ASCII Art

Por qué MAE en vez de MSE:
- MSE/255² da valores artificialmente altos incluso para imágenes muy distintas,
  porque en la práctica el error medio nunca se acerca a 255.
  Ej: imagen aleatoria vs imagen blanca → MSE ≈ 16000 → fitness = 0.75 (engañoso)

- MAE/255 es más honesto:
  Ej: imagen aleatoria vs imagen blanca → MAE ≈ 128 → fitness = 0.50 (intuitivo)
  Interpreta directamente "qué fracción del rango de brillo estoy equivocado en promedio"

Escala real:
  - fitness = 1.0  → pixel-perfect
  - fitness = 0.5  → error promedio de 127 valores (imagen aleatoria)
  - fitness = 0.0  → cada pixel es el opuesto exacto (blanco vs negro puro)
"""

import numpy as np
from .renderer import render_ascii_grid


def compute_fitness(genome: np.ndarray, grid_n: int, target: np.ndarray, cell_size: int = 8) -> float:
    """
    Fitness = 1 - MAE/255

    MAE = mean absolute error por pixel entre la imagen renderizada y el target.
    Dividir por 255 normaliza al rango [0, 1].
    """
    rendered = np.array(render_ascii_grid(genome, grid_n, cell_size), dtype=np.float32)
    mae = float(np.mean(np.abs(rendered - target.astype(np.float32))))
    return 1.0 - mae / 255.0
