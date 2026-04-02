"""
Individual representation for Exercise 1: ASCII Art

An individual IS a candidate solution: one specific arrangement of ASCII
characters in the NxN grid.

Genome structure:
- Flat numpy array of integers, shape (grid_n * grid_n,)
- Each value is an index into CHARSET = " .:-=+*#%@"  (0 to 9)
- Example for a 4x4 grid: [0, 5, 3, 9, 0, 2, 8, 1, ...]

Why flat array?
- Easier to apply crossover (just split at a point in the 1D array)
- Easy to index and mutate
- When we need the 2D grid, we .reshape(grid_n, grid_n)
"""

import numpy as np
from .renderer import N_CHARS


def random_individual(grid_n: int, rng: np.random.Generator = None) -> np.ndarray:
    """
    Create a random individual: uniformly random ASCII char indices.

    Args:
        grid_n: size of ASCII grid (NxN)
        rng: numpy random generator (for reproducibility)

    Returns:
        numpy int array of shape (grid_n * grid_n,)
    """
    if rng is None:
        rng = np.random.default_rng()
    return rng.integers(0, N_CHARS, size=grid_n * grid_n)


def greedy_individual(target: np.ndarray, grid_n: int, cell_size: int = 8) -> np.ndarray:
    """
    Create a 'greedy' individual by picking the best char per cell independently.

    This is NOT how a GA works (it ignores interactions between cells),
    but it's useful as a warm start or baseline to compare against.

    For each cell in the grid, we look at the average brightness of the
    corresponding region in the target image and pick the char with the
    closest brightness.

    Args:
        target: preprocessed target numpy array (grayscale, float32)
        grid_n: size of ASCII grid
        cell_size: pixels per cell

    Returns:
        numpy int array of shape (grid_n * grid_n,)
    """
    from .renderer import N_CHARS
    genome = np.zeros(grid_n * grid_n, dtype=int)

    for row in range(grid_n):
        for col in range(grid_n):
            y0, y1 = row * cell_size, (row + 1) * cell_size
            x0, x1 = col * cell_size, (col + 1) * cell_size
            cell_brightness = float(np.mean(target[y0:y1, x0:x1]))

            # char brightness: index 0 → 255 (white/space), index 9 → 0 (black/@)
            best_idx = round((1 - cell_brightness / 255) * (N_CHARS - 1))
            best_idx = max(0, min(N_CHARS - 1, best_idx))
            genome[row * grid_n + col] = best_idx

    return genome


def random_population(pop_size: int, grid_n: int, rng: np.random.Generator = None) -> list[np.ndarray]:
    """Create a list of pop_size random individuals."""
    if rng is None:
        rng = np.random.default_rng()
    return [random_individual(grid_n, rng) for _ in range(pop_size)]
