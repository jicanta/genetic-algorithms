"""
Genetic operators: selection, crossover, mutation.

These are the minimal operators needed for the GA to function.
Additional selection methods (roulette, Boltzmann, ranking, etc.) and
crossover/mutation variants required by the TP will be added in the next step.
"""

import numpy as np


def tournament_select(
    population: list[np.ndarray],
    fitnesses: np.ndarray,
    k: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Deterministic tournament selection: sample k individuals, return the best.

    Larger k → higher selection pressure → faster convergence, less diversity.
    """
    contestants = rng.integers(0, len(population), size=k)
    winner = contestants[int(np.argmin(fitnesses[contestants]))]
    return population[winner]


def crossover_uniform(
    p1: np.ndarray,
    p2: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Uniform crossover at the triangle level.

    For each triangle, independently choose to inherit it from p1 or p2.
    Operating on whole triangles (not individual genes) preserves the internal
    coherence of each triangle's position and color.

    Args:
        p1, p2:  float32 (N_triangles, 10).

    Returns:
        Two children of the same shape.
    """
    n = p1.shape[0]
    mask = rng.random(n) < 0.5           # (N,) boolean
    mask = mask[:, np.newaxis]            # (N, 1) for broadcasting over 10 genes

    c1 = np.where(mask, p1, p2)
    c2 = np.where(mask, p2, p1)
    return c1.astype(np.float32), c2.astype(np.float32)


def mutate(
    genome: np.ndarray,
    mutation_rate: float,
    mutation_sigma: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Per-gene Gaussian mutation.

    Each gene is mutated independently with probability `mutation_rate`.
    Mutation adds Gaussian noise N(0, mutation_sigma) and clamps to [0, 1].

    Args:
        genome:          float32 (N_triangles, 10).
        mutation_rate:   Probability that any single gene mutates.
        mutation_sigma:  Std of the Gaussian noise.

    Returns:
        Mutated copy of the genome.
    """
    child = genome.copy()
    mask = rng.random(child.shape) < mutation_rate
    noise = rng.normal(0.0, mutation_sigma, size=child.shape).astype(np.float32)
    child[mask] += noise[mask]
    np.clip(child, 0.0, 1.0, out=child)
    return child
