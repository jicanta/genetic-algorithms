"""
Genetic operators: all selection, crossover, and mutation methods required by the TP.

Genomes are float32 arrays of shape (N_triangles, 10).
"Genes" are individual float values; "loci" are whole triangles (rows).
Crossover operates at the locus (triangle) level to preserve internal coherence.
Mutation operates at the gene (individual float) level.

Selection methods
-----------------
tournament_det       Deterministic: best of k random contestants always wins.
tournament_prob      Probabilistic: best of k wins with probability p.
roulette             Fitness-proportional (inverted MSE). High variance.
universal            Stochastic Universal Sampling — same probs as roulette, lower variance.
boltzmann            Temperature-annealed: starts near-uniform, ends fitness-proportional.
ranking              Rank-proportional: avoids domination by a single very-fit individual.

Crossover methods
-----------------
crossover_uniform    Per-locus coin flip. Fine-grained mixing.
crossover_one_point  Single cut point. Preserves large contiguous regions.
crossover_two_point  Two cut points. Exchanges a segment between parents.
crossover_annular    Circular two-point. Segment can wrap around the genome end.

Mutation methods
----------------
mutate_uniform       Per-gene Gaussian noise with fixed sigma. (baseline)
mutate_gen           Exactly one gene mutated per call.
mutate_multigen      Random number of genes mutated (up to max_genes).
mutate_non_uniform   Gaussian noise with sigma that decays with generation progress.
"""

import numpy as np


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _invert_fitnesses(fitnesses: np.ndarray) -> np.ndarray:
    """
    Convert minimization fitnesses to selection probabilities.

    We invert and normalize: prob_i ∝ 1 / (fitness_i + ε).
    Adding ε avoids division by zero when fitness reaches 0.
    """
    inv = 1.0 / (fitnesses + 1e-10)
    return inv / inv.sum()


# ─── Selection ────────────────────────────────────────────────────────────────

def tournament_det(
    population: list[np.ndarray],
    fitnesses: np.ndarray,
    k: int,
    rng: np.random.Generator,
    **_kwargs,
) -> np.ndarray:
    """
    Deterministic tournament: sample k individuals, return the one with lowest MSE.

    Higher k → more selection pressure → faster convergence, less diversity.
    """
    contestants = rng.integers(0, len(population), size=k)
    winner = contestants[int(np.argmin(fitnesses[contestants]))]
    return population[winner]


def tournament_prob(
    population: list[np.ndarray],
    fitnesses: np.ndarray,
    k: int,
    rng: np.random.Generator,
    prob: float = 0.75,
    **_kwargs,
) -> np.ndarray:
    """
    Probabilistic tournament: the best of k contestants wins with probability p;
    otherwise the second-best wins, and so on.

    Lower p → more randomness → preserves diversity. At p=1 equivalent to det.
    """
    contestants = rng.integers(0, len(population), size=k)
    ranked = contestants[np.argsort(fitnesses[contestants])]  # best first

    for i, idx in enumerate(ranked):
        # Win probability decreases geometrically for each successive rank
        if rng.random() < prob * (1 - prob) ** i or i == len(ranked) - 1:
            return population[idx]
    return population[ranked[0]]


def roulette(
    population: list[np.ndarray],
    fitnesses: np.ndarray,
    rng: np.random.Generator,
    **_kwargs,
) -> np.ndarray:
    """
    Fitness-proportional (roulette wheel) selection.

    Each individual's probability is proportional to 1/fitness.
    High variance: a single very-fit individual can dominate.
    """
    probs = _invert_fitnesses(fitnesses)
    idx = rng.choice(len(population), p=probs)
    return population[idx]


def universal(
    population: list[np.ndarray],
    fitnesses: np.ndarray,
    rng: np.random.Generator,
    **_kwargs,
) -> np.ndarray:
    """
    Stochastic Universal Sampling (SUS) — single-selection variant.

    Places one pointer at a random position on the cumulative probability
    distribution (equivalent to a single SUS pointer). Same expected distribution
    as roulette but pairs of calls to this function are less likely to select the
    same individual, reducing variance across a generation.
    """
    probs = _invert_fitnesses(fitnesses)
    cumulative = np.cumsum(probs)
    ptr = rng.random()
    idx = int(np.searchsorted(cumulative, ptr))
    idx = min(idx, len(population) - 1)
    return population[idx]


def boltzmann(
    population: list[np.ndarray],
    fitnesses: np.ndarray,
    rng: np.random.Generator,
    temperature: float = 10.0,
    **_kwargs,
) -> np.ndarray:
    """
    Boltzmann (temperature-annealed) selection.

    Selection probabilities: p_i ∝ exp(-fitness_i / T).
    - High T: near-uniform → broad exploration.
    - Low T: concentrates on low-fitness (best) individuals → exploitation.

    Temperature is computed externally by the GA and passed in at each call.
    """
    scaled = -fitnesses / (temperature + 1e-10)
    scaled -= scaled.max()           # numerical stability: shift before exp
    weights = np.exp(scaled)
    probs = weights / weights.sum()
    idx = rng.choice(len(population), p=probs)
    return population[idx]


def ranking(
    population: list[np.ndarray],
    fitnesses: np.ndarray,
    rng: np.random.Generator,
    **_kwargs,
) -> np.ndarray:
    """
    Linear rank-based selection.

    Individuals are ranked by fitness (rank 1 = best). Selection probability
    is proportional to rank, not raw fitness. This prevents a single very-fit
    individual from dominating while still favoring better individuals.

    prob_i ∝ (N - rank_i + 1)  →  best gets rank N, worst gets rank 1.
    """
    n = len(population)
    order = np.argsort(fitnesses)   # order[0] = index of best (lowest MSE) individual
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.arange(n, 0, -1, dtype=float)  # best → rank n, worst → rank 1
    probs = ranks / ranks.sum()
    idx = rng.choice(n, p=probs)
    return population[idx]


# ─── Crossover ────────────────────────────────────────────────────────────────

def crossover_uniform(
    p1: np.ndarray,
    p2: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Uniform crossover at the locus (triangle) level.

    Each triangle is independently inherited from p1 or p2 with equal probability.
    Produces fine-grained mixing; does not preserve contiguous triangle sequences.
    """
    n = p1.shape[0]
    mask = (rng.random(n) < 0.5)[:, np.newaxis]
    c1 = np.where(mask, p1, p2).astype(np.float32)
    c2 = np.where(mask, p2, p1).astype(np.float32)
    return c1, c2


def crossover_one_point(
    p1: np.ndarray,
    p2: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    One-point crossover at a random locus.

    c1 = p1[:cut] + p2[cut:]
    c2 = p2[:cut] + p1[cut:]

    Preserves large contiguous blocks from each parent.
    """
    cut = int(rng.integers(1, p1.shape[0]))
    c1 = np.concatenate([p1[:cut], p2[cut:]], axis=0)
    c2 = np.concatenate([p2[:cut], p1[cut:]], axis=0)
    return c1.astype(np.float32), c2.astype(np.float32)


def crossover_two_point(
    p1: np.ndarray,
    p2: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Two-point crossover: exchange a contiguous segment between two cut points.

    c1 = p1[:i] + p2[i:j] + p1[j:]
    c2 = p2[:i] + p1[i:j] + p2[j:]

    The segment length is random; good for transferring localized features.
    """
    n = p1.shape[0]
    i, j = sorted(rng.integers(0, n + 1, size=2).tolist())
    c1 = np.concatenate([p1[:i], p2[i:j], p1[j:]], axis=0)
    c2 = np.concatenate([p2[:i], p1[i:j], p2[j:]], axis=0)
    return c1.astype(np.float32), c2.astype(np.float32)


def crossover_annular(
    p1: np.ndarray,
    p2: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Annular (circular) crossover: like two-point but on a circular genome.

    A random start and length define a segment that can wrap around the end.
    This gives uniform probability to all contiguous segments regardless of
    position, which regular two-point crossover cannot achieve.
    """
    n = p1.shape[0]
    start = int(rng.integers(0, n))
    length = int(rng.integers(1, n))

    indices = np.arange(start, start + length) % n
    c1 = p1.copy()
    c2 = p2.copy()
    c1[indices] = p2[indices]
    c2[indices] = p1[indices]
    return c1.astype(np.float32), c2.astype(np.float32)


# ─── Mutation ─────────────────────────────────────────────────────────────────

def mutate_uniform(
    genome: np.ndarray,
    rng: np.random.Generator,
    mutation_rate: float = 0.02,
    mutation_sigma: float = 0.05,
    **_kwargs,
) -> np.ndarray:
    """
    Per-gene Gaussian mutation with fixed probability and sigma.

    Each of the N*10 genes mutates independently with probability mutation_rate.
    Noise is drawn from N(0, mutation_sigma) and the result is clamped to [0, 1].
    """
    child = genome.copy()
    mask = rng.random(child.shape) < mutation_rate
    noise = rng.normal(0.0, mutation_sigma, size=child.shape).astype(np.float32)
    child[mask] += noise[mask]
    np.clip(child, 0.0, 1.0, out=child)
    return child


def mutate_gen(
    genome: np.ndarray,
    rng: np.random.Generator,
    mutation_sigma: float = 0.05,
    **_kwargs,
) -> np.ndarray:
    """
    Single-gene mutation: exactly one gene changes per call.

    Minimal perturbation. Useful for fine-tuning near a local optimum but
    very slow for large-scale exploration in a big genome.
    """
    child = genome.copy()
    row = int(rng.integers(0, child.shape[0]))
    col = int(rng.integers(0, child.shape[1]))
    child[row, col] += float(rng.normal(0.0, mutation_sigma))
    child[row, col] = float(np.clip(child[row, col], 0.0, 1.0))
    return child


def mutate_multigen(
    genome: np.ndarray,
    rng: np.random.Generator,
    max_genes: int = 5,
    mutation_sigma: float = 0.05,
    **_kwargs,
) -> np.ndarray:
    """
    Multi-gene mutation: a random number of genes (1 to max_genes) change.

    More disruptive than single-gene; less uniform than per-gene probability.
    max_genes lets you control the maximum jump size.
    """
    child = genome.copy()
    n_genes = int(rng.integers(1, max_genes + 1))
    flat = child.reshape(-1)
    indices = rng.integers(0, flat.size, size=n_genes)
    flat[indices] += rng.normal(0.0, mutation_sigma, size=n_genes).astype(np.float32)
    np.clip(flat, 0.0, 1.0, out=flat)
    return child


def mutate_non_uniform(
    genome: np.ndarray,
    rng: np.random.Generator,
    mutation_rate: float = 0.02,
    mutation_sigma: float = 0.10,
    generation: int = 0,
    max_generations: int = 500,
    **_kwargs,
) -> np.ndarray:
    """
    Non-uniform mutation: sigma decays with generation progress.

    sigma_effective = mutation_sigma * (1 - generation/max_generations)^2

    Early generations: large sigma → wide exploration.
    Late generations:  small sigma → local refinement.

    Mimics a simulated annealing schedule embedded in the mutation operator.
    """
    progress = generation / max(max_generations - 1, 1)
    effective_sigma = mutation_sigma * (1.0 - progress) ** 2

    child = genome.copy()
    mask = rng.random(child.shape) < mutation_rate
    noise = rng.normal(0.0, effective_sigma, size=child.shape).astype(np.float32)
    child[mask] += noise[mask]
    np.clip(child, 0.0, 1.0, out=child)
    return child
