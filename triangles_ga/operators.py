"""
Genetic operators: all selection, crossover, and mutation methods required by the TP.

Genomes are float32 arrays of shape (N, genes_per_shape) — triangles use 10 genes,
ovals use 8. "Genes" are individual float values; "loci" are whole shapes (rows).
Crossover operates at the locus (shape) level to preserve internal coherence.
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
mutate_layer_order   Swap or move triangles in draw order.
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


def _gene_sigma_scales(
    genome: np.ndarray,
    geometry_scale: float,
    color_scale: float,
    alpha_scale: float,
    genes_per_shape: int = 10,
) -> np.ndarray:
    """Return per-gene sigma multipliers for geometry, RGB, and alpha.

    Layout (last 4 genes are always color+alpha regardless of shape type):
        genes 0 .. n-5  → geometry (coords / radii)
        genes n-4 .. n-2 → RGB color
        gene  n-1        → alpha
    """
    n = genes_per_shape
    scales = np.empty_like(genome, dtype=np.float32)
    scales[:, :n - 4] = geometry_scale
    scales[:, n - 4:n - 1] = color_scale
    scales[:, n - 1] = alpha_scale
    return scales


def _scale_for_gene_index(
    gene_index: np.ndarray,
    geometry_scale: float,
    color_scale: float,
    alpha_scale: float,
    genes_per_shape: int = 10,
) -> np.ndarray:
    """Map flattened gene indices to the corresponding sigma multiplier."""
    n = genes_per_shape
    col = gene_index % n
    return np.where(col < n - 4, geometry_scale, np.where(col < n - 1, color_scale, alpha_scale))


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


def universal_batch(
    population: list[np.ndarray],
    fitnesses: np.ndarray,
    n: int,
    rng: np.random.Generator,
    **_kwargs,
) -> list[np.ndarray]:
    """
    Stochastic Universal Sampling for a batch of parent slots.

    Unlike roulette, SUS uses one random start plus evenly spaced pointers, so
    the selected parent set has lower sampling variance across a generation.
    """
    probs = _invert_fitnesses(fitnesses)
    cumulative = np.cumsum(probs)
    start = rng.random() / n
    pointers = start + np.arange(n, dtype=np.float64) / n
    indices = np.searchsorted(cumulative, pointers, side="left")
    indices = np.clip(indices, 0, len(population) - 1)
    return [population[int(idx)] for idx in indices]


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
    geometry_sigma_scale: float = 1.0,
    color_sigma_scale: float = 1.0,
    alpha_sigma_scale: float = 1.0,
    genes_per_shape: int = 10,
    **_kwargs,
) -> np.ndarray:
    """
    Per-gene Gaussian mutation with fixed probability and typed sigma.

    Each gene mutates independently with probability mutation_rate.
    Geometry, RGB, and alpha use separate sigma multipliers.
    """
    child = genome.copy()
    mask = rng.random(child.shape) < mutation_rate
    scales = _gene_sigma_scales(child, geometry_sigma_scale, color_sigma_scale, alpha_sigma_scale, genes_per_shape)
    noise = rng.normal(0.0, mutation_sigma, size=child.shape).astype(np.float32) * scales
    child[mask] += noise[mask]
    np.clip(child, 0.0, 1.0, out=child)
    return child


def mutate_gen(
    genome: np.ndarray,
    rng: np.random.Generator,
    mutation_sigma: float = 0.05,
    geometry_sigma_scale: float = 1.0,
    color_sigma_scale: float = 1.0,
    alpha_sigma_scale: float = 1.0,
    genes_per_shape: int = 10,
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
    scale = float(_scale_for_gene_index(
        np.array([col]), geometry_sigma_scale, color_sigma_scale, alpha_sigma_scale, genes_per_shape
    )[0])
    child[row, col] += float(rng.normal(0.0, mutation_sigma * scale))
    child[row, col] = float(np.clip(child[row, col], 0.0, 1.0))
    return child


def mutate_multigen(
    genome: np.ndarray,
    rng: np.random.Generator,
    max_genes: int = 5,
    mutation_sigma: float = 0.05,
    geometry_sigma_scale: float = 1.0,
    color_sigma_scale: float = 1.0,
    alpha_sigma_scale: float = 1.0,
    genes_per_shape: int = 10,
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
    scales = _scale_for_gene_index(indices, geometry_sigma_scale, color_sigma_scale, alpha_sigma_scale, genes_per_shape)
    noise = rng.normal(0.0, mutation_sigma, size=n_genes).astype(np.float32) * scales
    flat[indices] += noise
    np.clip(flat, 0.0, 1.0, out=flat)
    return child


def mutate_non_uniform(
    genome: np.ndarray,
    rng: np.random.Generator,
    mutation_rate: float = 0.02,
    mutation_sigma: float = 0.10,
    generation: int = 0,
    max_generations: int = 500,
    geometry_sigma_scale: float = 1.0,
    color_sigma_scale: float = 1.0,
    alpha_sigma_scale: float = 1.0,
    genes_per_shape: int = 10,
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
    scales = _gene_sigma_scales(child, geometry_sigma_scale, color_sigma_scale, alpha_sigma_scale, genes_per_shape)
    noise = rng.normal(0.0, effective_sigma, size=child.shape).astype(np.float32) * scales
    child[mask] += noise[mask]
    np.clip(child, 0.0, 1.0, out=child)
    return child


def mutate_layer_order(
    genome: np.ndarray,
    rng: np.random.Generator,
    layer_mutation_rate: float = 0.02,
    layer_mutation_max_shift: int = 8,
) -> np.ndarray:
    """
    Mutate draw order by swapping two triangles or moving one nearby.

    Alpha compositing makes order part of the phenotype. This operator lets the
    GA refine occlusion without having to rediscover the triangle geometry.
    """
    if layer_mutation_rate <= 0.0 or genome.shape[0] < 2 or rng.random() >= layer_mutation_rate:
        return genome

    child = genome.copy()
    n = child.shape[0]
    if rng.random() < 0.5:
        i, j = rng.choice(n, size=2, replace=False)
        child[[i, j]] = child[[j, i]]
        return child

    i = int(rng.integers(0, n))
    max_shift = min(layer_mutation_max_shift, n - 1)
    shift = int(rng.integers(-max_shift, max_shift + 1))
    if shift == 0:
        shift = 1 if i < n - 1 else -1
    j = int(np.clip(i + shift, 0, n - 1))

    row = child[i].copy()
    child = np.delete(child, i, axis=0)
    child = np.insert(child, j, row, axis=0)
    return child.astype(np.float32, copy=False)
