"""
Simple Genetic Algorithm for Exercise 1: ASCII Art

We build this incrementally:
  Phase 1 (here): Random init + mutation only (no crossover)
    → This is basically a "random local search" — shows the baseline
  Phase 2: Add crossover
  Phase 3: Add proper selection methods

For now:
- Selection: elite (keep best) + tournament
- Crossover: one-point
- Mutation: single gene (Gen) + multigene (MultiGen)
- Survival: exclusive (new gen replaces old)
"""

import numpy as np
from .individual import random_population
from .fitness import compute_fitness
from .renderer import N_CHARS


# ─────────────────────────────────────────────
# MUTATION
# ─────────────────────────────────────────────

def mutate_gen(individual: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Gen mutation: change exactly 1 random gene to a random value.

    This is the smallest possible change. Useful for fine-tuning late in evolution.
    """
    child = individual.copy()
    idx = rng.integers(0, len(child))
    child[idx] = rng.integers(0, N_CHARS)
    return child


def mutate_multigene(individual: np.ndarray, mutation_rate: float, rng: np.random.Generator) -> np.ndarray:
    """
    MultiGen mutation: each gene mutates independently with probability mutation_rate.

    mutation_rate=0.01 → on average 1% of genes change per mutation.
    Higher rates = more exploration (but risks destroying good solutions).
    """
    child = individual.copy()
    mask = rng.random(len(child)) < mutation_rate
    child[mask] = rng.integers(0, N_CHARS, size=mask.sum())
    return child


# ─────────────────────────────────────────────
# CROSSOVER
# ─────────────────────────────────────────────

def crossover_one_point(parent1: np.ndarray, parent2: np.ndarray, rng: np.random.Generator):
    """
    One-point crossover: split both parents at a random point, swap tails.

    parent1: [A A A A | B B B]
    parent2: [C C C C | D D D]
              ─────────────────
    child1:  [A A A A | D D D]   ← head from p1, tail from p2
    child2:  [C C C C | B B B]   ← head from p2, tail from p1

    Makes sense when genes near each other are correlated (e.g., adjacent cells
    in the grid tend to have similar brightness).
    """
    n = len(parent1)
    point = rng.integers(1, n)
    child1 = np.concatenate([parent1[:point], parent2[point:]])
    child2 = np.concatenate([parent2[:point], parent1[point:]])
    return child1, child2


def crossover_two_point(parent1: np.ndarray, parent2: np.ndarray, rng: np.random.Generator):
    """
    Two-point crossover: swap the middle segment between two cut points.

    child1: [p1 | p2_middle | p1]
    child2: [p2 | p1_middle | p2]

    More mixing than one-point, less disruptive than uniform.
    """
    n = len(parent1)
    pts = sorted(rng.integers(1, n, size=2))
    p, q = pts[0], pts[1]
    child1 = np.concatenate([parent1[:p], parent2[p:q], parent1[q:]])
    child2 = np.concatenate([parent2[:p], parent1[p:q], parent2[q:]])
    return child1, child2


# ─────────────────────────────────────────────
# SELECTION
# ─────────────────────────────────────────────

def select_elite(population: list, fitnesses: np.ndarray, k: int) -> list:
    """
    Elite selection: always take the top-k individuals by fitness.

    Pros: guarantees the best are kept.
    Cons: reduces diversity (everyone converges to copies of the best).
    """
    indices = np.argsort(fitnesses)[::-1][:k]
    return [population[i] for i in indices]


def select_roulette(population: list, fitnesses: np.ndarray, k: int, rng: np.random.Generator) -> list:
    """
    Roulette wheel (fitness-proportionate) selection.

    Each individual's probability of being selected = fitness / sum(fitnesses).
    Individuals with higher fitness spin the wheel more often.

    Problem: if one individual has fitness 100x others, it dominates everything
    → premature convergence. Ranking or tournament help with this.
    """
    total = fitnesses.sum()
    if total == 0:
        probs = np.ones(len(fitnesses)) / len(fitnesses)
    else:
        probs = fitnesses / total
    indices = rng.choice(len(population), size=k, p=probs, replace=True)
    return [population[i] for i in indices]


def select_tournament(population: list, fitnesses: np.ndarray, k: int, m: int, rng: np.random.Generator) -> list:
    """
    Deterministic tournament selection.

    For each of the k selections:
    1. Pick m individuals at random (the "tournament")
    2. The one with the highest fitness wins

    m controls selection pressure:
    - m=2: low pressure, more diversity
    - m=10: high pressure, converges faster (but risks premature convergence)
    """
    selected = []
    for _ in range(k):
        contestants = rng.integers(0, len(population), size=m)
        winner = contestants[np.argmax(fitnesses[contestants])]
        selected.append(population[winner])
    return selected


# ─────────────────────────────────────────────
# GA LOOP
# ─────────────────────────────────────────────

def run_ga(
    target: np.ndarray,
    grid_n: int,
    cell_size: int = 8,
    pop_size: int = 50,
    n_generations: int = 200,
    mutation_rate: float = 0.02,
    crossover_rate: float = 0.8,
    tournament_m: int = 3,
    seed: int = 42,
    warm_start: bool = True,  # seed population with perturbed greedy individuals
    callback=None,  # called each generation with (gen, best_genome, best_fitness, avg_fitness)
) -> tuple[np.ndarray, list[float]]:
    """
    Main GA loop for ASCII art.

    warm_start=True: half the initial population is greedy+noise, half is random.
    This gives the GA a much better starting point — without it, random genomes
    are so bad it takes hundreds of gens just to reach a mediocre solution.

    Returns:
        best_genome: the best individual found
        history: list of (best_fitness, avg_fitness) per generation
    """
    from .individual import greedy_individual
    rng = np.random.default_rng(seed)

    # ── Initialize ──
    if warm_start:
        greedy = greedy_individual(target, grid_n, cell_size)
        # Half population: greedy + noise at various intensities
        n_greedy = pop_size // 2
        population = []
        for i in range(n_greedy):
            noise_rate = 0.05 + 0.3 * (i / n_greedy)  # increasing noise
            ind = mutate_multigene(greedy.copy(), noise_rate, rng)
            population.append(ind)
        population += random_population(pop_size - n_greedy, grid_n, rng)
    else:
        population = random_population(pop_size, grid_n, rng)
    fitnesses = np.array([compute_fitness(ind, grid_n, target, cell_size) for ind in population])

    history = []
    best_genome = population[np.argmax(fitnesses)].copy()
    best_fitness = fitnesses.max()

    for gen in range(n_generations):
        # ── Select parents (tournament) ──
        parents = select_tournament(population, fitnesses, pop_size, tournament_m, rng)

        # ── Crossover + Mutation → offspring ──
        offspring = []
        for i in range(0, pop_size, 2):
            p1, p2 = parents[i], parents[min(i + 1, pop_size - 1)]

            if rng.random() < crossover_rate:
                c1, c2 = crossover_one_point(p1, p2, rng)
            else:
                c1, c2 = p1.copy(), p2.copy()

            c1 = mutate_multigene(c1, mutation_rate, rng)
            c2 = mutate_multigene(c2, mutation_rate, rng)
            offspring.extend([c1, c2])

        offspring = offspring[:pop_size]

        # ── Evaluate offspring ──
        offspring_fitnesses = np.array([
            compute_fitness(ind, grid_n, target, cell_size) for ind in offspring
        ])

        # ── Survival: Exclusive (offspring replaces parents) ──
        # but keep 1 elite from old gen (elitism)
        elite_idx = int(np.argmax(fitnesses))
        population = offspring
        fitnesses = offspring_fitnesses
        # Elitism: if best offspring is worse than old best, inject old best
        if fitnesses.max() < best_fitness:
            worst_idx = int(np.argmin(fitnesses))
            population[worst_idx] = best_genome.copy()
            fitnesses[worst_idx] = best_fitness
        else:
            best_genome = population[int(np.argmax(fitnesses))].copy()
            best_fitness = fitnesses.max()

        avg_fitness = float(fitnesses.mean())
        history.append((best_fitness, avg_fitness))

        if callback:
            callback(gen, best_genome, best_fitness, avg_fitness)

    return best_genome, history
