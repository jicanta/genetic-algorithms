"""
Genetic operators: initialization, selection, crossover, mutation.
"""

import numpy as np


# ─── Initialization ───────────────────────────────────────────────────────────

def greedy_genome(
    target: np.ndarray,
    rows: int,
    cols: int,
    cell_h: int,
    cell_w: int,
    darkness: np.ndarray,
) -> np.ndarray:
    """
    Build a genome by mapping each cell's average brightness to the nearest
    character by measured ink coverage.

    Used for warm-start: provides a reasonable baseline that the GA can then
    improve through global search, rather than starting from scratch.
    """
    sorted_dark_first = np.argsort(darkness)[::-1]  # index 0 = darkest char
    n = len(darkness)
    genome = np.zeros((rows, cols), dtype=np.int32)

    for r in range(rows):
        for c in range(cols):
            y0, x0 = r * cell_h, c * cell_w
            brightness = float(target[y0:y0 + cell_h, x0:x0 + cell_w].mean())
            t = brightness / 255.0           # 0 = black pixel, 1 = white pixel
            rank = round(t * (n - 1))        # 0 = darkest char, n-1 = lightest
            genome[r, c] = sorted_dark_first[rank]

    return genome


# ─── Selection ────────────────────────────────────────────────────────────────

def tournament_select(
    population: list[np.ndarray],
    fitnesses: np.ndarray,
    k: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Deterministic tournament: sample k individuals, return the one with lowest MSE.

    Larger k → more selection pressure → faster convergence but less diversity.
    """
    contestants = rng.integers(0, len(population), size=k)
    winner = contestants[np.argmin(fitnesses[contestants])]
    return population[winner]


# ─── Crossover ────────────────────────────────────────────────────────────────

def crossover_uniform(
    p1: np.ndarray, p2: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """
    Uniform crossover: each cell is independently inherited from either parent.
    Good at mixing fine-grained features from both parents.
    """
    mask = rng.random(p1.shape) < 0.5
    return np.where(mask, p1, p2), np.where(mask, p2, p1)


def crossover_block(
    p1: np.ndarray, p2: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """
    Block crossover: swap a random rectangular sub-region between parents.
    Preserves spatial coherence better than uniform crossover.
    """
    rows, cols = p1.shape
    r1 = int(rng.integers(0, rows))
    r2 = int(rng.integers(r1 + 1, rows + 1))
    c1 = int(rng.integers(0, cols))
    c2 = int(rng.integers(c1 + 1, cols + 1))
    ch1, ch2 = p1.copy(), p2.copy()
    ch1[r1:r2, c1:c2] = p2[r1:r2, c1:c2]
    ch2[r1:r2, c1:c2] = p1[r1:r2, c1:c2]
    return ch1, ch2


# ─── Mutation ─────────────────────────────────────────────────────────────────

def mutate(
    genome: np.ndarray,
    mutation_rate: float,
    n_chars: int,
    darkness: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Per-cell mutation with darkness-aware neighbor bias.

    70% of mutations step ±1 or ±2 positions in darkness rank — local search
    that nudges a cell toward a slightly brighter or darker character.
    30% jump to a completely random character — exploration.

    This balance avoids getting stuck in local optima while still making
    incremental progress most of the time.
    """
    child = genome.copy()
    if n_chars < 2:
        return child

    mask = rng.random(child.shape) < mutation_rate
    n_mut = int(mask.sum())
    if n_mut == 0:
        return child

    sorted_dark = np.argsort(darkness)   # sorted_dark[rank] = char index
    rank_of = np.argsort(sorted_dark)    # rank_of[char_idx] = darkness rank

    current = child[mask]
    use_neighbor = rng.random(n_mut) < 0.7
    new_vals = np.empty(n_mut, dtype=np.int32)

    if (~use_neighbor).any():
        random_current = current[~use_neighbor]
        random_draws = rng.integers(0, n_chars - 1, size=random_current.size)
        new_vals[~use_neighbor] = random_draws + (random_draws >= random_current)

    if use_neighbor.any():
        cur_ranks = rank_of[current[use_neighbor]]
        steps = rng.choice(np.array([-2, -1, 1, 2], dtype=np.int32), size=use_neighbor.sum())
        new_ranks = np.clip(cur_ranks + steps, 0, n_chars - 1)
        same_rank = new_ranks == cur_ranks
        if same_rank.any():
            at_darkest = cur_ranks[same_rank] == 0
            at_lightest = cur_ranks[same_rank] == (n_chars - 1)
            corrected = new_ranks[same_rank].copy()
            corrected[at_darkest] = 1
            corrected[at_lightest] = n_chars - 2

            middle = ~(at_darkest | at_lightest)
            if middle.any():
                corrected[middle] = cur_ranks[same_rank][middle] + rng.choice(
                    np.array([-1, 1], dtype=np.int32), size=middle.sum()
                )

            new_ranks[same_rank] = corrected
        new_vals[use_neighbor] = sorted_dark[new_ranks]

    child[mask] = new_vals
    return child
