"""
Numba JIT-compiled fitness functions.

Zero-allocation MSE: a single loop over flattened arrays, compiled to native
machine code. Avoids the ~15 MB of intermediate arrays that the NumPy path
creates per evaluation (diff, diff², mean).
"""

import numpy as np
import numba


@numba.njit(cache=True)
def mse_numba(a, b):
    """
    Mean Squared Error between two arrays, computed in a single pass.

    Both arrays must have the same shape (any dimensions). No intermediate
    arrays are allocated — the entire computation uses O(1) extra memory.
    """
    a_flat = a.ravel()
    b_flat = b.ravel()
    n = a_flat.shape[0]
    total = 0.0
    for i in range(n):
        d = a_flat[i] - b_flat[i]
        total += d * d
    return total / n


@numba.njit(cache=True)
def mse_numba_masked(a, b, mask):
    """
    MSE over a subset of pixels selected by a boolean mask.

    a, b:  float32 (H*W, 3) — flattened pixel arrays.
    mask:  bool (H*W,) — True for pixels to include.
    """
    n = 0
    total = 0.0
    for i in range(mask.shape[0]):
        if mask[i]:
            for c in range(3):
                d = a[i, c] - b[i, c]
                total += d * d
            n += 3
    if n == 0:
        return 0.0
    return total / n
