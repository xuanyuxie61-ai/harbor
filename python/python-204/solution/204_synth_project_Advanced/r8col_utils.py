"""
r8col_utils.py
==============

Column-wise operations on real M x N matrices, ported from the
``r8col`` library (Burkardt) and re-purposed for uncertainty
quantification.  In Sobol sensitivity analysis every sampling plan is
an M x N matrix whose columns are *independent parameter vectors*;
robust column operations (duplicate removal, lexicographic sorting,
mean / dispersion statistics) are therefore the backbone of any
reliable Saltelli / Jansen estimator pipeline.

All functions work with NumPy arrays in C-order and assume
``a.shape == (m, n)``.

References
----------
* J. Burkardt, ``r8col`` library,
  https://people.sc.fsu.edu/~jburkardt/m_src/r8col/r8col.html
* A. Saltelli et al., *Global Sensitivity Analysis*, Wiley, 2008.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------
# Lexicographic comparison
# ---------------------------------------------------------------------
def r8col_compare(a: np.ndarray, i: int, j: int) -> int:
    """Lexicographically compare columns ``i`` and ``j`` of ``a``.

    Returns
    -------
    int
        -1 if a[:, i] < a[:, j],  0 if equal, +1 otherwise.
    """
    if a.ndim != 2:
        raise ValueError("r8col_compare: a must be 2-D")
    m, n = a.shape
    if not (0 <= i < n and 0 <= j < n):
        raise IndexError(f"r8col_compare: column index out of range ({i},{j}) vs n={n}")
    col_i = a[:, i]
    col_j = a[:, j]
    diff = col_i - col_j
    # Numerical guard against round-off when columns are identical
    tol = 1.0e-14 * (1.0 + np.max(np.abs(col_i)) + np.max(np.abs(col_j)))
    for k in range(m):
        if diff[k] < -tol:
            return -1
        if diff[k] > tol:
            return +1
    return 0


# ---------------------------------------------------------------------
# Duplicate-column detection
# ---------------------------------------------------------------------
def r8col_duplicates(m: int, n: int, n_unique: int,
                    rng: np.random.Generator | None = None) -> np.ndarray:
    """Generate a random M x N matrix with exactly ``n_unique`` distinct columns.

    Parameters
    ----------
    m : int
        Row count.
    n : int
        Total column count (``n >= n_unique``).
    n_unique : int
        Number of *distinct* columns.
    rng : numpy Generator, optional
        Reproducible source of randomness.
    """
    if n_unique < 1 or n < n_unique:
        raise ValueError("r8col_duplicates: require 1 <= n_unique <= n")
    if m < 1 or n < 1:
        raise ValueError("r8col_duplicates: m,n must be positive")
    rng = rng or np.random.default_rng()
    # Build the distinct pool first, then sample with replacement
    pool = rng.standard_normal((m, n_unique))
    idx = rng.integers(0, n_unique, size=n)
    return pool[:, idx].copy()


# ---------------------------------------------------------------------
# Column mean
# ---------------------------------------------------------------------
def r8col_mean(a: np.ndarray) -> np.ndarray:
    """Return the per-column mean vector (length M)."""
    if a.ndim != 2:
        raise ValueError("r8col_mean: a must be 2-D")
    return a.mean(axis=1)


# ---------------------------------------------------------------------
# Column-wise maximum
# ---------------------------------------------------------------------
def r8col_max(a: np.ndarray) -> np.ndarray:
    if a.ndim != 2:
        raise ValueError("r8col_max: a must be 2-D")
    return a.max(axis=1)


# ---------------------------------------------------------------------
# Column-wise minimum
# ---------------------------------------------------------------------
def r8col_min(a: np.ndarray) -> np.ndarray:
    if a.ndim != 2:
        raise ValueError("r8col_min: a must be 2-D")
    return a.min(axis=1)


# ---------------------------------------------------------------------
# Linf normalisation (per column)
# ---------------------------------------------------------------------
def r8col_normalize_li(a: np.ndarray) -> np.ndarray:
    """Return a copy of ``a`` scaled so that each column has L-infinity norm 1."""
    if a.ndim != 2:
        raise ValueError("r8col_normalize_li: a must be 2-D")
    out = a.copy()
    for j in range(a.shape[1]):
        nrm = np.max(np.abs(out[:, j]))
        if nrm > 0.0:
            out[:, j] /= nrm
    return out


# ---------------------------------------------------------------------
# Lexicographic sort (returns permutation)
# ---------------------------------------------------------------------
def r8col_sort_lex(a: np.ndarray) -> np.ndarray:
    """Return the permutation that sorts columns of ``a`` lexicographically."""
    n = a.shape[1]
    perm = np.arange(n)
    # Simple insertion sort — sufficient since n is O(sampling budget)
    for i in range(1, n):
        j = i
        while j > 0 and r8col_compare(a, perm[j - 1], perm[j]) > 0:
            perm[j - 1], perm[j] = perm[j], perm[j - 1]
            j -= 1
    return perm


# ---------------------------------------------------------------------
# Undex (unique + index): identify distinct columns up to tolerance
# ---------------------------------------------------------------------
def r8col_sorted_tol_unique_count(a: np.ndarray, tol: float = 1.0e-12) -> int:
    """Count unique columns of a *pre-sorted* matrix up to tolerance ``tol``."""
    if a.shape[1] == 0:
        return 0
    n_unique = 1
    for j in range(1, a.shape[1]):
        diff = np.max(np.abs(a[:, j] - a[:, j - 1]))
        if diff > tol:
            n_unique += 1
    return n_unique


# ---------------------------------------------------------------------
# Saltelli-style "A, B, AB_i" deduplication
# ---------------------------------------------------------------------
def dedupe_sample_matrix(mat: np.ndarray, tol: float = 1.0e-12) -> np.ndarray:
    """Remove near-duplicate columns while preserving order.

    Used inside the Sobol engine to guarantee that the A / B / AB_i / BA_i
    matrices contain exactly N distinct parameter vectors even when the
    underlying quasi-random sampler occasionally produces duplicates.
    """
    if mat.shape[1] == 0:
        return mat.copy()
    keep = [0]
    for j in range(1, mat.shape[1]):
        is_dup = False
        for k in keep:
            if np.max(np.abs(mat[:, j] - mat[:, k])) <= tol:
                is_dup = True
                break
        if not is_dup:
            keep.append(j)
    return mat[:, keep].copy()


# ---------------------------------------------------------------------
# Quick sanity test (executed on import only in __main__)
# ---------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    A = r8col_duplicates(4, 12, 5, rng=rng)
    print("shape      :", A.shape)
    print("mean       :", r8col_mean(A))
    print("unique cnt :", r8col_sorted_tol_unique_count(A[:, r8col_sort_lex(A)]))
    print("dedupe cnt :", dedupe_sample_matrix(A).shape[1])
