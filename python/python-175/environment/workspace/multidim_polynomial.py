"""
multidim_polynomial.py
======================
Multidimensional polynomial algebra with graded lexicographic ordering.

Fused from seed project:
- 893_polynomial : multi-dimensional polynomial multiplication & grlex ranking

Mathematical foundation
-----------------------
A d-variate monomial is denoted x^alpha = x_1^{alpha_1} ... x_d^{alpha_d}
with multi-index alpha \in \mathbb{N}_0^d.

Graded lexicographic order (grlex):
1. Primary key: total degree |alpha| = sum alpha_i  (ascending)
2. Secondary key: lexicographic order (descending, i.e. compare alpha_1 first)

The rank of a multi-index under grlex can be computed combinatorially:
    rank(alpha) = C(|alpha|+d, d) - sum_{k=1}^{d} C(|alpha|_{k+1..d} + d - k, d - k + 1)
where |alpha|_{k+1..d} = sum_{i=k+1}^d alpha_i, and C(n,k) denotes binomial coefficients.

For gPC, the multi-dimensional orthogonal basis is the tensor product of 1-D bases:
    Psi_alpha(xi) = prod_{k=1}^d phi_{alpha_k}(xi_k)
where phi_m are normalized Legendre/Hermite/Chebyshev polynomials.
"""

import numpy as np
from math import comb
from itertools import combinations_with_replacement


def multi_index_total_degree(alpha):
    """Compute |alpha| = sum of components."""
    return int(np.sum(alpha))


def multi_index_grlex_compare(alpha, beta):
    """
    Compare two multi-indices under graded lexicographic order.
    Returns -1 if alpha < beta, 0 if equal, 1 if alpha > beta.
    """
    alpha = np.asarray(alpha, dtype=int)
    beta = np.asarray(beta, dtype=int)
    da = multi_index_total_degree(alpha)
    db = multi_index_total_degree(beta)
    if da < db:
        return -1
    if da > db:
        return 1
    # Same total degree: lexicographic comparison
    for a, b in zip(alpha, beta):
        if a > b:
            return -1
        if a < b:
            return 1
    return 0


def multi_index_rank_grlex(alpha, d):
    """
    Compute the 0-based rank of multi-index alpha in d dimensions under grlex order.
    Uses the combinatorial formula.

    Reference: John Burkardt, "MONO_RANK_GRLEX".
    """
    alpha = np.asarray(alpha, dtype=int)
    if len(alpha) != d:
        raise ValueError("alpha length must equal dimension d")
    n = multi_index_total_degree(alpha)
    # Rank of first monomial of total degree n
    rank = comb(n + d, d) - 1
    # Subtract contributions
    s = n
    for k in range(d - 1):
        s -= alpha[k]
        rank -= comb(s + d - k - 1, d - k)
    return rank


def multi_index_unrank_grlex(rank, d):
    """
    Inverse of rank: given rank and dimension d, return the multi-index alpha.
    """
    if rank < 0:
        raise ValueError("rank must be non-negative")
    # Find total degree n such that C(n+d,d)-1 <= rank < C(n+d+1,d)-1
    n = 0
    while comb(n + d, d) - 1 <= rank:
        n += 1
    n -= 1
    rank -= comb(n + d, d) - 1
    alpha = np.zeros(d, dtype=int)
    s = n
    for k in range(d - 1):
        # Find alpha[k] such that the decrement matches
        alpha_k = 0
        while alpha_k <= s:
            dec = comb(s - alpha_k + d - k - 1, d - k - 1)
            if dec <= rank:
                alpha_k += 1
            else:
                break
        alpha_k -= 1
        alpha[k] = alpha_k
        rank -= comb(s - alpha_k + d - k - 1, d - k - 1)
        s -= alpha_k
    alpha[-1] = s
    return alpha


def enumerate_multi_indices_grlex(d, max_degree):
    """
    Enumerate all d-dimensional multi-indices with total degree <= max_degree,
    sorted by graded lexicographic order.
    Returns a 2-D array of shape (N, d).
    """
    if d < 1:
        raise ValueError("dimension d must be positive")
    if max_degree < 0:
        return np.zeros((0, d), dtype=int)
    indices = []
    # Generate all combinations with replacement for each total degree
    for deg in range(max_degree + 1):
        # Use stars-and-bars: represent as d-1 dividers among deg stars
        for dividers in combinations_with_replacement(range(deg + d - 1), d - 1):
            alpha = np.zeros(d, dtype=int)
            prev = -1
            for i, pos in enumerate(dividers):
                alpha[i] = pos - prev - 1
                prev = pos
            alpha[-1] = deg + d - 1 - 1 - prev
            indices.append(alpha.copy())
    # Sort by grlex
    indices.sort(key=lambda a: (multi_index_total_degree(a), tuple(-a)))
    return np.array(indices, dtype=int)


def enumerate_multi_indices_total_degree(d, max_degree):
    """Alias for enumerate_multi_indices_grlex."""
    return enumerate_multi_indices_grlex(d, max_degree)


def sparse_grid_index_set(d, level, rule="tensor"):
    """
    Generate a sparse index set for d-dimensional polynomial chaos.

    Parameters
    ----------
    d : int
        Dimension
    level : int
        Sparse grid level (controls maximum total degree)
    rule : str
        "tensor" -> full tensor product (degree = level in each dimension)
        "total"  -> total degree truncation (sum of degrees <= level)
        "hyperbolic" -> hyperbolic cross (product of (deg_i+1) <= level+1)

    Returns
    -------
    indices : ndarray, shape (N, d)
    """
    if rule == "tensor":
        grids = [np.arange(level + 1) for _ in range(d)]
        mesh = np.array(np.meshgrid(*grids, indexing='ij'))
        return mesh.reshape(d, -1).T
    elif rule == "total":
        return enumerate_multi_indices_grlex(d, level)
    elif rule == "hyperbolic":
        all_idx = enumerate_multi_indices_grlex(d, level)
        mask = np.ones(all_idx.shape[0], dtype=bool)
        for i in range(all_idx.shape[0]):
            prod = 1.0
            for k in range(d):
                prod *= (all_idx[i, k] + 1)
            if prod > level + 1:
                mask[i] = False
        return all_idx[mask]
    else:
        raise ValueError(f"Unknown rule: {rule}")


def multivariate_orthogonal_basis(alpha, xi, poly_eval_1d):
    """
    Evaluate the multivariate orthogonal basis Psi_alpha(xi).

    Parameters
    ----------
    alpha : array-like of ints, shape (d,)
        Multi-index.
    xi : array-like, shape (d,) or (n_samples, d)
        Random variable samples.
    poly_eval_1d : callable
        Function poly_eval_1d(m, x) evaluating the 1-D normalized orthogonal polynomial
        of degree m at points x.

    Returns
    -------
    values : ndarray
        Psi_alpha(xi) evaluated at the given points.
    """
    alpha = np.asarray(alpha, dtype=int)
    xi = np.asarray(xi, dtype=float)
    if xi.ndim == 1:
        xi = xi.reshape(1, -1)
    d = len(alpha)
    if xi.shape[1] != d:
        raise ValueError("xi must have d columns")
    vals = np.ones(xi.shape[0])
    for k in range(d):
        vals *= poly_eval_1d(alpha[k], xi[:, k])
    return vals


def test_multidim_polynomial():
    """Self-tests for multi-index enumeration and ranking."""
    d = 3
    max_deg = 3
    idx = enumerate_multi_indices_grlex(d, max_deg)
    # Check monotonic rank
    for i, alpha in enumerate(idx):
        r = multi_index_rank_grlex(alpha, d)
        assert r == i, f"Rank mismatch at i={i}, alpha={alpha}"
    # Check unrank
    for i in range(len(idx)):
        alpha = multi_index_unrank_grlex(i, d)
        assert np.array_equal(alpha, idx[i])
    # Check total degree count
    n_choose = comb(max_deg + d, d)
    assert len(idx) == n_choose
    print("multidim_polynomial: all self-tests passed")


if __name__ == "__main__":
    test_multidim_polynomial()
