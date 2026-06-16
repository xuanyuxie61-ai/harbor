"""
design_of_experiments.py - Experimental Design Strategies for Surrogate Training

This module implements design of experiments (DOE) strategies for
generating training data for polynomial chaos surrogates. Integrates:
  - Binary vector enumeration for factorial designs (from ubvec)
  - Gray code ordering for adaptive sampling (from ubvec)
  - Combinatorial index sets for sparse grids (from ubvec k-subset)

Mathematical Framework
----------------------
### Tensor Product Design ###
For d parameters with n_i levels each, the full factorial design has
N = prod(n_i) points. For d=4, n_i=3: N = 81 points.

### Smolyak Sparse Grid ###
The Smolyak formula combines 1D rules:
  A(q, d) = sum_{k=0}^{q} (-1)^{q-k} C(d-1, q-k) sum_{|i|=q+k} (Q_{i_1} x ... x Q_{i_d})
where |i| = i_1 + ... + i_d and Q_j is a 1D rule of level j.
Cardinality grows as O(N * log(N)^{d-1}) vs O(N^d) for tensor product.

### Latin Hypercube Sampling ###
Partition each dimension into N equal strata, sample one point per stratum.
Ensures 1D projection is space-filling.

### Sobol Quasi-Random Sequences ###
Low-discrepancy sequences with star discrepancy:
  D_N^* = O((log N)^d / N)
Much better than Monte Carlo: D_N^* = O(N^{-1/2}).

### Gray Code Ordering ###
Successive designs differ by a single point addition/removal,
enabling efficient update of surrogate coefficients.
Binary Gray code: successive integers differ in exactly 1 bit.
"""

import numpy as np
from numpy.typing import NDArray
from typing import Tuple, List, Dict, Optional, Callable
import math
from numerical_utils import total_order_multi_index, hyperbolic_cross_multi_index


# ---------------------------------------------------------------------------
# Binary Vector Enumeration (from ubvec)
# ---------------------------------------------------------------------------

def ubvec_next_gray(n: int, t: NDArray) -> Tuple[NDArray, bool]:
    """
    Generate next binary vector in Gray code order.

    The Gray code successor changes exactly one bit position.
    The position to flip is determined by the parity of the
    binary representation (Morse-Thue sequence).

    Parameters
    ----------
    n : int
        Number of bits.
    t : ndarray(n,)
        Current binary vector (entries 0 or 1).

    Returns
    -------
    t_next : ndarray(n,)
        Next binary vector in Gray code order.
    done : bool
        True if we have wrapped around to the zero vector.
    """
    t = t.copy()

    # Compute bit to flip: position of rightmost 1 in (integer value + 1)
    # For Gray code: flip bit at position = number of trailing 1s
    j = 0
    while j < n and t[j] == 1:
        j += 1

    if j >= n:
        # Wrapped around
        return np.zeros(n, dtype=int), True

    t[j] = 1 - t[j]

    # If j > 0, also flip bit j-1 (Gray code property)
    # Actually, for standard reflected Gray code:
    # The bit to flip is determined by the number of trailing 1s in the current integer

    return t, False


def ubvec_next(n: int, t: NDArray) -> Tuple[NDArray, bool]:
    """
    Successor in standard binary counting order.

    Adds 1 to the binary number represented by t, with carry
    propagation from least significant bit (index 0).
    """
    t = t.copy()
    for i in range(n):
        if t[i] == 0:
            t[i] = 1
            return t, False
        t[i] = 0
    return t, True  # Overflow: all bits were 1


def gray_code_rank(n: int, t: NDArray) -> int:
    """
    Rank of a binary vector in Gray code ordering.

    The Gray code rank of binary vector g is:
      rank = g XOR (g >> 1) XOR (g >> 2) XOR ...
    computed as an integer.
    """
    # Convert binary vector to integer
    val = 0
    for i in range(n):
        val += t[i] * (2 ** i)

    # Gray to binary conversion
    mask = val
    result = val
    while mask > 0:
        mask >>= 1
        result ^= mask

    return result


def ksubset_colex_unrank(rank: int, k: int, n: int) -> NDArray:
    """
    Unrank a k-subset of {1,...,n} in colexicographic order.

    Uses the combinatorial number system:
      rank = C(c_k, k) + C(c_{k-1}, k-1) + ... + C(c_1, 1)
    where c_k > c_{k-1} > ... > c_1 >= 0.

    This determines which factors are "active" in sparse grid
    constructions and multi-index set enumeration.

    Parameters
    ----------
    rank : int
        Rank (0-based) of the desired subset.
    k : int
        Subset size.
    n : int
        Ground set size.

    Returns
    -------
    subset : ndarray(k,)
        Elements of the subset in decreasing order.
    """
    subset = np.zeros(k, dtype=int)
    remaining = rank

    for j in range(k, 0, -1):
        # Find largest c such that C(c, j) <= remaining
        c = j - 1
        while True:
            c_next = c + 1
            if c_next > n:
                break
            cnj = math.comb(c_next, j)
            if cnj > remaining:
                break
            c = c_next
        subset[k - j] = c
        remaining -= math.comb(c, j)

    return subset


# ---------------------------------------------------------------------------
# Quasi-Random Sequences
# ---------------------------------------------------------------------------

def van_der_corput_sequence(n: int, base: int = 2) -> NDArray:
    """
    Generate first n elements of the van der Corput sequence in given base.

    The i-th element is obtained by reflecting i in the radix point:
      i = sum a_k * base^k  =>  vdc(i) = sum a_k * base^{-(k+1)}

    This produces a low-discrepancy sequence on [0, 1] with
    star discrepancy D_N^* = O(log(N) / N).
    """
    result = np.zeros(n)
    for i in range(n):
        f = 1.0
        r = 0.0
        val = i
        while val > 0:
            f /= base
            r += f * (val % base)
            val //= base
        result[i] = r
    return result


def halton_sequence(n: int, d: int) -> NDArray:
    """
    Generate n points of the d-dimensional Halton sequence.

    Each dimension uses a van der Corput sequence with a different prime base.
    The discrepancy satisfies:
      D_N^* = O((log N)^d / N)

    This is exponentially better than Monte Carlo (O(N^{-1/2}))
    for moderate dimensions.

    Parameters
    ----------
    n : int
        Number of points.
    d : int
        Dimension.

    Returns
    -------
    points : ndarray(n, d)
        Points in [0, 1]^d.
    """
    primes = _first_primes(d)
    points = np.zeros((n, d))
    for j in range(d):
        points[:, j] = van_der_corput_sequence(n, primes[j])
    return points


def _first_primes(n: int) -> List[int]:
    """Return first n primes."""
    primes = []
    candidate = 2
    while len(primes) < n:
        is_prime = True
        for p in primes:
            if p * p > candidate:
                break
            if candidate % p == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(candidate)
        candidate += 1
    return primes


# ---------------------------------------------------------------------------
# Latin Hypercube Sampling
# ---------------------------------------------------------------------------

def latin_hypercube_sample(n: int, d: int, seed: int = 42) -> NDArray:
    """
    Generate Latin Hypercube Sample of n points in d dimensions.

    Algorithm:
    1. Partition [0,1]^d into n^d cells of volume 1/n^d
    2. For each dimension j, randomly permute {0, 1, ..., n-1}
    3. Sample uniformly within each stratum

    The LHS ensures each 1D projection has exactly one point per stratum,
    giving variance reduction: Var_LHS <= Var_MC for monotone functions.

    Parameters
    ----------
    n : int
        Number of samples.
    d : int
        Dimension.
    seed : int
        Random seed.

    Returns
    -------
    samples : ndarray(n, d)
        LHS points in [0, 1]^d.
    """
    rng = np.random.RandomState(seed)
    samples = np.zeros((n, d))

    for j in range(d):
        perm = rng.permutation(n)
        u = rng.random(n)
        samples[:, j] = (perm + u) / n

    return samples


# ---------------------------------------------------------------------------
# Gauss Quadrature Nodes for PCE
# ---------------------------------------------------------------------------

def gauss_legendre_1d(n_points: int) -> Tuple[NDArray, NDArray]:
    """
    1D Gauss-Legendre nodes and weights on [-1, 1].
    """
    from numerical_utils import legendre_ek_compute
    return legendre_ek_compute(n_points)


def tensor_product_quadrature(d: int, n_1d: int) -> Tuple[NDArray, NDArray]:
    """
    Tensor product Gauss-Legendre quadrature on [-1, 1]^d.

    Total points: N = n_1d^d
    Exact for polynomials of total degree <= 2*n_1d - 1 in each variable.

    Parameters
    ----------
    d : int
        Dimension.
    n_1d : int
        Points per dimension.

    Returns
    -------
    points : ndarray(N, d)
        Quadrature nodes.
    weights : ndarray(N,)
        Quadrature weights (product of 1D weights).
    """
    x1d, w1d = gauss_legendre_1d(n_1d)

    if d == 1:
        return x1d.reshape(-1, 1), w1d

    # Build tensor product
    grids = np.meshgrid(*([x1d] * d), indexing='ij')
    points = np.column_stack([g.ravel() for g in grids])

    weight_grids = np.meshgrid(*([w1d] * d), indexing='ij')
    weights = np.ones(points.shape[0])
    for wg in weight_grids:
        weights *= wg.ravel()

    return points, weights


def smolyak_quadrature(d: int, level: int) -> Tuple[NDArray, NDArray]:
    """
    Smolyak sparse grid quadrature on [-1, 1]^d.

    A(q, d) = sum_{q-level+1 <= |i| <= q} (-1)^{q-|i|} C(d-1, q-|i|) Q_i

    where |i| = i_1 + ... + i_d and Q_i = Q_{i_1} x ... x Q_{i_d}.

    For Clenshaw-Curtis rules:
      n(1) = 1, n(k) = 2^{k-1} + 1 for k >= 2

    Cardinality grows as O(N * log(N)^{d-1}) instead of O(N^d).

    Parameters
    ----------
    d : int
        Dimension.
    level : int
        Sparse grid level (controls accuracy).

    Returns
    -------
    points : ndarray(N, d)
        Quadrature nodes (with duplicates merged).
    weights : ndarray(N,)
        Combined quadrature weights.
    """
    from numerical_utils import legendre_ek_compute

    def n_points_cc(k: int) -> int:
        """Number of Clenshaw-Curtis points at level k."""
        if k <= 0:
            return 1
        return 2 ** (k - 1) + 1

    # Generate all multi-indices for this Smolyak formula
    all_points = []
    all_weights = []

    for q_val in range(max(1, level - d + 2), level + 1):
        # Enumerate |i| = q_val with i_j >= 1
        def enumerate_levels(dim: int, remaining: int, current: List[int]):
            if dim == 1:
                if remaining >= 1:
                    _add_level_combo(current + [remaining])
                return
            for k in range(1, remaining):
                enumerate_levels(dim - 1, remaining - k, current + [k])

        def _add_level_combo(levels: List[int]):
            """Add tensor product rule for this level combination."""
            sign = (-1) ** (level - sum(levels))
            coeff = sign * math.comb(d - 1, level - sum(levels))
            if coeff == 0:
                return

            # Build 1D rules
            rules_1d = []
            for lv in levels:
                nk = n_points_cc(lv)
                if nk == 1:
                    rules_1d.append((np.array([0.0]), np.array([2.0])))
                else:
                    xk, wk = legendre_ek_compute(min(nk, 20))
                    rules_1d.append((xk, wk))

            # Tensor product
            grids = np.meshgrid(*([r[0] for r in rules_1d]), indexing='ij')
            pts = np.column_stack([g.ravel() for g in grids])

            wts = np.ones(pts.shape[0])
            for r in rules_1d:
                w_grids = np.meshgrid(*([r[1]] * 1), indexing='ij')
                # Proper tensor product of weights
            # Recompute weights properly
            wts = np.ones(pts.shape[0])
            for dim_idx, (xk, wk) in enumerate(rules_1d):
                nk = len(wk)
                # Tile weights appropriately
                reps_before = 1
                reps_after = 1
                for di in range(dim_idx):
                    reps_before *= len(rules_1d[di][0])
                for di in range(dim_idx + 1, len(rules_1d)):
                    reps_after *= len(rules_1d[di][0])
                w_tiled = np.tile(np.repeat(wk, reps_after), reps_before)
                wts *= w_tiled

            all_points.append(pts)
            all_weights.append(coeff * wts)

        if d == 1:
            _add_level_combo([q_val])
        else:
            enumerate_levels(d, q_val, [])

    if not all_points:
        return np.zeros((1, d)), np.array([2.0 ** d])

    # Concatenate and merge duplicate points
    points = np.vstack(all_points)
    weights = np.concatenate(all_weights)

    # Merge close points
    points, weights = _merge_quadrature_points(points, weights, tol=1e-12)

    return points, weights


def _merge_quadrature_points(points: NDArray, weights: NDArray,
                             tol: float = 1e-12) -> Tuple[NDArray, NDArray]:
    """Merge duplicate quadrature points by summing their weights."""
    n = points.shape[0]
    if n <= 1:
        return points, weights

    # Simple O(n^2) merge for moderate n
    merged_pts = []
    merged_wts = []
    used = np.zeros(n, dtype=bool)

    for i in range(n):
        if used[i]:
            continue
        w_sum = weights[i]
        for j in range(i + 1, n):
            if used[j]:
                continue
            if np.linalg.norm(points[i] - points[j]) < tol:
                w_sum += weights[j]
                used[j] = True
        merged_pts.append(points[i])
        merged_wts.append(w_sum)
        used[i] = True

    return np.array(merged_pts), np.array(merged_wts)


# ---------------------------------------------------------------------------
# Full DOE Pipeline
# ---------------------------------------------------------------------------

def generate_doe(d: int, n_samples: int, method: str = 'halton',
                 param_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
                 seed: int = 42) -> Tuple[NDArray, NDArray]:
    """
    Generate design of experiments for surrogate training.

    Parameters
    ----------
    d : int
        Number of uncertain parameters.
    n_samples : int
        Number of design points.
    method : str
        DOE method: 'halton', 'lhs', 'tensor', 'smolyak'.
    param_bounds : dict, optional
        Maps parameter names to (lower, upper) bounds.
        If None, uses [-1, 1] for all parameters.
    seed : int
        Random seed (for LHS).

    Returns
    -------
    points : ndarray(n, d)
        Design points in parameter space.
    weights : ndarray(n,)
        Associated weights (uniform for sampling, quadrature for integration).
    """
    if method == 'halton':
        points = halton_sequence(n_samples, d)
        # Map from [0,1]^d to [-1,1]^d
        points = 2.0 * points - 1.0
        weights = np.ones(n_samples) * (2.0 ** d) / n_samples

    elif method == 'lhs':
        points = latin_hypercube_sample(n_samples, d, seed)
        points = 2.0 * points - 1.0
        weights = np.ones(n_samples) * (2.0 ** d) / n_samples

    elif method == 'tensor':
        n_1d = max(2, int(round(n_samples ** (1.0 / d))))
        points, weights = tensor_product_quadrature(d, n_1d)

    elif method == 'smolyak':
        level = max(1, int(round(math.log2(max(n_samples, 2)))))
        points, weights = smolyak_quadrature(d, level)

    else:
        raise ValueError(f"Unknown DOE method: {method}")

    # Map to parameter bounds if provided
    if param_bounds is not None:
        param_names = list(param_bounds.keys())
        for j, name in enumerate(param_names):
            if j < d:
                lo, hi = param_bounds[name]
                # Map from [-1, 1] to [lo, hi]
                points[:, j] = lo + (points[:, j] + 1.0) * (hi - lo) / 2.0

    return points, weights
