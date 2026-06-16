"""
quasi_mc.py -- Quasi-Monte Carlo Sampling Module
==================================================
Implements low-discrepancy sequences (van der Corput, Halton) and scrambled
variants for high-dimensional uncertainty quantification. Provides star
discrepancy computation and convergence verification demonstrating the
O(N^{-1} log(N)^s) advantage of QMC over MC's O(N^{-1/2}).

Seed references:
  - 235_cube_monte_carlo: cube01_sample, monomial integration
  - 264_cvtp: CVT Monte Carlo sampling (cvtp_region_sampler)
  - 804_nint_exactness_mixed: quadrature exactness testing framework

Scientific context:
  In UQ, the random input space Omega = [0,1]^s must be sampled to estimate
  statistical quantities E[g(xi)] = integral_{[0,1]^s} g(xi) dxi.
  QMC replaces random samples with deterministic low-discrepancy points,
  achieving superior convergence for smooth integrands (Koksma-Hlawka).
"""
import numpy as np
from typing import Tuple


# ---------------------------------------------------------------------------
# Prime table for Halton sequence bases
# ---------------------------------------------------------------------------
_PRIME_TABLE = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47,
                53, 59, 61, 67, 71, 73, 79, 83, 89, 97]


def van_der_corput(n: int, base: int = 2) -> np.ndarray:
    """
    Generate the first *n* elements of the van der Corput sequence in given base.

    The van der Corput sequence is the 1-D building block for Halton sequences.
    For index i, we write i in the given base, then reflect the digits about
    the decimal point:

        i = d_k d_{k-1} ... d_1 d_0  (base b)
        vdc(i) = 0.d_0 d_1 ... d_{k-1} d_k  (base b)

    Parameters
    ----------
    n : int
        Number of points (>= 1).
    base : int
        Prime base for radical-inverse computation (>= 2).

    Returns
    -------
    seq : ndarray, shape (n,)
        Points in (0, 1).
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    if base < 2:
        raise ValueError("base must be >= 2")
    seq = np.zeros(n, dtype=np.float64)
    for i in range(1, n + 1):
        f, r, val = 1.0, 0.0, i
        while val > 0:
            f /= base
            r += f * (val % base)
            val //= base
        seq[i - 1] = r
    return seq


def halton(n: int, dim: int) -> np.ndarray:
    """
    Generate *n* points of the Halton sequence in *dim* dimensions.

    The s-dimensional Halton sequence uses s distinct primes as bases:
        x_i = (vdc_{b_1}(i), vdc_{b_2}(i), ..., vdc_{b_s}(i))

    The star discrepancy of the Halton sequence satisfies:
        D*_N = O(N^{-1} (log N)^s)
    compared to MC's probabilistic O(N^{-1/2}).

    Parameters
    ----------
    n : int
        Number of sample points (>= 1).
    dim : int
        Dimension of the sample space (1 <= dim <= len(_PRIME_TABLE)).

    Returns
    -------
    points : ndarray, shape (n, dim)
        Quasi-random points in [0,1]^dim.
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    if dim < 1 or dim > len(_PRIME_TABLE):
        raise ValueError(f"dim must be in [1, {len(_PRIME_TABLE)}]")
    points = np.zeros((n, dim), dtype=np.float64)
    for d in range(dim):
        points[:, d] = van_der_corput(n, _PRIME_TABLE[d])
    return points


def scrambled_halton(n: int, dim: int, seed: int = 42) -> np.ndarray:
    """
    Owen-scrambled Halton sequence for randomized QMC error estimation.

    Digital scrambling applies a random permutation to each digit of the
    radical-inverse expansion, preserving the low-discrepancy property in
    expectation while enabling variance estimation via multiple scramblings.

    The scrambled estimator is:
        mu_scram = (1/R) sum_{r=1}^{R} (1/N) sum_{i=1}^{N} g(x_i^{(r)})
    where x_i^{(r)} is the r-th scrambled copy.

    Parameters
    ----------
    n : int
        Number of sample points.
    dim : int
        Dimension of sample space.
    seed : int
        Random seed for reproducibility of permutations.

    Returns
    -------
    points : ndarray, shape (n, dim)
        Scrambled quasi-random points in [0,1]^dim.
    """
    rng = np.random.RandomState(seed)
    base_points = halton(n, dim)
    points = base_points.copy()
    for d in range(dim):
        base = _PRIME_TABLE[d]
        # Random digit permutation for this dimension
        perm = rng.permutation(base)
        for i in range(n):
            # Extract digits of the van der Corput index
            val = i + 1
            digits = []
            temp = val
            while temp > 0:
                digits.append(temp % base)
                temp //= base
            # Apply permutation to each digit and reconstruct
            r = 0.0
            f = 1.0 / base
            for digit in digits:
                r += f * perm[digit]
                f /= base
            points[i, d] = min(r, 1.0 - 1e-15)
    return points


def star_discrepancy(points: np.ndarray) -> float:
    """
    Compute the star discrepancy D*_N of a point set in [0,1]^s.

    The star discrepancy measures the maximum deviation between the empirical
    distribution and the uniform distribution:
        D*_N = sup_{x in [0,1]^s} |F_N(x) - prod(x_j)|
    where F_N(x) = (1/N) |{i : x_i <= x componentwise}|.

    For s=1, an exact O(N log N) formula is used.
    For s>=2, a Monte Carlo estimate with 2000 test points is used.

    Parameters
    ----------
    points : ndarray, shape (n, s)
        Point set in [0,1]^s.

    Returns
    -------
    d_star : float
        Estimated star discrepancy in [0, 1].
    """
    points = np.atleast_2d(points)
    n, s = points.shape
    if n == 0:
        return 1.0
    if s == 1:
        sorted_pts = np.sort(points.ravel())
        d_plus = np.max(np.arange(1, n + 1) / n - sorted_pts)
        d_minus = np.max(sorted_pts - np.arange(0, n) / n)
        return float(max(d_plus, d_minus))

    # Monte Carlo estimate for multi-D discrepancy
    rng = np.random.RandomState(0)
    test_points = rng.rand(2000, s)
    d_star = 0.0
    for tp in test_points:
        # Fraction of points dominated by tp
        dominated = np.all(points <= tp, axis=1)
        empirical = np.sum(dominated) / n
        volume = np.prod(tp)
        d_star = max(d_star, abs(empirical - volume))
    return float(d_star)


def qmc_convergence_test(dim: int = 3, n_max: int = 2000) -> dict:
    """
    Verify that QMC converges faster than MC for a smooth test integrand.

    Test function: g(xi) = exp(-sum_j xi_j) * prod_j cos(2*pi*xi_j)
    This is a smooth, periodic function where QMC should show advantage.

    The Koksma-Hlawka inequality guarantees:
        |QMC - I| <= V(g) * D*_N
    where V(g) is the variation of g in the sense of Hardy-Krause.

    Returns dict with mc_errors, qmc_errors, mc_rates, qmc_rates.
    """
    sample_sizes = np.array([100, 200, 500, 1000, 2000, n_max])

    # Reference integral via high-order QMC
    ref_pts = halton(50000, dim)
    ref_val = np.mean(np.exp(-np.sum(ref_pts, axis=1)) *
                      np.prod(np.cos(2 * np.pi * ref_pts), axis=1))

    mc_errors = []
    qmc_errors = []
    rng = np.random.RandomState(123)

    for n in sample_sizes:
        def integrand(x):
            return np.exp(-np.sum(x, axis=1)) * np.prod(np.cos(2 * np.pi * x), axis=1)

        # MC estimate (average of 5 trials)
        mc_est = 0.0
        for trial in range(5):
            pts = rng.rand(n, dim)
            mc_est += np.mean(integrand(pts))
        mc_est /= 5
        mc_errors.append(abs(mc_est - ref_val))

        # QMC estimate
        pts_q = halton(n, dim)
        qmc_est = np.mean(integrand(pts_q))
        qmc_errors.append(abs(qmc_est - ref_val))

    mc_errors = np.array(mc_errors)
    qmc_errors = np.array(qmc_errors)

    # Convergence rates via log-log slope
    mc_rates = -np.diff(np.log(mc_errors + 1e-16)) / np.diff(np.log(sample_sizes.astype(float)))
    qmc_rates = -np.diff(np.log(qmc_errors + 1e-16)) / np.diff(np.log(sample_sizes.astype(float)))

    return {
        'sample_sizes': sample_sizes,
        'mc_errors': mc_errors,
        'qmc_errors': qmc_errors,
        'mc_rates': mc_rates,
        'qmc_rates': qmc_rates,
        'reference_value': ref_val
    }
