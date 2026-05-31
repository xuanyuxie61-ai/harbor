"""
qmc_optimizer.py
----------------
Quasi-Monte Carlo (QMC) parameter optimization for plasma coating
design using Sobol sequences, lattice rules, and probability-based
uncertainty quantification.

Incorporates core ideas from:
  - 1097_sobol        (Sobol low-discrepancy sequences)
  - 654_lattice_rule  (Lattice-rule multidimensional integration)
  - 918_prob          (Probability distributions and random sampling)
"""

import numpy as np
import math
from utils import clamp, ensure_positive


# ---------------------------------------------------------------------------
# Sobol-like sequence generator (simplified directional numbers for dim <= 10)
# ---------------------------------------------------------------------------
_DIRECTION_NUMBERS = {
    1: {"m": [1], "a": 0, "s": 1},
    2: {"m": [1, 3], "a": 1, "s": 2},
    3: {"m": [1, 3, 1], "a": 1, "s": 3},
    4: {"m": [1, 1, 1], "a": 1, "s": 3},
    5: {"m": [1, 3, 3], "a": 2, "s": 3},
    6: {"m": [1, 3, 5], "a": 1, "s": 3},
    7: {"m": [1, 1, 5], "a": 4, "s": 3},
    8: {"m": [1, 3, 7], "a": 2, "s": 3},
    9: {"m": [1, 3, 7], "a": 4, "s": 3},
    10: {"m": [1, 1, 5], "a": 7, "s": 3},
}


def sobol_sequence(dim_num: int, n: int, skip: int = 0) -> np.ndarray:
    """
    Generate a Sobol-like low-discrepancy sequence in [0,1]^dim_num.

    Parameters
    ----------
    dim_num : int
        Dimensionality (1..10 supported).
    n : int
        Number of points to generate.
    skip : int
        Number of initial points to skip.

    Returns
    -------
    points : (n, dim_num) ndarray
    """
    dim_num = int(clamp(dim_num, 1, 10))
    n = max(int(n), 1)
    skip = max(int(skip), 0)

    L = max(int(math.ceil(math.log2(n + skip + 1))), 1)
    points = np.zeros((n, dim_num), dtype=float)

    for d in range(dim_num):
        dn = _DIRECTION_NUMBERS.get(d + 1, _DIRECTION_NUMBERS[1])
        s = dn["s"]
        a = dn["a"]
        m = dn["m"][:s]
        while len(m) < L:
            m.append(0)

        # Extend m using recurrence
        for j in range(s, L):
            m[j] = m[j - s] ^ (m[j - s] << s)
            for k in range(1, s):
                m[j] ^= ((a >> (k - 1)) & 1) * (m[j - k] << k)
            # Keep within bits
            m[j] &= (1 << (j + 1)) - 1

        v = np.zeros(L, dtype=int)
        for j in range(L):
            v[j] = m[j] << (L - 1 - j)

        x = 0
        g = skip
        # Gray code initialization
        for j in range(L):
            if ((g >> j) & 1):
                x ^= v[j]

        for i in range(n):
            points[i, d] = x / (2.0 ** L)
            # Gray code increment
            g = i + skip
            c = 0
            while ((g >> c) & 1) == 1:
                c += 1
            if c < L:
                x ^= v[c]

    return points


# ---------------------------------------------------------------------------
# Lattice rule integration (adapted from 654_lattice_rule)
# ---------------------------------------------------------------------------
def lattice_rule_integrate(
    func,
    dim_num: int,
    m: int,
    z: np.ndarray = None,
) -> float:
    """
    Evaluate a periodic integrand over [0,1]^dim_num using a rank-1 lattice rule.

        Q = (1/m) * sum_{j=0}^{m-1} f( (j/m * z) mod 1 )

    Parameters
    ----------
    func : callable
        f(x_vec) -> float
    dim_num : int
    m : int
        Number of lattice points.
    z : (dim_num,) ndarray of int, optional
        Generating vector.  Defaults to a simple Fibonacci-like vector.

    Returns
    -------
    quad : float
        Integral estimate.
    """
    dim_num = int(dim_num)
    m = max(int(m), 1)
    if z is None:
        z = np.ones(dim_num, dtype=int)
        for d in range(1, dim_num):
            z[d] = (z[d - 1] * 177) % m
    else:
        z = np.asarray(z, dtype=int)

    total = 0.0
    for j in range(m):
        x = np.mod(j * z / m, 1.0)
        total += func(x)
    return total / m


def fibonacci_lattice_integrate_2d(func, m: int) -> float:
    """
    2-D Fibonacci lattice rule (optimal for periodic integrands).

    Uses generating vector z = [1, F_{k-1}] where F_k ≈ m.
    """
    m = max(int(m), 1)
    # Find a Fibonacci number close to m
    a, b = 1, 1
    while b < m:
        a, b = b, a + b
    z = np.array([1, a], dtype=int)
    return lattice_rule_integrate(func, 2, m, z=z)


# ---------------------------------------------------------------------------
# Probability distributions (adapted from 918_prob)
# ---------------------------------------------------------------------------
def normal_pdf(x: float, mu: float = 0.0, sigma: float = 1.0) -> float:
    """Normal (Gaussian) probability density function."""
    sigma = max(float(sigma), 1e-12)
    return (1.0 / (sigma * math.sqrt(2.0 * math.pi))) * math.exp(-0.5 * ((x - mu) / sigma) ** 2)


def normal_cdf(x: float, mu: float = 0.0, sigma: float = 1.0) -> float:
    """Normal cumulative distribution function (error function)."""
    sigma = max(float(sigma), 1e-12)
    return 0.5 * (1.0 + math.erf((x - mu) / (sigma * math.sqrt(2.0))))


def normal_sample(n: int, mu: float = 0.0, sigma: float = 1.0, seed: int = None) -> np.ndarray:
    """Generate n Normal random samples using Box-Muller."""
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()
    return rng.normal(loc=mu, scale=max(sigma, 1e-12), size=n)


def gamma_sample(n: int, shape: float = 1.0, scale: float = 1.0, seed: int = None) -> np.ndarray:
    """Generate n Gamma-distributed random samples."""
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()
    return rng.gamma(shape=max(shape, 1e-12), scale=max(scale, 1e-12), size=n)


def uniform_sample(n: int, low: float = 0.0, high: float = 1.0, seed: int = None) -> np.ndarray:
    """Generate n uniform random samples."""
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()
    return rng.uniform(low=low, high=high, size=n)


# ---------------------------------------------------------------------------
# QMC-based coating parameter optimization
# ---------------------------------------------------------------------------
def optimize_coating_parameters_qmc(
    objective_func,
    dim: int,
    n_samples: int = 512,
    param_bounds: np.ndarray = None,
    seed: int = 42,
) -> tuple:
    """
    Optimize coating parameters by evaluating the objective function on
    a Sobol sequence and selecting the best point.

    Parameters
    ----------
    objective_func : callable
        f(params_vec) -> float (to be minimized).
    dim : int
        Number of parameters.
    n_samples : int
        Number of Sobol points.
    param_bounds : (dim, 2) ndarray, optional
        Lower and upper bounds for each parameter.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    (best_params, best_value)
    """
    dim = int(dim)
    n_samples = max(int(n_samples), 4)

    points = sobol_sequence(dim, n_samples, skip=seed % 64)

    if param_bounds is None:
        param_bounds = np.array([[0.0, 1.0]] * dim, dtype=float)
    else:
        param_bounds = np.asarray(param_bounds, dtype=float)
        if param_bounds.shape != (dim, 2):
            raise ValueError("param_bounds must have shape (dim, 2).")

    best_value = float("inf")
    best_params = None

    for i in range(n_samples):
        p = points[i]
        # Map [0,1] to [low, high]
        params = param_bounds[:, 0] + p * (param_bounds[:, 1] - param_bounds[:, 0])
        val = objective_func(params)
        if val < best_value:
            best_value = val
            best_params = params.copy()

    return best_params, best_value


def monte_carlo_uncertainty_propagation(
    func,
    param_means: np.ndarray,
    param_stds: np.ndarray,
    n_mc: int = 1000,
    seed: int = 42,
) -> tuple:
    """
    Propagate Gaussian parameter uncertainties through a model using
    Monte Carlo sampling.

    Parameters
    ----------
    func : callable
        f(params_vec) -> float
    param_means : (dim,) ndarray
    param_stds : (dim,) ndarray
    n_mc : int
    seed : int

    Returns
    -------
    (mean, std, samples)
    """
    param_means = np.asarray(param_means, dtype=float)
    param_stds = np.asarray(param_stds, dtype=float)
    dim = param_means.size
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    samples = np.zeros(n_mc, dtype=float)
    for i in range(n_mc):
        params = param_means + rng.normal(size=dim) * param_stds
        samples[i] = func(params)

    return float(np.mean(samples)), float(np.std(samples)), samples
