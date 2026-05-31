"""
sampling_distribution.py

Probability Sampling, Monte Carlo Validation, and Centroidal Voronoi Tessellation.

Scientific Background:
----------------------
1. Truncated Log-Normal Distribution:
   The log-normal PDF with parameters mu, sigma:
   
       f(x) = 1/(x sigma sqrt(2*pi)) * exp(-(ln x - mu)^2 / (2 sigma^2))
   
   Truncated to [a, b]:
       f_{ab}(x) = f(x) / (Phi((ln b - mu)/sigma) - Phi((ln a - mu)/sigma))
   
   Sampling via inverse CDF:
       X = exp(mu + sigma * Phi^{-1}(U))
   where U ~ Uniform[Phi((ln a - mu)/sigma), Phi((ln b - mu)/sigma)].

2. Walker Alias Method:
   For a discrete probability vector p[1..n], preprocess in O(n) to build:
   - y[i]: threshold values
   - a[i]: alias indices
   
   Sampling in O(1):
       i = random integer in [1, n]
       if U < y[i]: return i
       else: return a[i]

3. Centroidal Voronoi Tessellation (CVT):
   For a region Omega with density rho(x), the CVT minimizes:
   
       E = sum_{i=1}^n integral_{V_i} ||x - z_i||^2 rho(x) dx
   
   where V_i = {x in Omega : ||x - z_i|| <= ||x - z_j|| for all j}
   and z_i are the generators.
   
   Lloyd's algorithm iterates:
       z_i^{new} = integral_{V_i} x rho(x) dx / integral_{V_i} rho(x) dx

4. Disk Sampling (Monte Carlo):
   For the unit disk, uniform sampling:
       theta = 2*pi*U1
       r = sqrt(U2)
       x = r*cos(theta), y = r*sin(theta)

5. Application to Matrix Multiplication Validation:
   Random matrices with entries drawn from truncated log-normal distributions
   can be used to test numerical stability of parallel algorithms.
"""

import numpy as np
from typing import Tuple
import math


def normal_01_cdf(x: float) -> float:
    """
    Standard normal CDF using rational approximation.
    
    Phi(x) = 0.5 * [1 + erf(x / sqrt(2))]
    """
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    p = 0.3275911
    
    sign = 1.0 if x >= 0 else -1.0
    x = abs(x) / math.sqrt(2.0)
    
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-x * x)
    
    return 0.5 * (1.0 + sign * y)


def normal_01_cdf_inv(p: float) -> float:
    """
    Inverse standard normal CDF using rational approximation.
    
    For 0 < p < 1, find x such that Phi(x) = p.
    """
    if p <= 0.0:
        return -10.0
    if p >= 1.0:
        return 10.0
    
    # Rational approximation coefficients
    a = [
        -3.969683028665376e+01,
        2.209460984245205e+02,
        -2.759285104469687e+02,
        1.383577518672690e+02,
        -3.066479806614716e+01,
        2.506628277459239e+00
    ]
    b = [
        -5.447609879822406e+01,
        1.615858368580409e+02,
        -1.556989798598866e+02,
        6.680131188771972e+01,
        -1.328068155288572e+01
    ]
    c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
        4.374664141464968e+00,
        2.938163982698783e+00
    ]
    d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e+00,
        3.754408661907416e+00
    ]
    
    p_low = 0.02425
    p_high = 1.0 - p_low
    
    if p < p_low:
        q = math.sqrt(-2.0 * math.log(p))
        x = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        )
    elif p <= p_high:
        q = p - 0.5
        r = q * q
        x = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (
            (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
        )
    else:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        x = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        )
    
    return x


def log_normal_truncated_ab_sample(
    mu: float,
    sigma: float,
    a: float,
    b: float
) -> float:
    """
    Sample from truncated log-normal distribution on [a, b].
    
    Algorithm:
        cdf_a = Phi((ln a - mu) / sigma)   [if a > 0, else 0]
        cdf_b = Phi((ln b - mu) / sigma)
        U = Uniform(cdf_a, cdf_b)
        X = exp(mu + sigma * Phi^{-1}(U))
    
    Args:
        mu: location parameter
        sigma: scale parameter (sigma > 0)
        a, b: truncation limits (0 <= a < b)
    
    Returns:
        sample value
    """
    if sigma <= 0.0:
        raise ValueError("sigma must be positive")
    if a < 0.0:
        raise ValueError("a must be non-negative")
    if a >= b:
        raise ValueError("require a < b")
    
    ln_a = math.log(a) if a > 0 else -1e300
    ln_b = math.log(b)
    
    cdf_a = normal_01_cdf((ln_a - mu) / sigma) if a > 0 else 0.0
    cdf_b = normal_01_cdf((ln_b - mu) / sigma)
    
    if cdf_b <= cdf_a:
        return math.exp(mu)
    
    u = np.random.uniform(cdf_a, cdf_b)
    z = normal_01_cdf_inv(u)
    x = math.exp(mu + sigma * z)
    
    return max(a, min(b, x))


def walker_build(prob: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build Walker alias tables for discrete sampling.
    
    Given probability vector p[0..n-1], construct y and a such that:
    - y[i] in [0, 1]
    - a[i] in [0, n-1]
    - The sampling procedure produces i with probability p[i]
    
    Args:
        prob: probability vector (must sum to 1, non-negative)
    
    Returns:
        y: threshold array of shape (n,)
        a: alias array of shape (n,)
    """
    prob = np.asarray(prob, dtype=np.float64)
    if np.any(prob < -1e-12):
        raise ValueError("probabilities must be non-negative")
    n = len(prob)
    if n == 0:
        return np.array([]), np.array([])
    
    prob = prob / np.sum(prob)
    
    y = np.zeros(n)
    a = np.zeros(n, dtype=np.int64)
    
    # Work arrays
    over = []
    under = []
    
    for i in range(n):
        y[i] = n * prob[i]
        a[i] = i
        if y[i] > 1.0:
            over.append(i)
        elif y[i] < 1.0:
            under.append(i)
    
    while over and under:
        i = over.pop()
        j = under.pop()
        a[j] = i
        y[i] -= (1.0 - y[j])
        if y[i] > 1.0:
            over.append(i)
        elif y[i] < 1.0:
            under.append(i)
    
    # Remaining entries are already ~1
    return y, a


def walker_sampler(y: np.ndarray, a: np.ndarray) -> int:
    """
    Sample from Walker alias tables.
    
    Algorithm:
        i = random integer in [0, n-1]
        if U < y[i]: return i
        else: return a[i]
    
    Args:
        y: threshold array
        a: alias array
    
    Returns:
        sampled index
    """
    n = len(y)
    if n == 0:
        raise ValueError("Empty alias table")
    i = np.random.randint(0, n)
    u = np.random.uniform(0.0, 1.0)
    if u < y[i]:
        return int(i)
    else:
        return int(a[i])


def cvt_energy_2d(
    generators: np.ndarray,
    samples: np.ndarray
) -> float:
    """
    Compute CVT energy for 2D generators with uniform samples.
    
    E = (1/N_s) * sum_{s=1}^{N_s} min_i ||sample_s - generator_i||^2
    
    Args:
        generators: array of shape (n, 2)
        samples: array of shape (m, 2)
    
    Returns:
        energy: float
    """
    if generators.shape[0] == 0 or samples.shape[0] == 0:
        return 0.0
    
    # Compute all pairwise distances
    diffs = samples[:, np.newaxis, :] - generators[np.newaxis, :, :]
    dists_sq = np.sum(diffs ** 2, axis=2)
    min_dists = np.min(dists_sq, axis=1)
    return np.mean(min_dists)


def lloyd_step_2d(
    generators: np.ndarray,
    samples: np.ndarray
) -> np.ndarray:
    """
    One Lloyd iteration for CVT in 2D.
    
    Assign each sample to nearest generator, then move generator to centroid.
    
    Args:
        generators: current generators, shape (n, 2)
        samples: sample points, shape (m, 2)
    
    Returns:
        new_generators: updated generators
    """
    n = generators.shape[0]
    if n == 0:
        return generators.copy()
    
    sums = np.zeros((n, 2))
    counts = np.zeros(n)
    
    for s in samples:
        dists = np.sum((generators - s) ** 2, axis=1)
        nearest = np.argmin(dists)
        sums[nearest] += s
        counts[nearest] += 1
    
    new_gens = generators.copy()
    for i in range(n):
        if counts[i] > 0:
            new_gens[i] = sums[i] / counts[i]
        else:
            # No samples assigned, keep current or jitter
            new_gens[i] = generators[i]
    
    return new_gens


def disk01_positive_sample(n: int) -> np.ndarray:
    """
    Uniformly sample n points from the unit disk.
    
    Algorithm:
        theta = 2*pi*U1
        r = sqrt(U2)
        x = r*cos(theta), y = r*sin(theta)
    
    Args:
        n: number of samples
    
    Returns:
        points: array of shape (n, 2)
    """
    if n < 0:
        raise ValueError("n must be non-negative")
    theta = 2.0 * np.pi * np.random.uniform(0.0, 1.0, size=n)
    r = np.sqrt(np.random.uniform(0.0, 1.0, size=n))
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.column_stack([x, y])


def generate_random_matrix_lognormal(
    n: int,
    mu: float = 0.0,
    sigma: float = 1.0,
    a: float = 1e-6,
    b: float = 10.0
) -> np.ndarray:
    """
    Generate an n x n random matrix with truncated log-normal entries.
    
    Args:
        n: matrix size
        mu, sigma: log-normal parameters
        a, b: truncation limits
    
    Returns:
        matrix of shape (n, n)
    """
    if n < 0:
        raise ValueError("n must be non-negative")
    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            mat[i, j] = log_normal_truncated_ab_sample(mu, sigma, a, b)
    return mat


if __name__ == "__main__":
    # Test log-normal sampling
    samples = [log_normal_truncated_ab_sample(0.0, 1.0, 0.1, 5.0) for _ in range(1000)]
    print("Log-normal mean:", np.mean(samples), "std:", np.std(samples))
    
    # Test Walker alias
    prob = np.array([0.1, 0.3, 0.4, 0.2])
    y, a = walker_build(prob)
    counts = np.zeros(4)
    for _ in range(10000):
        counts[walker_sampler(y, a)] += 1
    print("Walker empirical frequencies:", counts / 10000)
    
    # Test CVT
    gens = np.random.rand(5, 2)
    samps = np.random.rand(1000, 2)
    E0 = cvt_energy_2d(gens, samps)
    for _ in range(10):
        gens = lloyd_step_2d(gens, samps)
    E1 = cvt_energy_2d(gens, samps)
    print("CVT energy before/after Lloyd:", E0, E1)
