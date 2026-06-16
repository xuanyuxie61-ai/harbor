"""
uncertainty_propagation.py - Monte Carlo Uncertainty Propagation via Surrogates

This module implements uncertainty propagation methods using the
polynomial chaos surrogate, integrating:
  - Monte Carlo sampling on simplicial domains (from wedge_monte_carlo)
  - Moment computation via PCE coefficients
  - Probability density estimation

Mathematical Framework
----------------------
### Monte Carlo Integration ###

For QoI Y = M(X) with X ~ f_X:
  E[g(Y)] ≈ (1/N) sum_{i=1}^N g(M(x_i))
  with error O(1/sqrt(N)) by CLT.

Using the surrogate Y ≈ Y^PCE:
  E[Y] ≈ c_0  (exact from PCE)
  Var[Y] ≈ sum_{j>0} c_j^2 ||Psi_j||^2  (exact from PCE)

### Monte Carlo on Simplices ###

For integration over the standard d-simplex:
  Delta_d = {x in R^d : x_i >= 0, sum x_i <= 1}

Sampling: X_i = E_i / sum(E_j) where E_j ~ Exp(1) (Dirichlet)
Volume: V(Delta_d) = 1/d!

Exact monomial integrals:
  integral_{Delta_d} prod x_i^{e_i} dx = prod Gamma(e_i + 1) / Gamma(sum(e_i) + d + 1)

### Kernel Density Estimation ###

  f_hat(y) = (1/(N*h)) sum K((y - Y_i)/h)
with Gaussian kernel K(u) = (1/sqrt(2*pi)) exp(-u^2/2)
Bandwidth: h = 1.06 * sigma * N^{-1/5} (Silverman's rule)

### Confidence Intervals ###

  P(Y in [y_lo, y_hi]) = 1 - alpha
  For normal: y_lo = mu - z_{alpha/2} * sigma
  For non-normal: use percentile method from MC samples.
"""

import numpy as np
from numpy.typing import NDArray
from typing import Dict, List, Optional, Tuple, Any
import math
from polynomial_chaos import PolynomialChaosSurrogate
from numerical_utils import beta_function


# ---------------------------------------------------------------------------
# Monte Carlo Sampling
# ---------------------------------------------------------------------------

def sample_uniform_cube(n: int, d: int,
                        lower: NDArray, upper: NDArray,
                        seed: int = 42) -> NDArray:
    """
    Generate uniform random samples in a d-dimensional box.

    Parameters
    ----------
    n : int
        Number of samples.
    d : int
        Dimension.
    lower : ndarray(d,)
        Lower bounds.
    upper : ndarray(d,)
        Upper bounds.
    seed : int
        Random seed.

    Returns
    -------
    ndarray(n, d)
        Uniform samples in [lower, upper].
    """
    rng = np.random.RandomState(seed)
    samples = rng.random((n, d))
    return lower + samples * (upper - lower)


def sample_simplex(n: int, d: int, seed: int = 42) -> NDArray:
    """
    Generate uniform random samples on the standard d-simplex.

    Delta_d = {x in R^d : x_i >= 0, sum x_i <= 1}

    Method: Generate d+1 exponential random variables, normalize.
    This is equivalent to sampling from Dirichlet(1, 1, ..., 1).

    Volume of Delta_d = 1/d!

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
    ndarray(n, d)
        Points on the standard simplex.
    """
    rng = np.random.RandomState(seed)
    E = rng.exponential(1.0, (n, d + 1))
    S = E.sum(axis=1, keepdims=True)
    samples = E[:, :d] / S  # Drop last coordinate (1 - sum)
    return samples


def sample_sphere(n: int, d: int, seed: int = 42) -> NDArray:
    """
    Generate uniform random samples on the unit d-sphere.

    Method: Normalize d-dimensional standard normal vectors.

    Surface area of S^{d-1} = 2 * pi^{d/2} / Gamma(d/2)
    """
    rng = np.random.RandomState(seed)
    Z = rng.randn(n, d)
    norms = np.linalg.norm(Z, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-30)
    return Z / norms


def exact_simplex_monomial_integral(e: NDArray) -> float:
    """
    Exact integral of monomial prod(x_i^{e_i}) over standard simplex.

    integral_{Delta_d} prod_{i=1}^d x_i^{e_i} dx
      = prod_{i=1}^d Gamma(e_i + 1) / Gamma(sum(e_i) + d + 1)

    This generalizes the wedge integration from wedge_monte_carlo
    to arbitrary-dimensional simplices.
    """
    d = len(e)
    log_num = 0.0
    for ei in e:
        log_num += math.lgamma(ei + 1.0)
    log_den = math.lgamma(np.sum(e) + d + 1.0)
    return math.exp(log_num - log_den)


def monte_carlo_integration(surrogate: PolynomialChaosSurrogate,
                            n_samples: int = 10000,
                            lower: Optional[NDArray] = None,
                            upper: Optional[NDArray] = None,
                            seed: int = 42) -> Dict[str, Any]:
    """
    Monte Carlo integration using the surrogate model.

    Computes E[Y], Var[Y], and higher moments by sampling from
    the input distribution and evaluating the surrogate.

    Parameters
    ----------
    surrogate : PolynomialChaosSurrogate
        Fitted PCE surrogate.
    n_samples : int
        Number of Monte Carlo samples.
    lower, upper : ndarray(d,), optional
        Parameter bounds (uniform distribution).
    seed : int
        Random seed.

    Returns
    -------
    dict with 'mean', 'variance', 'std', 'percentiles', 'samples', 'y_samples'
    """
    d = surrogate.d
    if lower is None:
        lower = -np.ones(d)
    if upper is None:
        upper = np.ones(d)

    # Generate samples
    X = sample_uniform_cube(n_samples, d, lower, upper, seed)

    # Evaluate surrogate
    Y = surrogate.predict(X)

    # Statistics
    mean = float(np.mean(Y))
    variance = float(np.var(Y))
    std = float(np.std(Y))
    percentiles = {
        'p5': float(np.percentile(Y, 5)),
        'p25': float(np.percentile(Y, 25)),
        'p50': float(np.percentile(Y, 50)),
        'p75': float(np.percentile(Y, 75)),
        'p95': float(np.percentile(Y, 95))
    }

    # MC error estimate
    mc_error = std / math.sqrt(n_samples) if n_samples > 0 else 0.0

    return {
        'mean': mean,
        'variance': variance,
        'std': std,
        'mc_error': mc_error,
        'percentiles': percentiles,
        'y_samples': Y,
        'x_samples': X
    }


# ---------------------------------------------------------------------------
# Kernel Density Estimation
# ---------------------------------------------------------------------------

def kernel_density_estimate(y_samples: NDArray, n_grid: int = 200,
                            bandwidth: Optional[float] = None) -> Tuple[NDArray, NDArray]:
    """
    Gaussian kernel density estimation of output PDF.

    f_hat(y) = (1/(N*h)) sum_{i=1}^N K((y - Y_i)/h)
    with K(u) = (1/sqrt(2*pi)) * exp(-u^2/2)

    Bandwidth (Silverman's rule):
      h = 1.06 * sigma * N^{-1/5}

    Parameters
    ----------
    y_samples : ndarray(n,)
        Output samples.
    n_grid : int
        Number of grid points for PDF evaluation.
    bandwidth : float, optional
        Kernel bandwidth. If None, uses Silverman's rule.

    Returns
    -------
    y_grid : ndarray(n_grid,)
        Grid points.
    pdf : ndarray(n_grid,)
        Estimated PDF values.
    """
    n = len(y_samples)
    sigma = np.std(y_samples)
    mu = np.mean(y_samples)

    if bandwidth is None:
        # Silverman's rule of thumb
        h = 1.06 * sigma * n ** (-0.2) if sigma > 1e-14 else 1.0
    else:
        h = bandwidth

    h = max(h, 1e-14)

    y_grid = np.linspace(mu - 4 * sigma, mu + 4 * sigma, n_grid)
    pdf = np.zeros(n_grid)

    for yi in y_samples:
        pdf += np.exp(-0.5 * ((y_grid - yi) / h) ** 2) / (h * math.sqrt(2.0 * math.pi))

    pdf /= n

    # Normalize
    dy = y_grid[1] - y_grid[0] if n_grid > 1 else 1.0
    total = np.sum(pdf) * dy
    if total > 1e-14:
        pdf /= total

    return y_grid, pdf


# ---------------------------------------------------------------------------
# Confidence Intervals
# ---------------------------------------------------------------------------

def compute_confidence_intervals(y_samples: NDArray,
                                 confidence_level: float = 0.95) -> Dict[str, float]:
    """
    Compute confidence intervals from MC samples.

    Uses the percentile method (non-parametric):
      [y_lo, y_hi] = [percentile(alpha/2), percentile(1 - alpha/2)]

    For a Gaussian approximation:
      [y_lo, y_hi] = [mu - z * sigma, mu + z * sigma]
    where z = Phi^{-1}(1 - alpha/2).

    Parameters
    ----------
    y_samples : ndarray(n,)
        MC output samples.
    confidence_level : float
        Confidence level (e.g., 0.95 for 95%).

    Returns
    -------
    dict with 'lower', 'upper', 'mean', 'std', 'method'
    """
    alpha = 1.0 - confidence_level
    lower_pct = 100.0 * alpha / 2.0
    upper_pct = 100.0 * (1.0 - alpha / 2.0)

    y_lo = float(np.percentile(y_samples, lower_pct))
    y_hi = float(np.percentile(y_samples, upper_pct))
    mu = float(np.mean(y_samples))
    sigma = float(np.std(y_samples))

    return {
        'lower': y_lo,
        'upper': y_hi,
        'mean': mu,
        'std': sigma,
        'confidence_level': confidence_level,
        'width': y_hi - y_lo
    }


# ---------------------------------------------------------------------------
# Full Uncertainty Propagation Pipeline
# ---------------------------------------------------------------------------

def run_uncertainty_propagation(surrogate: PolynomialChaosSurrogate,
                                param_names: List[str],
                                param_bounds: Dict[str, Tuple[float, float]],
                                n_mc: int = 5000,
                                seed: int = 42) -> Dict[str, Any]:
    """
    Run complete uncertainty propagation pipeline.

    1. MC sampling from surrogate
    2. Moment computation (PCE-based and MC-based)
    3. Sobol sensitivity indices
    4. PDF estimation
    5. Confidence intervals

    Parameters
    ----------
    surrogate : PolynomialChaosSurrogate
        Fitted PCE surrogate.
    param_names : list of str
        Parameter names.
    param_bounds : dict
        Maps parameter names to (lower, upper) bounds.
    n_mc : int
        Number of MC samples.
    seed : int
        Random seed.

    Returns
    -------
    dict with comprehensive UQ results.
    """
    d = surrogate.d
    lower = np.array([param_bounds[name][0] for name in param_names[:d]])
    upper = np.array([param_bounds[name][1] for name in param_names[:d]])

    # 1. Monte Carlo propagation
    mc_result = monte_carlo_integration(surrogate, n_mc, lower, upper, seed)

    # 2. PCE-based moments
    pce_moments = surrogate.statistical_moments()

    # 3. Sobol indices
    sobol = surrogate.sobol_indices()

    # 4. PDF estimation
    y_grid, pdf = kernel_density_estimate(mc_result['y_samples'])

    # 5. Confidence intervals
    ci_95 = compute_confidence_intervals(mc_result['y_samples'], 0.95)
    ci_99 = compute_confidence_intervals(mc_result['y_samples'], 0.99)

    # Compile results
    results = {
        'mc_moments': {
            'mean': mc_result['mean'],
            'variance': mc_result['variance'],
            'std': mc_result['std'],
            'mc_error': mc_result['mc_error']
        },
        'pce_moments': pce_moments,
        'sobol_indices': {
            'first_order': {param_names[i]: float(sobol['first_order'][i])
                            for i in range(d)},
            'total_order': {param_names[i]: float(sobol['total_order'][i])
                            for i in range(d)},
            'interactions': {param_names[i]: float(sobol['interaction_order'][i])
                             for i in range(d)}
        },
        'pdf': {'y_grid': y_grid, 'pdf_values': pdf},
        'confidence_intervals': {
            '95pct': ci_95,
            '99pct': ci_99
        },
        'percentiles': mc_result['percentiles']
    }

    return results
