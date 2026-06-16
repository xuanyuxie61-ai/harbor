# -*- coding: utf-8 -*-
"""
Monte-Carlo quadrature for path-integral observables of the TFIM.

Adapted from the ``941_quad_monte_carlo`` seed project, which uses
random sampling to estimate 1-D integrals.  We extend the idea to
the multi-dimensional integral that defines the quantum partition
function:

    Z(L, beta, lam)  =  int D[sigma]  exp( - S_E[sigma] )

where the Euclidean action on the L x M Trotter lattice is
    S_E = - sum_{i, tau} [ K_tau sigma_{i, tau} sigma_{i, tau+1}
                          + K_x   sigma_{i, tau} sigma_{i+1, tau} ].

We provide:

  1. Plain MC quadrature of <O>  = int O exp(-S) / int exp(-S)
  2. Variance-reduced control-variate estimator using the conformal
     2-point function as a reference.
  3. Bootstrap error bars on any scalar observable.

This module is *distinct* from ``mc_path_integral.py``: here we treat
the *k-space integral* of the single-particle dispersion to compute
exact thermodynamic observables in the L -> infinity limit.
"""

from __future__ import annotations
from typing import Callable, Tuple
import numpy as np
try:
    from . import constants as C
except ImportError:
    import constants as C


# ---------------------------------------------------------------------------
# Plain MC quadrature  (replicates 941_quad_monte_carlo)
# ---------------------------------------------------------------------------
def mc_quadrature_1d(f: Callable[[np.ndarray], np.ndarray],
                      a: float, b: float,
                      n_samples: int = 10000,
                      seed: int = 0) -> Tuple[float, float]:
    """Estimate  I = int_a^b f(x) dx  by uniform MC sampling.
    Returns (estimate, standard_error)."""
    rng = np.random.default_rng(seed)
    xs = rng.uniform(a, b, size=n_samples)
    fs = f(xs)
    est = (b - a) * float(np.mean(fs))
    se = (b - a) * float(np.std(fs, ddof=1) / np.sqrt(n_samples))
    return est, se


def mc_quadrature_nd(f: Callable[[np.ndarray], np.ndarray],
                      lo: np.ndarray, hi: np.ndarray,
                      n_samples: int = 20000,
                      seed: int = 1) -> Tuple[float, float]:
    """Multi-dimensional uniform MC quadrature on a box."""
    lo = np.asarray(lo)
    hi = np.asarray(hi)
    dim = len(lo)
    rng = np.random.default_rng(seed)
    xs = rng.uniform(0.0, 1.0, size=(n_samples, dim))
    xs = lo + xs * (hi - lo)
    fs = f(xs)
    vol = float(np.prod(hi - lo))
    est = vol * float(np.mean(fs))
    se = vol * float(np.std(fs, ddof=1) / np.sqrt(n_samples))
    return est, se


# ---------------------------------------------------------------------------
# TFIM thermodynamic integrals
# ---------------------------------------------------------------------------
def tfim_free_energy_density_integrand(k: np.ndarray,
                                         J: float, h: float,
                                         beta: float) -> np.ndarray:
    """Integrand of the thermodynamic-limit free-energy density:
        f = - (1/beta) * (1/2pi) int_0^{2 pi} log(2 cosh(beta eps_k / 2)) dk
    The integral is over the first Brillouin zone; we restrict to
    [0, pi] by symmetry.
    """
    lam = h / J if J != 0.0 else 0.0
    eps = 2.0 * abs(J) * np.sqrt(np.maximum(
        1.0 + lam * lam - 2.0 * lam * np.cos(k), 0.0))
    # log(2 cosh(beta eps/2)) ~ beta eps/2 for large beta eps
    arg = 0.5 * beta * eps
    return np.log(2.0 * np.cosh(np.minimum(arg, 500.0)))


def free_energy_density(L: int, J: float, h: float, beta: float,
                          n_samples: int = 10000,
                          seed: int = 0) -> Tuple[float, float]:
    """MC estimate of f(L, J, h, beta).  The finite-L correction is
    captured by replacing the integral with a discrete sum."""
    def f(k):
        return tfim_free_energy_density_integrand(k, J, h, beta)
    # MC on [0, pi]
    est, se = mc_quadrature_1d(f, 0.0, C.PI, n_samples=n_samples, seed=seed)
    f_bulk = -est / (beta * C.PI)
    se = se / (beta * C.PI)
    # Finite-L correction:  - pi c v / (6 L^2)
    v = 2.0 * abs(J) * abs(1.0 - (h / J) ** 2) if abs(J) > C.EPS_NUM else 1.0
    finite = -C.PI * 0.5 * max(v, C.EPS_NUM) / (6.0 * L * L)
    return f_bulk + finite / beta, se


# ---------------------------------------------------------------------------
# Control-variate estimator
# ---------------------------------------------------------------------------
def control_variate_estimate(f: Callable[[np.ndarray], np.ndarray],
                              g: Callable[[np.ndarray], np.ndarray],
                              a: float, b: float,
                              g_exact: float,
                              n_samples: int = 10000,
                              seed: int = 2,
                              alpha: float = None) -> Tuple[float, float]:
    """Variance-reduced estimator using g as a control variate with
    known integral ``g_exact = int_a^b g(x) dx``.

    Under uniform sampling on [a, b] the mean of g is g_exact / (b-a);
    we subtract the zero-mean fluctuation alpha * (g - E[g]) from f.
    alpha* = Cov(f, g) / Var(g) estimated from the sample.
    """
    rng = np.random.default_rng(seed)
    xs = rng.uniform(a, b, size=n_samples)
    fs = f(xs)
    gs = g(xs)
    g_mean = g_exact / max(b - a, C.EPS_NUM)
    if alpha is None:
        cov = float(np.cov(fs, gs, ddof=1)[0, 1])
        var_g = float(np.var(gs, ddof=1))
        alpha = cov / max(var_g, C.EPS_NUM)
    hs = fs - alpha * (gs - g_mean)
    est = (b - a) * float(np.mean(hs))
    se = (b - a) * float(np.std(hs, ddof=1) / np.sqrt(n_samples))
    return est, se


# ---------------------------------------------------------------------------
# Bootstrap error bar
# ---------------------------------------------------------------------------
def bootstrap_statistic(values: np.ndarray,
                          statistic: Callable[[np.ndarray], float] = np.mean,
                          n_boot: int = 500,
                          alpha: float = 0.05,
                          seed: int = 3) -> Tuple[float, float, float]:
    """Return (estimate, lower, upper) for a bootstrap (1-alpha) CI
    on ``statistic(values)``."""
    rng = np.random.default_rng(seed)
    n = len(values)
    stats = np.zeros(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        stats[b] = statistic(values[idx])
    estimate = float(statistic(values))
    lower = float(np.percentile(stats, 100 * alpha / 2))
    upper = float(np.percentile(stats, 100 * (1 - alpha / 2)))
    return estimate, lower, upper
