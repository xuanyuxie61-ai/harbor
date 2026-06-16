"""
spectral_quadrature.py — Gauss-Chebyshev / Gauss-Hermite quadrature for matched filtering.

Matched filtering of a gravitational-wave signal  h(t) against a template bank
{T_theta(t)} requires evaluating the noise-weighted inner product

    <h, T> = 4 Re int_0^infty |h_tilde(f)|^2 / S_n(f) df,

where S_n(f) is the one-sided power spectral density of the detector noise.

This module provides:
  * Gauss-Chebyshev quadrature of the first kind
        int_{-1}^{1} f(x) / sqrt(1-x^2) dx  ~  sum w_k f(x_k),
    which naturally matches the 1/f frequency weighting of some noise models;
  * Gauss-Hermite quadrature for rapidly decaying integrands
        int_{-infty}^{+infty} f(x) exp(-x^2) dx;
  * nested Clenshaw-Curtis rules for adaptive refinement;
  * exactness tests (monomial integration) that mirror the classical
    hermite_exactness / chebyshev1_exactness seed projects but in the
    gravitational-wave context.
"""

from __future__ import annotations
import math
import numpy as np
from typing import Callable, Tuple, Dict


# ---------------------------------------------------------------------------
#  Gauss-Chebyshev type I quadrature
# ---------------------------------------------------------------------------
def gauss_chebyshev1_nodes_weights(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return nodes and weights for n-point Gauss-Chebyshev type I rule.

    Nodes:  x_k = cos( (2k-1) pi / (2n) ),  k = 1, ..., n.
    Weights: w_k = pi / n  (all equal).
    Exact for  int_{-1}^{1} f(x) / sqrt(1-x^2) dx.
    """
    k = np.arange(1, n + 1)
    x = np.cos((2.0 * k - 1.0) * math.pi / (2.0 * n))
    w = np.full(n, math.pi / n)
    return x, w


def gauss_chebyshev1_integrate(f: Callable[[np.ndarray], np.ndarray],
                               n: int) -> float:
    """Evaluate int_{-1}^{1} f(x) / sqrt(1-x^2) dx with n-point GC1."""
    x, w = gauss_chebyshev1_nodes_weights(n)
    return float(np.sum(w * f(x)))


# ---------------------------------------------------------------------------
#  Gauss-Hermite quadrature (Golub-Welsch / eigenvalue method)
# ---------------------------------------------------------------------------
def gauss_hermite_nodes_weights(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """Compute n-point Gauss-Hermite nodes and weights via the Jacobi matrix.

    The tridiagonal Jacobi matrix for Hermite polynomials has
        a_k = 0,    b_k = sqrt(k / 2)
    so the nodes are eigenvalues and  w_i = pi (v_{i,0})^2  where v_{i,0}
    is the first component of the i-th normalised eigenvector.
    """
    b = np.sqrt(np.arange(1, n) / 2.0)
    J = np.diag(b, k=1) + np.diag(b, k=-1)
    eigvals, eigvecs = np.linalg.eigh(J)
    w = math.sqrt(math.pi) * eigvecs[0, :] ** 2
    return eigvals, w


def gauss_hermite_integrate(f: Callable[[np.ndarray], np.ndarray],
                            n: int) -> float:
    """Evaluate  int_{-infty}^{+infty} f(x) exp(-x^2) dx  with n-point GH."""
    x, w = gauss_hermite_nodes_weights(n)
    return float(np.sum(w * f(x)))


# ---------------------------------------------------------------------------
#  Nested Clenshaw-Curtis quadrature
# ---------------------------------------------------------------------------
def clenshaw_curtis_nodes(n: int) -> np.ndarray:
    """Compute n nested Clenshaw-Curtis nodes on [-1, +1].

    The sequence is built hierarchically: 1/2, {0,1}, {1/4, 3/4}, ...
    and mapped to [-1, +1] via x = cos(pi * t).
    """
    x = np.zeros(n)
    if n >= 1:
        x[0] = 0.5
    if n >= 2:
        x[1] = 1.0
    if n >= 3:
        x[2] = 0.0
    m = 3
    d = 2.0
    while m < n:
        tu = np.arange(d + 1, 2 * d, 2)
        td = np.arange(d - 1, 0, -2)
        t = np.reshape(np.stack([tu, td], axis=1), (d,))[:min(d, n - m)]
        x[m: m + len(t)] = t / (2.0 * d)
        m += len(t)
        d *= 2.0
    return np.cos(x * math.pi)


def clenshaw_curtis_weights(n: int) -> np.ndarray:
    """Compute Clenshaw-Curtis weights for n nodes via the DCT-based formula.

    For interior points:  w_j = (2/n) sum_{k=0}^{n/2} '' b_k cos(2 pi j k / n)
    with b_0 = 1, b_{n/2} = 1/3 (when n is even), and the double-prime
    indicating halving of the first and last term.
    """
    if n == 1:
        return np.array([2.0])
    x = clenshaw_curtis_nodes(n)
    # use the explicit formula for n odd (simpler)
    N = n - 1
    theta = np.arccos(x)
    w = np.zeros(n)
    for j in range(n):
        s = 0.0
        for k in range(N // 2 + 1):
            c_k = 1.0 if k == 0 or k == N // 2 else 2.0
            s += c_k * math.cos(2.0 * k * theta[j]) / (1.0 - 4.0 * k * k + 1.0e-300)
        w[j] = 2.0 * s / N
    # endpoints get half weight
    w[0] *= 0.5
    w[-1] *= 0.5
    return w


# ---------------------------------------------------------------------------
#  Rescaling to arbitrary intervals
# ---------------------------------------------------------------------------
def rescale_rule(a: float, b: float,
                 x: np.ndarray, w: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Map a quadrature rule from [-1, +1] to [a, b].

    x_new = (a + b) / 2 + (b - a) / 2 * x
    w_new = (b - a) / 2 * w
    """
    mid = 0.5 * (a + b)
    half = 0.5 * (b - a)
    return mid + half * x, half * w


# ---------------------------------------------------------------------------
#  Exactness tests (adapted from hermite_exactness / chebyshev1_exactness)
# ---------------------------------------------------------------------------
def chebyshev1_exactness(n: int, degree_max: int) -> Dict[int, float]:
    """Test the polynomial exactness of an n-point GC1 rule.

    The rule integrates x^k / sqrt(1-x^2) exactly for  k <= 2n - 1.
    Returns a dict {k: |approx - exact|}.
    """
    x, w = gauss_chebyshev1_nodes_weights(n)
    errors = {}
    for k in range(degree_max + 1):
        approx = float(np.sum(w * x ** k))
        # exact: 0 for odd k; Beta((k+1)/2, 1/2) = pi * (k-1)!! / k!! for even k
        if k % 2 == 1:
            exact = 0.0
        else:
            num = 1.0
            for j in range(1, k, 2):
                num *= j
            den = 1.0
            for j in range(2, k + 1, 2):
                den *= j
            exact = math.pi * num / den
        errors[k] = abs(approx - exact)
    return errors


def hermite_exactness(n: int, degree_max: int) -> Dict[int, float]:
    """Test exactness of n-point Gauss-Hermite for x^k exp(-x^2) integrals.

    Exact value: 0 for odd k;  Gamma((k+1)/2) for even k.
    """
    x, w = gauss_hermite_nodes_weights(n)
    errors = {}
    for k in range(degree_max + 1):
        approx = float(np.sum(w * x ** k))
        if k % 2 == 1:
            exact = 0.0
        else:
            exact = math.gamma(0.5 * (k + 1))
        errors[k] = abs(approx - exact)
    return errors


# ---------------------------------------------------------------------------
#  Matched-filter SNR via Gauss-Hermite
# ---------------------------------------------------------------------------
def matched_filter_snr(h_tilde: Callable[[np.ndarray], np.ndarray],
                       template: Callable[[np.ndarray], np.ndarray],
                       psd: Callable[[np.ndarray], np.ndarray],
                       f_min: float, f_max: float,
                       n_quad: int = 64) -> float:
    """Compute the optimal matched-filter SNR using Gauss-Hermite quadrature.

    SNR^2 = 4 Re int_{f_min}^{f_max} |h_tilde(f)|^2 / S_n(f) df.

    The integral is mapped to (-infty, +infty) via  f = f_c + w tanh(u)  or
    simply computed by Gauss-Chebyshev on the mapped interval.
    """
    # map [f_min, f_max] -> [-1, +1]
    x_nodes, w_nodes = gauss_chebyshev1_nodes_weights(n_quad)
    f_nodes, w_f = rescale_rule(f_min, f_max, x_nodes, w_nodes)
    integrand = np.abs(h_tilde(f_nodes)) ** 2 / psd(f_nodes)
    rho2 = 4.0 * float(np.real(np.sum(w_f * integrand)))
    return math.sqrt(max(rho2, 0.0))


# ---------------------------------------------------------------------------
#  Simple noise PSD (aLIGO design curve, analytic fit)
# ---------------------------------------------------------------------------
def aligo_psd_design(f: np.ndarray, f0: float = 215.0) -> np.ndarray:
    """Analytic fit to the aLIGO zero-detuning high-power PSD.

        S_n(f) = S_0 [ x^{-15.6} + 0.1 x^{-2.6} + 0.5 + 2.6 x + 0.5 x^2 ]
    with x = f / f0 and S_0 = 1.0e-49 Hz^{-1}.
    """
    S0 = 1.0e-49
    x = np.maximum(np.asarray(f, dtype=float) / f0, 1.0e-6)
    return S0 * (x ** (-15.6) + 0.1 * x ** (-2.6) + 0.5 + 2.6 * x + 0.5 * x ** 2)
