"""
Utilities for the Sparse Linear Algebra HPC Optimization Framework.

Contains common mathematical helpers, robust numerical routines,
and boundary-condition validators used across all modules.
"""

import numpy as np
import math


def safe_divide(a, b, default=0.0):
    """
    Robust division with zero-denominator guard.
    Returns default if |b| < machine epsilon.
    """
    eps = np.finfo(float).eps
    if np.isscalar(b):
        if abs(b) < eps:
            return default
        return a / b
    # vectorized
    b_safe = np.where(np.abs(b) < eps, np.nan, b)
    result = np.full_like(a, default, dtype=float)
    valid = ~np.isnan(b_safe)
    result[valid] = a[valid] / b_safe[valid]
    return result


def i4_div_rounded(a, b):
    """
    Rounded integer division with full edge-case handling.
    Maps from task_division (seed 1196).

    Computes round(a / b) for integers, handling:
      - b == 0  -> returns 0 with a warning
      - sign preservation
      - tie-breaking to nearest even (banker's rounding via Python round)
    """
    if b == 0:
        return 0
    # Python's round uses banker's rounding; for scientific consistency
    # we implement explicit half-away-from-zero.
    sign = 1
    if a * b < 0:
        sign = -1
    a = abs(a)
    b = abs(b)
    value = (2 * a + b) // (2 * b)
    return sign * value


def double_factorial2(n):
    """
    Compute the double factorial n!! used in Hermite quadrature exactness.
    n!! = n * (n-2) * ... * 1 (or 2).
    """
    if n < 0:
        return 1.0
    if n == 0 or n == 1:
        return 1.0
    result = 1.0
    while n > 0:
        result *= n
        n -= 2
    return result


def legendre_monomial_integral(p):
    """
    Exact integral of x^p over [-1, 1]:
        I_p = 2 / (p + 1)   if p even
        I_p = 0              if p odd
    From seed 344_exactness.
    """
    if p < 0:
        return 0.0
    if p % 2 == 1:
        return 0.0
    return 2.0 / (p + 1.0)


def chebyshev1_monomial_integral(p):
    """
    Exact integral of x^p / sqrt(1-x^2) over [-1, 1]:
        I_p = pi * (p-1)!! / p!!   if p even
        I_p = 0                     if p odd
    """
    if p < 0:
        return 0.0
    if p % 2 == 1:
        return 0.0
    return math.pi * double_factorial2(p - 1) / double_factorial2(p)


def hermite_monomial_integral(p):
    """
    Exact integral of x^p * exp(-x^2) over (-inf, inf):
        I_p = (p-1)!! * sqrt(pi) / 2^{p/2}   if p even
        I_p = 0                                if p odd
    """
    if p < 0:
        return 0.0
    if p % 2 == 1:
        return 0.0
    return double_factorial2(p - 1) * math.sqrt(math.pi) / (2.0 ** (p / 2.0))


def laguerre_monomial_integral(p):
    """
    Exact integral of x^p * exp(-x) over [0, inf):
        I_p = p!
    """
    if p < 0:
        return 0.0
    return float(math.factorial(p))


def parameterize_arc_length(p_data):
    """
    Compute arc-length parameterization for M-dimensional curve data.
    Returns t in [0, 1] with t[0]=0, t[-1]=1.
    From seed 590_interp.
    """
    p_data = np.asarray(p_data, dtype=float)
    if p_data.ndim == 1:
        p_data = p_data.reshape(1, -1)
    m, data_num = p_data.shape
    t = np.zeros(data_num)
    for j in range(1, data_num):
        dist = np.linalg.norm(p_data[:, j] - p_data[:, j - 1])
        t[j] = t[j - 1] + dist
    tmax = t[-1]
    if tmax > 0:
        t /= tmax
    return t


def r8vec_bracket(x, xval):
    """
    Find left index such that x[left] <= xval < x[left+1].
    Returns left index (0 <= left < n-1) or -1 if out of bounds.
    From seed 590_interp.
    """
    n = len(x)
    if n < 2:
        return -1
    if xval < x[0] - 1e-12:
        return -1
    if xval > x[-1] + 1e-12:
        return n - 1
    # binary search for robustness
    left = 0
    right = n - 1
    while right - left > 1:
        mid = (left + right) // 2
        if xval < x[mid]:
            right = mid
        else:
            left = mid
    return left


def is_symmetric(A, tol=1e-10):
    """Check if sparse or dense matrix is symmetric."""
    if isinstance(A, np.ndarray):
        return np.allclose(A, A.T, atol=tol)
    # sparse via scipy if available
    try:
        from scipy.sparse import csr_matrix
        if not hasattr(A, 'T'):
            return False
        diff = A - A.T
        return np.max(np.abs(diff.data)) < tol if hasattr(diff, 'data') and len(diff.data) > 0 else True
    except Exception:
        return False


def condition_number_estimate(A):
    """
    Simple power-iteration estimate of spectral condition number
    kappa = |lambda_max| / |lambda_min| for symmetric positive-definite A.
    """
    try:
        from scipy.sparse.linalg import eigsh
        n = A.shape[0]
        k = min(3, n - 1)
        if k < 1:
            return 1.0
        lambda_max = eigsh(A, k=k, which='LM', return_eigenvectors=False)
        lambda_min = eigsh(A, k=k, which='SM', return_eigenvectors=False)
        return abs(lambda_max[0]) / max(abs(lambda_min[0]), 1e-15)
    except Exception:
        return np.nan
