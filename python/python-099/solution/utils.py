"""
utils.py
--------
Numerical utilities, environment detection, boundary checks,
Fresnel integrals, Newton iteration, and Jacobi relaxation.

Incorporates core ideas from:
  - 824_octopus  (environment detection)
  - 448_fresnel  (Fresnel sine/cosine integrals)
  - 808_nonlin_newton (Newton root finding)
  - 603_jacobi   (Jacobi iterative solver)
"""

import sys
import math
import numpy as np


# ---------------------------------------------------------------------------
# Environment detection (adapted from 824_octopus)
# ---------------------------------------------------------------------------
def is_running_in_ipython() -> bool:
    """Return True if the interpreter is IPython/Jupyter."""
    try:
        __IPYTHON__  # type: ignore
        return True
    except NameError:
        return False


def check_runtime_environment() -> dict:
    """Return a dictionary describing the runtime environment."""
    env = {
        "python_version": sys.version,
        "is_ipython": is_running_in_ipython(),
        "numpy_version": np.__version__,
    }
    return env


# ---------------------------------------------------------------------------
# Boundary & robustness helpers
# ---------------------------------------------------------------------------
def safe_sqrt(x: float) -> float:
    """Return sqrt(max(x, 0)) to avoid negative rounding errors."""
    return math.sqrt(max(float(x), 0.0))


def safe_divide(a: float, b: float, fallback: float = 0.0) -> float:
    """Return a / b if |b| > eps, else fallback."""
    eps = np.finfo(float).eps * max(1.0, abs(a))
    if abs(b) < eps:
        return fallback
    return a / b


def clamp(x: float, low: float, high: float) -> float:
    """Clamp x to the closed interval [low, high]."""
    return max(low, min(high, x))


def ensure_positive(x: np.ndarray, min_val: float = 1e-12) -> np.ndarray:
    """Ensure all entries of x are >= min_val."""
    return np.where(x < min_val, min_val, x)


# ---------------------------------------------------------------------------
# Fresnel integrals (adapted from 448_fresnel)
# ---------------------------------------------------------------------------
def fresnel_cos(x: float) -> float:
    """
    Compute the Fresnel cosine integral C(x) = integral_0^x cos(pi*t^2/2) dt.
    Uses series expansion for |x| < 2.5, backward recurrence for 2.5 <= |x| < 4.5,
    and asymptotic expansion for |x| >= 4.5.
    """
    ax = abs(x)
    sgn = 1.0 if x >= 0 else -1.0

    if ax < 2.5:
        # Series expansion
        term = ax
        sum_val = ax
        n = 1
        x2 = (math.pi / 2.0) * ax * ax
        while abs(term) > 1e-15 and n < 200:
            term *= -x2 * x2 / ((4 * n - 1) * (4 * n))
            sum_val += term
            n += 1
        return sgn * sum_val

    if ax < 4.5:
        # Backward recurrence (rational approximation style)
        # Use auxiliary functions f and g
        t = (math.pi / 2.0) * ax * ax
        f = 1.0 - 3.0 / t**2 + 105.0 / t**4 - 10395.0 / t**6
        g = 1.0 / t - 15.0 / t**3 + 945.0 / t**5 - 135135.0 / t**7
        c = 0.5 + (f * math.sin(t) - g * math.cos(t)) / (math.pi * ax)
        return sgn * c

    # Asymptotic expansion for large x
    t = (math.pi / 2.0) * ax * ax
    f = 1.0 - 3.0 / t**2 + 105.0 / t**4 - 10395.0 / t**6
    g = 1.0 / t - 15.0 / t**3 + 945.0 / t**5 - 135135.0 / t**7
    c = 0.5 + (f * math.sin(t) - g * math.cos(t)) / (math.pi * ax)
    return sgn * c


def fresnel_sin(x: float) -> float:
    """
    Compute the Fresnel sine integral S(x) = integral_0^x sin(pi*t^2/2) dt.
    """
    ax = abs(x)
    sgn = 1.0 if x >= 0 else -1.0

    if ax < 2.5:
        term = (math.pi / 6.0) * ax * ax * ax
        sum_val = term
        n = 1
        x2 = (math.pi / 2.0) * ax * ax
        while abs(term) > 1e-15 and n < 200:
            term *= -x2 * x2 / ((4 * n + 1) * (4 * n + 2))
            sum_val += term
            n += 1
        return sgn * sum_val

    if ax < 4.5:
        t = (math.pi / 2.0) * ax * ax
        f = 1.0 - 3.0 / t**2 + 105.0 / t**4 - 10395.0 / t**6
        g = 1.0 / t - 15.0 / t**3 + 945.0 / t**5 - 135135.0 / t**7
        s = 0.5 - (f * math.cos(t) + g * math.sin(t)) / (math.pi * ax)
        return sgn * s

    t = (math.pi / 2.0) * ax * ax
    f = 1.0 - 3.0 / t**2 + 105.0 / t**4 - 10395.0 / t**6
    g = 1.0 / t - 15.0 / t**3 + 945.0 / t**5 - 135135.0 / t**7
    s = 0.5 - (f * math.cos(t) + g * math.sin(t)) / (math.pi * ax)
    return sgn * s


# ---------------------------------------------------------------------------
# Newton iteration for scalar non-linear equations (adapted from 808_nonlin_newton)
# ---------------------------------------------------------------------------
def newton_solve(
    f_func,
    fp_func,
    x0: float,
    tol: float = 1e-12,
    max_iter: int = 100,
    f_diverge: float = 1e12,
) -> tuple:
    """
    Solve f(x) = 0 using Newton's method.

    Parameters
    ----------
    f_func : callable
        Scalar function f(x).
    fp_func : callable
        Derivative f'(x).
    x0 : float
        Initial guess.
    tol : float
        Tolerance on |f(x)| for convergence.
    max_iter : int
        Maximum iterations.
    f_diverge : float
        If |f(x)| > f_diverge, abort (divergence guard).

    Returns
    -------
    (root, f_root, iterations, converged)
    """
    x = float(x0)
    for it in range(1, max_iter + 1):
        fx = f_func(x)
        if abs(fx) < tol:
            return x, fx, it, True
        if abs(fx) > f_diverge:
            return x, fx, it, False
        fpx = fp_func(x)
        if abs(fpx) < 1e-14:
            return x, fx, it, False
        x_new = x - fx / fpx
        # Damping if step is too large
        if abs(x_new - x) > 10.0:
            x_new = x - 0.1 * fx / fpx
        x = x_new
    fx = f_func(x)
    converged = abs(fx) < tol
    return x, fx, max_iter, converged


# ---------------------------------------------------------------------------
# Jacobi iterative solver (adapted from 603_jacobi)
# ---------------------------------------------------------------------------
def jacobi_solve(
    A: np.ndarray,
    b: np.ndarray,
    x0: np.ndarray = None,
    max_iter: int = 5000,
    tol: float = 1e-10,
) -> tuple:
    """
    Solve Ax = b via Jacobi iteration.

    Parameters
    ----------
    A : (N, N) ndarray
        Coefficient matrix.
    b : (N,) ndarray
        Right-hand side.
    x0 : (N,) ndarray, optional
        Initial guess.
    max_iter : int
    tol : float
        Stopping tolerance on the residual 2-norm.

    Returns
    -------
    (x, residual_norm, iterations, converged)
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    n = A.shape[0]
    if A.shape != (n, n):
        raise ValueError("A must be square.")
    if b.shape != (n,):
        raise ValueError("b must have length N.")

    diag = np.diag(A)
    if np.any(np.abs(diag) < 1e-14):
        raise ValueError("Zero diagonal entries detected; Jacobi cannot proceed.")

    if x0 is None:
        x = np.zeros(n, dtype=float)
    else:
        x = np.asarray(x0, dtype=float).copy()

    for it in range(1, max_iter + 1):
        # Vectorized Jacobi update: x_new = (b - (A*x) + diag(A)*x) / diag(A)
        x_new = (b - A @ x + diag * x) / diag
        res = np.linalg.norm(A @ x_new - b)
        if res < tol:
            return x_new, res, it, True
        x = x_new
    res = np.linalg.norm(A @ x - b)
    return x, res, max_iter, res < tol


# ---------------------------------------------------------------------------
# Sparse matrix utilities (adapted from 783_msm_to_st)
# ---------------------------------------------------------------------------
def matrix_to_st(A: np.ndarray) -> tuple:
    """
    Convert a dense matrix to sparse triplet (ST) format.

    Returns
    -------
    rows, cols, vals : three 1-D ndarrays of non-zero entries.
    """
    A = np.asarray(A)
    rows, cols = np.nonzero(A)
    vals = A[rows, cols]
    return rows.astype(int), cols.astype(int), vals


def st_to_dense(rows: np.ndarray, cols: np.ndarray, vals: np.ndarray, shape: tuple, dtype=float) -> np.ndarray:
    """Reconstruct a dense matrix from ST format."""
    A = np.zeros(shape, dtype=dtype)
    for r, c, v in zip(rows, cols, vals):
        A[r, c] += v
    return A


# ---------------------------------------------------------------------------
# Linear least-squares fit (adapted from 692_llsq)
# ---------------------------------------------------------------------------
def llsq_fit(x: np.ndarray, y: np.ndarray) -> tuple:
    """
    Fit y = a*x + b via linear least squares.

    Returns
    -------
    (a, b, residual_norm)
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size != y.size or x.size < 2:
        raise ValueError("x and y must have equal length >= 2.")
    xm = np.mean(x)
    ym = np.mean(y)
    ss_xy = np.sum((x - xm) * (y - ym))
    ss_xx = np.sum((x - xm) ** 2)
    if abs(ss_xx) < 1e-15:
        a = 0.0
        b = ym
    else:
        a = ss_xy / ss_xx
        b = ym - a * xm
    residual = np.linalg.norm(y - (a * x + b))
    return a, b, residual


def llsq_fit_through_origin(x: np.ndarray, y: np.ndarray) -> tuple:
    """
    Fit y = a*x (through origin) via linear least squares.

    Returns
    -------
    (a, residual_norm)
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size != y.size or x.size < 1:
        raise ValueError("x and y must have equal length >= 1.")
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size != y.size or x.size < 1:
        raise ValueError("x and y must have equal length >= 1.")
    scale = np.max(np.abs(x))
    if scale < np.finfo(float).tiny:
        a = 0.0
    else:
        # Normalize x to avoid scale-dependent threshold issues
        xn = x / scale
        ss_xx_n = np.sum(xn * xn)
        a_n = np.sum(xn * y) / ss_xx_n
        a = a_n / scale
    residual = np.linalg.norm(y - a * x)
    return a, residual


# ---------------------------------------------------------------------------
# Gaussian quadrature weights & nodes for [-1, 1]
# ---------------------------------------------------------------------------
def gauss_legendre_1d(order: int) -> tuple:
    """
    Return weights and nodes for Gauss-Legendre quadrature on [-1, 1].
    Supports orders 1 through 5 with tabulated values.
    """
    order = int(order)
    if order < 1 or order > 5:
        raise ValueError("Gauss-Legendre order must be in [1, 5].")

    tables = {
        1: {
            "x": np.array([0.0]),
            "w": np.array([2.0]),
        },
        2: {
            "x": np.array([-1.0 / math.sqrt(3.0), 1.0 / math.sqrt(3.0)]),
            "w": np.array([1.0, 1.0]),
        },
        3: {
            "x": np.array([0.0, -math.sqrt(3.0 / 5.0), math.sqrt(3.0 / 5.0)]),
            "w": np.array([8.0 / 9.0, 5.0 / 9.0, 5.0 / 9.0]),
        },
        4: {
            "x": np.array([
                -math.sqrt(3.0 / 7.0 - 2.0 / 7.0 * math.sqrt(6.0 / 5.0)),
                math.sqrt(3.0 / 7.0 - 2.0 / 7.0 * math.sqrt(6.0 / 5.0)),
                -math.sqrt(3.0 / 7.0 + 2.0 / 7.0 * math.sqrt(6.0 / 5.0)),
                math.sqrt(3.0 / 7.0 + 2.0 / 7.0 * math.sqrt(6.0 / 5.0)),
            ]),
            "w": np.array([
                (18.0 + math.sqrt(30.0)) / 36.0,
                (18.0 + math.sqrt(30.0)) / 36.0,
                (18.0 - math.sqrt(30.0)) / 36.0,
                (18.0 - math.sqrt(30.0)) / 36.0,
            ]),
        },
        5: {
            "x": np.array([
                0.0,
                -1.0 / 3.0 * math.sqrt(5.0 - 2.0 * math.sqrt(10.0 / 7.0)),
                1.0 / 3.0 * math.sqrt(5.0 - 2.0 * math.sqrt(10.0 / 7.0)),
                -1.0 / 3.0 * math.sqrt(5.0 + 2.0 * math.sqrt(10.0 / 7.0)),
                1.0 / 3.0 * math.sqrt(5.0 + 2.0 * math.sqrt(10.0 / 7.0)),
            ]),
            "w": np.array([
                128.0 / 225.0,
                (322.0 + 13.0 * math.sqrt(70.0)) / 900.0,
                (322.0 + 13.0 * math.sqrt(70.0)) / 900.0,
                (322.0 - 13.0 * math.sqrt(70.0)) / 900.0,
                (322.0 - 13.0 * math.sqrt(70.0)) / 900.0,
            ]),
        },
    }
    t = tables[order]
    return t["w"].copy(), t["x"].copy()
