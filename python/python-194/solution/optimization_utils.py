"""
optimization_utils.py
=====================
Unimodal optimization and line search for nonlinear iterations
and parameter optimization in domain-decomposition solvers.

Integrates concepts from:
  * test_unimodal (1-D unimodal test functions and minimization)

Mathematical background
-----------------------
A function f: [a,b] -> R is unimodal if it has exactly one local
minimum x* in [a,b], and f is strictly decreasing on [a, x*) and
strictly increasing on (x*, b].

Golden-section search:
    Given interval [a,b] and interior points c < d with
    b - d = c - a = phi * (b - a) where phi = (sqrt(5)-1)/2 ~ 0.618,
    compare f(c) and f(d):
        if f(c) < f(d): minimum lies in [a, d], set b = d
        else:            minimum lies in [c, b], set a = c
    Repeat until interval length < tol.

Convergence rate: linear with ratio phi ~ 0.618 per iteration,
so after k iterations the interval shrinks by phi^k.

Line search (Wolfe conditions):
    For descent direction p at point x, find alpha > 0 such that:
        Armijo:   f(x + alpha p) <= f(x) + c1 alpha grad_f(x)^T p
        Curvature: grad_f(x + alpha p)^T p >= c2 grad_f(x)^T p
    with 0 < c1 < c2 < 1 (typically c1=1e-4, c2=0.9).
"""

import numpy as np
from typing import Tuple, Callable, Optional


def golden_section_search(
    f: Callable[[float], float],
    a: float,
    b: float,
    tol: float = 1e-8,
    max_iter: int = 100
) -> Tuple[float, float]:
    """
    Golden-section search for minimum of a unimodal function on [a,b].

    Returns
    -------
    x_min : Approximate minimizer.
    f_min : Function value at x_min.
    """
    if a >= b:
        a, b = min(a, b), max(a, b)
        if a == b:
            return a, f(a)

    phi = (np.sqrt(5.0) - 1.0) / 2.0  # ~0.618
    resphi = 1.0 - phi  # ~0.382

    c = a + resphi * (b - a)
    d = a + phi * (b - a)
    fc = f(c)
    fd = f(d)

    for _ in range(max_iter):
        if abs(b - a) < tol:
            break
        if fc < fd:
            b = d
            d = c
            fd = fc
            c = a + resphi * (b - a)
            fc = f(c)
        else:
            a = c
            c = d
            fc = fd
            d = a + phi * (b - a)
            fd = f(d)

    x_min = 0.5 * (a + b)
    f_min = f(x_min)
    return x_min, f_min


def backtracking_line_search(
    f: Callable[[np.ndarray], float],
    grad_f: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    p: np.ndarray,
    alpha_init: float = 1.0,
    c1: float = 1e-4,
    rho: float = 0.5,
    max_iter: int = 20
) -> float:
    """
    Backtracking line search satisfying the Armijo condition.

    Parameters
    ----------
    f           : Objective function.
    grad_f      : Gradient of f.
    x           : Current point.
    p           : Descent direction.
    alpha_init  : Initial step length.
    c1          : Armijo parameter.
    rho         : Reduction factor (0 < rho < 1).
    max_iter    : Maximum backtracking steps.

    Returns
    -------
    alpha : Accepted step length.
    """
    fx = f(x)
    gp = float(np.dot(grad_f(x), p))
    if gp >= 0.0:
        # Not a descent direction
        return 0.0

    alpha = alpha_init
    for _ in range(max_iter):
        x_new = x + alpha * p
        fx_new = f(x_new)
        if fx_new <= fx + c1 * alpha * gp:
            return alpha
        alpha *= rho
        if alpha < 1e-15:
            return 0.0
    return alpha


def unimodal_test_suite():
    """
    Return a dictionary of test functions for algorithm validation.
    Each entry is (f, a, b, x_star).
    """
    tests = {}
    # p10: sharp high-order power
    tests["sharp_power"] = (
        lambda x: 12.0 + 1000.0 * abs(x - 2.8) ** 8.4,
        2.0, 3.5, 2.8
    )
    # p20: cubic polynomial
    tests["cubic"] = (
        lambda x: x ** 3 - 3.0 * x ** 2 - 5.0 * x + 8.0,
        -2.0, 5.0, (3.0 + np.sqrt(24.0)) / 3.0  # ~2.633
    )
    # p30: trigonometric
    tests["trig"] = (
        lambda x: 1.2 + x - 5.0 * np.sin(2.0 * x),
        -2.0, 5.0, None  # no closed form
    )
    return tests


def optimize_subdomain_overlap(
    residual_func: Callable[[float], float],
    overlap_range: Tuple[float, float] = (0.0, 0.5),
    tol: float = 1e-4
) -> float:
    """
    Optimize the overlap fraction for Schwarz domain decomposition
    by minimizing the residual as a function of overlap.
    """
    a, b = overlap_range
    if a < 0:
        a = 0.0
    if b > 1.0:
        b = 1.0
    x_min, f_min = golden_section_search(residual_func, a, b, tol=tol, max_iter=50)
    return x_min


def power_iteration_estimate(
    A_matvec: Callable[[np.ndarray], np.ndarray],
    n: int,
    max_iter: int = 30,
    tol: float = 1e-6
) -> float:
    """
    Estimate the largest eigenvalue (spectral radius) of a symmetric
    matrix via power iteration.  Used for choosing relaxation parameters.
    """
    v = np.random.randn(n)
    v = v / (np.linalg.norm(v) + 1e-15)
    lam_old = 0.0
    for _ in range(max_iter):
        Av = A_matvec(v)
        lam = float(np.dot(v, Av))
        v = Av / (np.linalg.norm(Av) + 1e-15)
        if abs(lam - lam_old) < tol * abs(lam + 1e-15):
            break
        lam_old = lam
    return abs(lam)
