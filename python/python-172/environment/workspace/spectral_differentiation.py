# -*- coding: utf-8 -*-
"""
Spectral Calculus on Chebyshev Series
======================================
Operations in coefficient space: differentiation, integration,
multiplication, and inversion of Chebyshev series.
Directly inspired by ACM TOMS Algorithm 446 (Broucke, 1973).

Key formulas:
- Derivative: if f(x) = sum a_k T_k(x), then
    f'(x) = sum b_k T_k(x), with
    b_{N-1} = 2 N a_N, b_N = 0,
    b_k = b_{k+2} + 2 (k+1) a_{k+1}  for k = N-2, ..., 0.
- Integral: if F(x) = integral f(x) dx = sum c_k T_k(x), then
    c_k = (a_{k-1} - a_{k+1}) / (2k)  for k >= 1,
    c_0 = 2 c_1 - 2 c_2 + ... + 2 (-1)^{N+1} c_N + 2 (-1)^N a_N / (2N)
    (c_0 is chosen so that F(-1) = 0).
- Multiplication: if f = sum a_k T_k, g = sum b_k T_k, then h = f*g
    h_m = 0.5 * sum_{j+k=m} a_j b_k + 0.5 * sum_{|j-k|=m} a_j b_k.
- Inversion: if y = f(x), seek x = f^{-1}(y) as Chebyshev series.
"""

import numpy as np


def chebyshev_derivative_series(coef):
    """
    Compute the Chebyshev series of the derivative.
    Implements the recurrence from TOMS 446 / dfrnt.m.

    Parameters
    ----------
    coef : ndarray
        Input Chebyshev coefficients.

    Returns
    -------
    dcoef : ndarray
        Chebyshev coefficients of the derivative.
    """
    coef = np.asarray(coef, dtype=np.float64)
    npl = len(coef)
    if npl <= 1:
        return np.zeros_like(coef)
    n = npl - 1
    dcoef = np.zeros(npl, dtype=np.float64)
    xxn = coef[n - 1]
    dcoef[n - 1] = 2.0 * coef[n] * n
    dcoef[n] = 0.0
    for k in range(3, npl + 1):
        l = npl - k
        xxl = coef[l]
        dcoef[l] = dcoef[l + 2] + 2.0 * xxn * (l + 1)
        xxn = xxl
    return dcoef


def chebyshev_integral_series(coef):
    """
    Compute the Chebyshev series of the indefinite integral with F(-1)=0.
    Inspired by TOMS 446 ntgrt.m.

    Parameters
    ----------
    coef : ndarray
        Input Chebyshev coefficients.

    Returns
    -------
    icoef : ndarray
        Chebyshev coefficients of the integral.
    """
    coef = np.asarray(coef, dtype=np.float64)
    npl = len(coef)
    if npl == 0:
        return np.array([0.0])
    n = npl - 1
    icoef = np.zeros(npl + 1, dtype=np.float64)
    # Coefficients for k >= 1
    icoef[1] = coef[0] - 0.5 * coef[1]
    for k in range(2, n):
        icoef[k] = (coef[k - 1] - coef[k + 1]) / (2.0 * k)
    icoef[n] = coef[n - 1] / (2.0 * n)
    icoef[n + 1] = coef[n] / (2.0 * (n + 1))
    # Determine c_0 from F(-1) = sum (-1)^k c_k = 0
    alt_sum = np.sum(icoef[1:] * np.power(-1.0, np.arange(1, npl + 1)))
    icoef[0] = -alt_sum
    return icoef


def chebyshev_series_multiply(a, b):
    """
    Multiply two Chebyshev series.
    Uses the product formula for Chebyshev polynomials:
      T_j(x) * T_k(x) = 0.5 * (T_{j+k}(x) + T_{|j-k|}(x)).

    Parameters
    ----------
    a, b : ndarray
        Chebyshev coefficients.

    Returns
    -------
    c : ndarray
        Chebyshev coefficients of the product, truncated to max(len(a),len(b)).
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    na, nb = len(a), len(b)
    nc = max(na, nb)
    c = np.zeros(nc, dtype=np.float64)
    for j in range(na):
        for k in range(nb):
            if j + k < nc:
                c[j + k] += 0.5 * a[j] * b[k]
            if abs(j - k) < nc:
                c[abs(j - k)] += 0.5 * a[j] * b[k]
    return c


def chebyshev_series_invert(coef, y_target, max_iter=50, tol=1e-12):
    """
    Approximate the inverse of a Chebyshev series via Newton iteration.
    Given y = p(x), find x such that p(x) = y_target.

    Parameters
    ----------
    coef : ndarray
        Chebyshev coefficients of p(x).
    y_target : float
        Target y value.
    max_iter : int
        Maximum Newton iterations.
    tol : float
        Convergence tolerance.

    Returns
    -------
    x_sol : float
        Approximate inverse value in [-1, 1].
    """
    from chebyshev_spectral import clenshaw_evaluate
    # Initial guess: linear inverse approximation
    x_sol = 0.0
    for _ in range(max_iter):
        y_val = clenshaw_evaluate(coef, x_sol)
        dcoef = chebyshev_derivative_series(coef)
        dy_val = clenshaw_evaluate(dcoef, x_sol)
        if abs(dy_val) < 1e-14:
            break
        dx = (y_val - y_target) / dy_val
        x_sol -= dx
        x_sol = np.clip(x_sol, -1.0, 1.0)
        if abs(dx) < tol:
            break
    return x_sol


def chebyshev_l2_norm(coef):
    """
    Compute the weighted L2 norm of a Chebyshev series:
        ||f||^2 = integral_{-1}^{1} f(x)^2 / sqrt(1-x^2) dx
                = (pi/2) * (2 a_0^2 + sum_{k=1}^{N-1} a_k^2).

    Parameters
    ----------
    coef : ndarray
        Chebyshev coefficients.

    Returns
    -------
    norm : float
        Weighted L2 norm.
    """
    coef = np.asarray(coef, dtype=np.float64)
    s = 2.0 * coef[0] * coef[0] + np.sum(coef[1:] ** 2)
    return np.sqrt(0.5 * np.pi * s)
