"""
spectral_approx.py
==================
Spectral approximation toolkit for swarm robotics control kernels.

Incorporates:
  - chebyshev_coefficients / chebyshev_interpolant (from 159_chebyshev)
  - bernstein_poly_ab / bernstein_poly_ab_approx (from 077_bernstein_approximation)

Scientific role:
  Chebyshev and Bernstein spectral bases are used to parameterize
  communication-delay kernels and local control policies for each robot.
  A control policy u(d) is expanded as a finite Bernstein series
  over the sensing radius [0, R_s], ensuring positivity and
  partition-of-unity constraints required for consensus stability.
"""

import numpy as np
from functools import lru_cache


def chebyshev_coefficients(a: float, b: float, n: int, f):
    """
    Compute Chebyshev interpolation coefficients for f on [a, b].

    For the n Chebyshev nodes
        x_k = 0.5*(a+b) + 0.5*(b-a)*cos(pi*(2k-1)/(2n)),  k=1..n
    the coefficients are
        c_j = (2/n) * sum_{k=1}^{n} f(x_k) * T_{j-1}(x_k)
    with T_m(z) = cos(m*arccos(z)).

    Parameters
    ----------
    a, b : float
        Interval endpoints (a < b).
    n : int
        Order of the interpolant (number of nodes).
    f : callable
        Function to approximate.

    Returns
    -------
    c : ndarray, shape (n,)
        Chebyshev coefficients.
    """
    if a >= b:
        raise ValueError("Require a < b for Chebyshev interval.")
    if n <= 0:
        raise ValueError("Require n > 0.")

    angle = (2.0 * np.arange(1, n + 1) - 1.0) * np.pi / (2.0 * n)
    x = np.cos(angle)
    x_phys = 0.5 * (a + b) + 0.5 * (b - a) * x
    fx = np.asarray(f(x_phys), dtype=float)

    c = np.zeros(n, dtype=float)
    for j in range(n):
        c[j] = np.sum(fx * np.cos((j) * angle))
    c *= 2.0 / n
    return c


def chebyshev_interpolant(a: float, b: float, n: int, c: np.ndarray, x: np.ndarray):
    """
    Evaluate Chebyshev interpolant via Clenshaw recurrence.

    For transformed variable y = (2x - a - b)/(b - a), the recurrence is
        d_{n+1} = d_n = 0
        d_k = 2*y*d_{k+1} - d_{k+2} + c_{k+1}   for k = n-1 .. 0
        C(f)(x) = y*d_0 - d_1 + 0.5*c_0

    Parameters
    ----------
    a, b : float
        Interval endpoints.
    n : int
        Order.
    c : ndarray, shape (n,)
        Chebyshev coefficients.
    x : ndarray
        Evaluation points.

    Returns
    -------
    value : ndarray
        Interpolant values.
    """
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return np.array([], dtype=float)
    if a >= b:
        raise ValueError("Require a < b.")
    if c.shape[0] != n:
        raise ValueError("Coefficient length mismatch.")

    y = (2.0 * x - a - b) / (b - a)
    y = np.clip(y, -1.0, 1.0)  # numerical robustness

    d_ip1 = np.zeros_like(y)
    d_i = np.zeros_like(y)
    for i in range(n - 1, 0, -1):
        d_ip2 = d_ip1
        d_ip1 = d_i
        d_i = 2.0 * y * d_ip1 - d_ip2 + c[i]

    value = y * d_i - d_ip1 + 0.5 * c[0]
    return value


def bernstein_poly_ab(n: int, a: float, b: float, x: float):
    """
    Evaluate all n+1 Bernstein basis polynomials of degree n on [a,b] at x.

    B_{i,n}(x) = C(n,i) * (x-a)^i * (b-x)^{n-i} / (b-a)^n

    Parameters
    ----------
    n : int
        Degree (>= 0).
    a, b : float
        Interval endpoints.
    x : float
        Evaluation point.

    Returns
    -------
    bern : ndarray, shape (n+1,)
        Basis values.
    """
    if abs(b - a) < 1e-14:
        raise ValueError("bernstein_poly_ab: a and b must differ.")
    if n < 0:
        raise ValueError("bernstein_poly_ab: n must be non-negative.")

    bern = np.zeros(n + 1, dtype=float)
    if n == 0:
        bern[0] = 1.0
        return bern

    bern[0] = (b - x) / (b - a)
    bern[1] = (x - a) / (b - a)

    for i in range(2, n + 1):
        bern[i] = (x - a) * bern[i - 1] / (b - a)
        for j in range(i - 1, 0, -1):
            bern[j] = ((b - x) * bern[j] + (x - a) * bern[j - 1]) / (b - a)
        bern[0] = (b - x) * bern[0] / (b - a)
    return bern


def bernstein_poly_ab_approx(n: int, a: float, b: float, ydata: np.ndarray, xval: np.ndarray):
    """
    Bernstein polynomial approximant to data on [a,b].

    BPAB(f)(x) = sum_{i=0}^{n} ydata_i * B_{i,n}(x)

    Parameters
    ----------
    n : int
        Degree.
    a, b : float
        Interval.
    ydata : ndarray, shape (n+1,)
        Data values at uniformly spaced nodes in [a,b].
    xval : ndarray
        Evaluation points.

    Returns
    -------
    yval : ndarray
        Approximant values.
    """
    ydata = np.asarray(ydata, dtype=float)
    xval = np.asarray(xval, dtype=float)
    if ydata.shape[0] != n + 1:
        raise ValueError("ydata length must be n+1.")
    yval = np.zeros_like(xval, dtype=float)
    for i in range(xval.size):
        bvec = bernstein_poly_ab(n, a, b, xval.flat[i])
        yval.flat[i] = np.dot(ydata, bvec)
    return yval


def delay_kernel_chebyshev(tau_max: float, n: int, kernel_type: str = "exponential"):
    """
    Compute a Chebyshev-spectral delay kernel for communication delay.

    Models the probability density of delay tau in [0, tau_max] as
        K(tau) = exp(-lambda*tau) / (1 - exp(-lambda*tau_max))
    and returns its Chebyshev coefficients for fast evaluation.

    Parameters
    ----------
    tau_max : float
        Maximum delay.
    n : int
        Chebyshev order.
    kernel_type : str
        "exponential" or "gaussian".

    Returns
    -------
    c : ndarray
        Chebyshev coefficients.
    """
    if tau_max <= 0:
        raise ValueError("tau_max must be positive.")

    lam = 5.0 / tau_max

    if kernel_type == "exponential":
        def f(t):
            return np.exp(-lam * t) / (1.0 - np.exp(-lam * tau_max))
    elif kernel_type == "gaussian":
        sigma = tau_max / 3.0

        def f(t):
            return np.exp(-0.5 * (t / sigma) ** 2) / (sigma * np.sqrt(2.0 * np.pi))
    else:
        raise ValueError("Unknown kernel_type.")

    c = chebyshev_coefficients(0.0, tau_max, n, f)
    return c
