
import numpy as np
import math
from scipy.special import gamma as Gamma
from typing import Tuple, Optional
from utils import check_bounds, EPSILON_MACHINE


def laguerre_polynomial(m: int, n: int, x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float).flatten()
    if n < 0:
        return np.empty((m, 0))

    v = np.zeros((m, n + 1), dtype=float)
    v[:, 0] = 1.0
    if n == 0:
        return v

    v[:, 1] = 1.0 - x
    for j in range(2, n + 1):
        v[:, j] = (
            ((2.0 * j - 1.0) - x) * v[:, j - 1]
            - (j - 1.0) * v[:, j - 2]
        ) / j

    return v


def generalized_laguerre_function(m: int, n: int, alpha: float,
                                   x: np.ndarray) -> np.ndarray:
    if alpha <= -1.0:
        raise ValueError(f"alpha must be > -1, got {alpha}")
    x = np.asarray(x, dtype=float).flatten()
    if n < 0:
        return np.empty((m, 0))

    v = np.zeros((m, n + 1), dtype=float)
    v[:, 0] = 1.0
    if n == 0:
        return v

    v[:, 1] = 1.0 + alpha - x
    for i in range(2, n + 1):
        v[:, i] = (
            ((2.0 * i - 1.0 + alpha) - x) * v[:, i - 1]
            + (-i + 1.0 - alpha) * v[:, i - 2]
        ) / i

    return v


def chebyshev_nodes(a: float, b: float, n: int) -> np.ndarray:
    if n <= 0:
        return np.array([])
    if n == 1:
        return np.array([(a + b) / 2.0])

    k = np.arange(1, n + 1, dtype=float)
    theta = (2.0 * k - 1.0) * np.pi / (2.0 * n)
    c = np.cos(theta)


    if n % 2 == 1:
        mid = (n + 1) // 2
        c[mid - 1] = 0.0

    x = 0.5 * ((1.0 - c) * a + (1.0 + c) * b)
    return x


def divided_differences(xd: np.ndarray, yd: np.ndarray) -> np.ndarray:
    xd = np.asarray(xd, dtype=float)
    yd = np.asarray(yd, dtype=float)
    n = len(xd)
    d = yd.copy()
    for i in range(1, n):
        for j in range(n - 1, i - 1, -1):
            denom = xd[j] - xd[j - i]
            if abs(denom) < EPSILON_MACHINE * 100:
                d[j] = 0.0
            else:
                d[j] = (d[j] - d[j - 1]) / denom
    return d


def newton_interpolate(xd: np.ndarray, dd: np.ndarray, xp: np.ndarray) -> np.ndarray:
    xd = np.asarray(xd, dtype=float)
    dd = np.asarray(dd, dtype=float)
    xp = np.asarray(xp, dtype=float)
    nd = len(xd)
    yp = dd[nd - 1] * np.ones_like(xp)
    for i in range(nd - 2, -1, -1):
        yp = dd[i] + (xp - xd[i]) * yp
    return yp


def chebyshev_interpolate(func: callable, a: float, b: float,
                          n: int, xp: np.ndarray) -> Tuple[np.ndarray, float]:
    xd = chebyshev_nodes(a, b, n)
    yd = func(xd)
    dd = divided_differences(xd, yd)
    yp = newton_interpolate(xd, dd, xp)


    ne = 10001
    xe = np.linspace(a, b, ne)
    ye = newton_interpolate(xd, dd, xe)
    fe = func(xe)
    maxerr = np.max(np.abs(ye - fe))
    return yp, maxerr


def radial_distribution_spectrum(r: np.ndarray, g: np.ndarray,
                                  n_modes: int = 10, alpha: float = 0.0,
                                  beta: float = 1.0) -> np.ndarray:
    r = np.asarray(r, dtype=float)
    g = np.asarray(g, dtype=float)
    m = len(r)


    L = generalized_laguerre_function(m, n_modes - 1, alpha, beta * r)


    coeffs = np.zeros(n_modes)

    w = np.exp(-beta * r) * np.maximum(beta * r, 0.0) ** alpha

    for n in range(n_modes):
        integrand = g * L[:, n] * w

        if len(r) > 1:
            coeffs[n] = np.trapezoid(integrand, r)
        else:
            coeffs[n] = integrand[0] * r[0] if len(r) == 1 else 0.0


    for n in range(n_modes):
        norm = Gamma(n + alpha + 1.0) / math.factorial(n)
        if norm > 0:
            coeffs[n] /= norm

    return coeffs


def chebyshev_spectral_derivative(u: np.ndarray, L: float) -> np.ndarray:
    u = np.asarray(u, dtype=float)
    n = len(u)
    if n < 2:
        return np.zeros_like(u)
    h = L / (n - 1)
    dudx = np.zeros_like(u)

    dudx[1:-1] = (u[2:] - u[:-2]) / (2.0 * h)

    dudx[0] = (-3.0 * u[0] + 4.0 * u[1] - u[2]) / (2.0 * h)
    dudx[-1] = (3.0 * u[-1] - 4.0 * u[-2] + u[-3]) / (2.0 * h)
    return dudx
