
import numpy as np
from typing import Tuple






def _besselj1_squared_approx(z: float) -> float:
    if z < 1e-8:
        return 0.0
    inv_z = 1.0 / z

    amp = 2.0 / (np.pi * z)
    phase = z - 0.75 * np.pi
    return amp * (np.cos(phase) ** 2)


def _besseljzero(k: int) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    if k <= 20:

        tab = np.array([
            2.404825558, 5.520078110, 8.653727913, 11.79153444, 14.93091771,
            18.07106397, 21.21163663, 24.35247153, 27.49347913, 30.63460647,
            33.77582021, 36.91709835, 40.05842576, 43.19979171, 46.34118837,
            49.48260990, 52.62405184, 55.76551076, 58.90698393, 62.04846919
        ], dtype=float)
        if k <= len(tab):
            return tab[k - 1]

    beta = (k - 0.25) * np.pi
    mu = 4.0 * (0.0 ** 2)
    inv_beta = 1.0 / beta
    j0 = beta + 0.125 * inv_beta - 0.03125 * inv_beta ** 3 + 0.0537109375 * inv_beta ** 5
    return j0


def _glpair_asymptotic(n: int, k: int) -> Tuple[float, float]:
    if n <= 100:


        t = np.cos(np.pi * (k - 0.25) / (n + 0.5))
        for _ in range(10):
            p0 = 1.0
            p1 = t
            for j in range(2, n + 1):
                p2 = ((2 * j - 1) * t * p1 - (j - 1) * p0) / j
                p0, p1 = p1, p2
            dp = n * (t * p1 - p0) / (t * t - 1.0)
            if abs(dp) < 1e-15:
                break
            t -= p1 / dp

        w = 2.0 / ((1.0 - t * t) * dp * dp)
        return t, w

    jk = _besseljzero(k)
    theta = jk / (n + 0.5)
    t = np.cos(theta)

    j1sq = _besselj1_squared_approx(jk)
    if j1sq < 1e-18:
        w = 1.0
    else:
        w = 2.0 / ((n + 0.5) ** 2 * j1sq)
    return t, w


def gauss_legendre_nodes_weights(n: int) -> Tuple[np.ndarray, np.ndarray]:
    if n < 1:
        raise ValueError("n must be >= 1")
    nodes = np.zeros(n, dtype=float)
    weights = np.zeros(n, dtype=float)
    m = (n + 1) // 2
    for k in range(1, m + 1):
        xk, wk = _glpair_asymptotic(n, k)
        nodes[k - 1] = xk
        weights[k - 1] = wk
        nodes[n - k] = -xk
        weights[n - k] = wk
    if n % 2 == 1:
        nodes[m - 1] = 0.0

        t = 0.0
        p0, p1 = 1.0, t
        for j in range(2, n + 1):
            p2 = ((2 * j - 1) * t * p1 - (j - 1) * p0) / j
            p0, p1 = p1, p2
        dp = n * (t * p1 - p0) / (t * t - 1.0 + 1e-18)
        weights[m - 1] = 2.0 / (dp * dp)
    return nodes, weights






def log_gamma(x: float) -> float:
    if x <= 0.0:
        return np.nan

    y = x
    corr = 0.0
    while y < 7.0:
        corr -= np.log(y)
        y += 1.0
    inv_y = 1.0 / y
    inv_y2 = inv_y * inv_y


    p = inv_y * (1.0 / 12.0 - inv_y2 * (1.0 / 360.0 - inv_y2 * (
        1.0 / 1260.0 - inv_y2 * (1.0 / 1680.0 - inv_y2 / 1188.0))))
    val = (y - 0.5) * np.log(y) - y + 0.918938533204673 + p + corr
    return val


def incomplete_beta_ratio(x: float, p: float, q: float) -> float:
    if x < 0.0 or x > 1.0 or p <= 0.0 or q <= 0.0:
        return np.nan
    if x == 0.0:
        return 0.0
    if x == 1.0:
        return 1.0


    if x > 0.5:
        return 1.0 - incomplete_beta_ratio(1.0 - x, q, p)

    log_beta = log_gamma(p) + log_gamma(q) - log_gamma(p + q)
    front = np.exp(np.log(x) * p + np.log(1.0 - x) * q - log_beta) / p


    term = 1.0
    sum_val = 1.0
    psq = p + q
    cx = 1.0 - x
    i = 0
    while True:
        i += 1
        ai = i
        term *= (psq + ai - 1.0) * x / (p + ai)
        sum_val += term
        if abs(term) < 1e-14 * abs(sum_val):
            break
        if i > 10000:
            break

    return front * sum_val






def triangle_unit_rule(order: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if order == 1:
        xi = np.array([1.0 / 3.0])
        eta = np.array([1.0 / 3.0])
        w = np.array([0.5])
    elif order == 3:
        xi = np.array([1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0])
        eta = np.array([1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0])
        w = np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
    elif order == 7:
        a = 0.059715871789770
        b = 0.797426985353087
        c = 0.333333333333333
        P1 = 0.1125
        P2 = 0.066197076394253
        P3 = 0.062969590272413
        xi = np.array([a, b, a, c, c, c, c])
        eta = np.array([a, a, b, c, a, b, c])
        w = np.array([P3, P3, P3, P1, P2, P2, P2])
    else:

        xi = np.array([1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0])
        eta = np.array([1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0])
        w = np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
    return xi, eta, w
