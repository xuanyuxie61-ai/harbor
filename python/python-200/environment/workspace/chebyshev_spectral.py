
import numpy as np
from typing import Callable, Tuple


def chebyshev_nodes(a: float, b: float, n: int) -> np.ndarray:
    if n < 1:
        return np.array([])
    if n == 1:
        return np.array([(a + b) / 2.0])
    k = np.arange(n)
    theta = np.pi * k / (n - 1)
    x = np.cos(theta)

    return 0.5 * (b - a) * x + 0.5 * (b + a)


def divided_differences(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    n = len(x)
    d = np.array(y, dtype=float)
    for j in range(1, n):
        for k in range(n - 1, j - 1, -1):
            denom = x[k] - x[k - j]
            if abs(denom) < 1e-30:
                denom = 1e-30 if denom >= 0 else -1e-30
            d[k] = (d[k] - d[k - 1]) / denom
    return d


def newton_interpolate(xd: np.ndarray, dd: np.ndarray,
                       xp: np.ndarray) -> np.ndarray:
    nd = len(dd)
    yp = dd[-1] * np.ones_like(xp, dtype=float)
    for i in range(nd - 2, -1, -1):
        yp = dd[i] + (xp - xd[i]) * yp
    return yp


def chebyshev_interpolate(func: Callable, a: float, b: float,
                          n: int, xp: np.ndarray) -> np.ndarray:
    xd = chebyshev_nodes(a, b, n)
    yd = func(xd)
    dd = divided_differences(xd, yd)
    return newton_interpolate(xd, dd, xp)


def chebyshev_differentiation_matrix(n: int) -> np.ndarray:
    if n < 2:
        return np.zeros((n, n))
    x = np.cos(np.pi * np.arange(n) / (n - 1))
    c = np.ones(n)
    c[0] = 2.0
    c[-1] = 2.0

    D = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            D[i, j] = (c[i] / c[j]) * ((-1) ** (i + j)) / (x[i] - x[j])

    D[0, 0] = (2.0 * (n - 1) ** 2 + 1.0) / 6.0
    D[-1, -1] = -D[0, 0]
    for i in range(1, n - 1):
        D[i, i] = -x[i] / (2.0 * (1.0 - x[i] ** 2))
    return D


def chebyshev_derivative(f_vals: np.ndarray) -> np.ndarray:
    n = len(f_vals)
    D = chebyshev_differentiation_matrix(n)
    return D @ f_vals


def chebyshev_spectral_solve_ode_bvp(coeff_func: Callable,
                                      rhs_func: Callable,
                                      n: int = 32,
                                      bc_left: Tuple[float, float] = (0.0, 0.0),
                                      bc_right: Tuple[float, float] = (0.0, 0.0)) -> Tuple[np.ndarray, np.ndarray]:
    x = np.cos(np.pi * np.arange(n) / (n - 1))
    D = chebyshev_differentiation_matrix(n)
    D2 = D @ D


    A = np.zeros((n, n))
    b = np.zeros(n)

    for i in range(n):
        a_i, b_i, c_i = coeff_func(x[i])
        A[i, :] = a_i * D2[i, :] + b_i * D[i, :] + c_i * np.eye(n)[i, :]
        b[i] = rhs_func(x[i])


    A[0, :] = 0.0
    A[0, 0] = 1.0
    b[0] = bc_left[0]

    A[-1, :] = 0.0
    A[-1, -1] = 1.0
    b[-1] = bc_right[0]


    if bc_left[1] is not None and n > 2:
        A[1, :] = D[0, :]
        b[1] = bc_left[1]
    if bc_right[1] is not None and n > 2:
        A[-2, :] = D[-1, :]
        b[-2] = bc_right[1]

    u = np.linalg.solve(A, b)
    return x, u


def chebyshev_clenshaw_eval(coeffs: np.ndarray, x: float) -> float:
    N = len(coeffs) - 1
    b2 = 0.0
    b1 = 0.0
    for k in range(N, -1, -1):
        b0 = 2.0 * x * b1 - b2 + coeffs[k]
        b2 = b1
        b1 = b0
    return 0.5 * (b0 - b2)
