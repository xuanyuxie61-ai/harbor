
import numpy as np
from utils import check_finite


def legendre_polynomial_values(m, n, x):
    x = np.asarray(x).reshape(-1)
    if np.min(x) < -1.0 - 1e-12 or np.max(x) > 1.0 + 1e-12:
        raise ValueError("legendre_polynomial_values: x must be in [-1, 1]")
    v = np.zeros((m, n + 1))
    v[:, 0] = 1.0
    if n >= 1:
        v[:, 1] = x
    for j in range(2, n + 1):
        v[:, j] = ((2.0 * j - 1.0) * x * v[:, j - 1] -
                   (j - 1.0) * v[:, j - 2]) / j
    check_finite(v, "legendre_polynomial_values")
    return v


def legendre_polynomial_derivative(m, n, x):
    x = np.asarray(x).reshape(-1)
    v = legendre_polynomial_values(m, n, x)
    dp = np.zeros((m, n + 1))
    if n >= 1:
        dp[:, 1] = 1.0
    for j in range(2, n + 1):
        dp[:, j] = ((2.0 * j - 1.0) * (v[:, j - 1] + x * dp[:, j - 1]) -
                    (j - 1.0) * dp[:, j - 2]) / j
    check_finite(dp, "legendre_polynomial_derivative")
    return dp


def hermite_probabilist_coefficients(n):
    if n < 0:
        raise ValueError("hermite_probabilist_coefficients: n >= 0 required")

    ct = np.zeros((n + 1, n + 1))
    ct[0, 0] = 1.0
    if n >= 1:
        ct[1, 1] = 1.0
    for i in range(2, n + 1):
        ct[i, 0] = -(i - 1) * ct[i - 2, 0]
        for k in range(1, i + 1):
            ct[i, k] = ct[i - 1, k - 1] - (i - 1) * ct[i - 2, k]
    c = ct[n, :n + 1]
    return c


def hermite_probabilist_value(n, x):
    x = np.asarray(x)
    if n == 0:
        return np.ones_like(x)
    if n == 1:
        return x.copy()
    h_prev2 = np.ones_like(x)
    h_prev1 = x.copy()
    for i in range(2, n + 1):
        h_curr = x * h_prev1 - (i - 1) * h_prev2
        h_prev2 = h_prev1
        h_prev1 = h_curr
    return h_prev1


def hermite_probabilist_values_array(max_n, x):
    x = np.asarray(x).reshape(-1)
    m = len(x)
    v = np.zeros((m, max_n + 1))
    v[:, 0] = 1.0
    if max_n >= 1:
        v[:, 1] = x
    for i in range(2, max_n + 1):
        v[:, i] = x * v[:, i - 1] - (i - 1) * v[:, i - 2]
    return v


def mixed_legendre_hermite_basis_2d(x, y, n_leg, n_herm):
    x = np.asarray(x).reshape(-1)
    y = np.asarray(y).reshape(-1)
    if len(x) != len(y):
        raise ValueError("mixed_legendre_hermite_basis_2d: x and y must have same length")
    m = len(x)
    P = legendre_polynomial_values(m, n_leg, x)

    y_std = y / (np.std(y) + 1e-10)
    H = hermite_probabilist_values_array(n_herm, y_std)
    B = np.zeros((m, (n_leg + 1) * (n_herm + 1)))
    idx = 0
    for i in range(n_leg + 1):
        for j in range(n_herm + 1):
            B[:, idx] = P[:, i] * H[:, j]
            idx += 1
    return B


def gauss_legendre_quadrature_weights(n):
    x, w = np.polynomial.legendre.leggauss(n)
    return x, w
