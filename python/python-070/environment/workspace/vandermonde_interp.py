
import numpy as np
from utils import NumericalConfig


def pvand(n, alpha, b):
    alpha = np.asarray(alpha, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(alpha) != n or len(b) != n:
        raise ValueError("Dimension mismatch in pvand")


    for i in range(n):
        for j in range(i + 1, n):
            if abs(alpha[i] - alpha[j]) < NumericalConfig.TOL:
                raise ValueError(f"Vandermonde nodes must be distinct: alpha[{i}]={alpha[i]}, alpha[{j}]={alpha[j]}")

    x = b.copy()


    for k in range(n - 1):
        for j in range(n - 1, k, -1):
            x[j] = x[j] - alpha[k] * x[j - 1]


    for k in range(n - 2, -1, -1):
        for j in range(k + 1, n):
            denom = alpha[j] - alpha[j - k - 1]
            if abs(denom) < NumericalConfig.EPS:
                denom = NumericalConfig.EPS
            x[j] = x[j] / denom
        for j in range(k, n - 1):
            x[j] = x[j] - x[j + 1]

    return x


def dvand(n, alpha, b):
    alpha = np.asarray(alpha, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(alpha) != n or len(b) != n:
        raise ValueError("Dimension mismatch in dvand")

    x = b.copy()


    for k in range(n - 1):
        for j in range(n - 1, k, -1):
            denom = alpha[j] - alpha[j - k - 1]
            if abs(denom) < NumericalConfig.EPS:
                denom = NumericalConfig.EPS
            x[j] = x[j] / denom
        for j in range(k, n - 1):
            x[j] = x[j] - x[j + 1]


    for k in range(n - 2, -1, -1):
        for j in range(k + 1, n):
            x[j] = x[j] - alpha[k] * x[j - 1]

    return x


def bidim_vandermonde_solve(n, alpha, beta, b):
    alpha = np.asarray(alpha, dtype=float)
    beta = np.asarray(beta, dtype=float)
    b = np.asarray(b, dtype=float)


    temp = np.zeros((n, n), dtype=float)
    for i in range(n):
        rhs = b[i * n:(i + 1) * n]
        temp[i, :] = pvand(n, alpha, rhs)


    x = np.zeros(n * n, dtype=float)
    for j in range(n):
        rhs = temp[:, j]
        sol = pvand(n, beta, rhs)
        x[j * n:(j + 1) * n] = sol

    return x


def vandermonde_interp_1d(alpha, y, x_eval):
    alpha = np.asarray(alpha, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(alpha)


    V = np.vander(alpha, N=n, increasing=True)
    c = np.linalg.solve(V, y)

    x_eval = np.asarray(x_eval, dtype=float)
    values = np.zeros_like(x_eval, dtype=float)
    for j in range(n):
        values += c[j] * (x_eval ** j)
    return values
