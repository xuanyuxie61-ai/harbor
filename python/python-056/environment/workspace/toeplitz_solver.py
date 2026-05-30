
import numpy as np
from typing import Tuple


def r8sto_sl(n: int, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float).flatten()
    b = np.asarray(b, dtype=float).flatten()
    if a.size < n or b.size < n:
        raise ValueError("r8sto_sl: 输入数组长度不足")

    x = np.zeros(n)
    y = np.zeros(n)

    beta = 1.0
    x[0] = b[0] / beta
    if n > 1:
        y[0] = -a[1] / beta

    for k in range(1, n):
        beta = (1.0 - y[k - 1] * y[k - 1]) * beta
        if abs(beta) < 1e-14:
            raise RuntimeError(f"r8sto_sl: 在步骤 k={k} 处 β 接近零，矩阵可能不正定")


        dot_ax = np.dot(a[1:k + 1], x[k - 1::-1])
        x[k] = (b[k] - dot_ax) / beta
        x[:k] = x[:k] + x[k] * y[k - 1::-1]

        if k < n - 1:
            dot_ay = np.dot(a[1:k + 1], y[k - 1::-1])
            y[k] = (-a[k + 1] - dot_ay) / beta
            y[:k] = y[:k] + y[k] * y[k - 1::-1]

    return x


def r8sto_yw_sl(n: int, a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float).flatten()
    if a.size < n:
        raise ValueError("r8sto_yw_sl: 输入数组长度不足")

    y = np.zeros(n)
    beta = a[0]
    if abs(beta) < 1e-14:
        raise RuntimeError("r8sto_yw_sl: a[0] 为零")

    y[0] = -a[1] / beta
    for k in range(1, n - 1):
        beta = (1.0 - y[k - 1] * y[k - 1]) * beta
        dot_ay = np.dot(a[1:k + 1], y[k - 1::-1])
        y[k] = (-a[k + 1] - dot_ay) / beta
        y[:k] = y[:k] + y[k] * y[k - 1::-1]
    return y


def build_toeplitz_first_row(n: int, correlation_length: float = 5.0) -> np.ndarray:
    a = np.exp(-np.arange(n) / correlation_length)
    a[0] = 1.0
    return a


def solve_periodic_boundary_system(
    rhs: np.ndarray,
    correlation_length: float = 5.0,
) -> np.ndarray:
    n = len(rhs)
    a = build_toeplitz_first_row(n, correlation_length)
    return r8sto_sl(n, a, rhs)
