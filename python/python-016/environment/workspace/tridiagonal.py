
import numpy as np
from typing import Tuple


def tridiagonal_solve(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    d: np.ndarray,
    check_diagonal_dominance: bool = True,
) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    c = np.asarray(c, dtype=float)
    d = np.asarray(d, dtype=float)

    n = b.size
    if a.size != n or c.size != n or d.size != n:
        raise ValueError("All input arrays must have the same length.")
    if n == 0:
        return np.array([], dtype=float)

    if check_diagonal_dominance:
        for i in range(n):
            off_diag = 0.0
            if i > 0:
                off_diag += abs(a[i])
            if i < n - 1:
                off_diag += abs(c[i])
            if abs(b[i]) < off_diag:


                pass


    cp = np.zeros(n)
    dp = np.zeros(n)

    if abs(b[0]) < 1e-15:
        raise ValueError("Zero pivot at index 0.")
    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]

    for i in range(1, n):
        denom = b[i] - a[i] * cp[i - 1]
        if abs(denom) < 1e-15:
            raise ValueError(f"Zero pivot at index {i} during forward sweep.")
        if i < n - 1:
            cp[i] = c[i] / denom
        dp[i] = (d[i] - a[i] * dp[i - 1]) / denom


    x = np.zeros(n)
    x[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]

    return x


def tridiagonal_matvec(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    x: np.ndarray,
) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    c = np.asarray(c, dtype=float)
    x = np.asarray(x, dtype=float)
    n = b.size

    if a.size != n or c.size != n:
        raise ValueError("Diagonal arrays must have the same length.")
    if x.shape[0] != n:
        raise ValueError("First dimension of x must match matrix size.")

    y = np.zeros_like(x)
    y[0] = b[0] * x[0]
    if n > 1:
        y[0] += c[0] * x[1]
        y[-1] = a[-1] * x[-2] + b[-1] * x[-1]

    for i in range(1, n - 1):
        y[i] = a[i] * x[i - 1] + b[i] * x[i] + c[i] * x[i + 1]

    return y


def build_tridiagonal_from_1d_chain(
    onsite: np.ndarray,
    hopping: np.ndarray,
    periodic: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = onsite.size
    if hopping.size != n - 1:
        raise ValueError("hopping must have length n-1.")
    if periodic:
        raise ValueError("Periodic boundary conditions break tridiagonal structure.")

    a = np.zeros(n)
    b = onsite.copy()
    c = np.zeros(n)
    a[1:] = hopping
    c[:-1] = hopping

    return a, b, c


def solve_layer_potential_1d(
    layer_density: np.ndarray,
    interlayer_coupling: float,
    epsilon_screening: float,
) -> np.ndarray:
    n = layer_density.size
    if n < 2:
        raise ValueError("At least two layers required.")

    d = 0.335
    a = np.full(n, -epsilon_screening / (d ** 2))
    b = np.full(n, 2.0 * epsilon_screening / (d ** 2) + interlayer_coupling)
    c = np.full(n, -epsilon_screening / (d ** 2))

    a[0] = 0.0
    c[-1] = 0.0
    b[0] = epsilon_screening / (d ** 2) + interlayer_coupling
    b[-1] = epsilon_screening / (d ** 2) + interlayer_coupling

    return tridiagonal_solve(a, b, c, layer_density)
