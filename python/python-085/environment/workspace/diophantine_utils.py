import numpy as np
from typing import Tuple, List


def i4vec_gcd(a: np.ndarray) -> int:
    a = np.array(a, dtype=int)
    g = 0
    for val in a:
        g = np.gcd(g, int(val))
    return int(g)


def diophantine_nonnegative_solve(a: np.ndarray, b: int) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    a = np.array(a, dtype=int)
    if np.any(a < 0):
        raise ValueError("Coefficients must be nonnegative")
    if np.sum(a) <= 0:
        raise ValueError("At least one coefficient must be positive")
    if b < 0:
        raise ValueError("Right-hand side must be nonnegative")

    d = i4vec_gcd(a)
    if b % d != 0:
        raise ValueError(f"b={b} is not divisible by gcd(a)={d}")

    n = len(a)
    np1 = n + 1
    A_mat = np.zeros((n, np1), dtype=int)
    A_mat[:, 0] = a
    A_mat[:, 1:] = np.eye(n, dtype=int)


    while np.count_nonzero(A_mat[:, 0]) > 1:
        nonzero = np.where(A_mat[:, 0] != 0)[0]
        magnitudes = np.abs(A_mat[nonzero, 0])
        p_idx = nonzero[np.argmin(magnitudes)]

        A_mat[[0, p_idx], :] = A_mat[[p_idx, 0], :].copy()
        for i in range(1, n):
            s = int(np.fix(A_mat[i, 0] / A_mat[0, 0]))
            A_mat[i, :] -= s * A_mat[0, :]

    d_out = A_mat[0, 0]
    f = b // d_out
    v = A_mat[0, 1:].copy() * f
    B = A_mat[1:, 1:].T.copy()

    kmin = -np.inf * np.ones(n - 1)
    kmax = np.inf * np.ones(n - 1)
    for j in range(n - 1):
        for i in range(n):
            if B[i, j] < 0:
                kmax[j] = min(kmax[j], -v[i] / B[i, j])
            elif B[i, j] > 0:
                kmin[j] = max(kmin[j], -v[i] / B[i, j])

    return d_out, v, B, kmin, kmax


def optimize_node_numbering_bandwidth(contact_nodes: np.ndarray,
                                       total_nodes: int,
                                       max_band: int = 10) -> np.ndarray:
    contact_nodes = np.array(contact_nodes, dtype=int)
    n_c = len(contact_nodes)
    if n_c == 0:
        return np.arange(total_nodes)



    new_order = np.arange(total_nodes)
    other_nodes = np.setdiff1d(np.arange(total_nodes), contact_nodes)
    reordered = np.concatenate([contact_nodes, other_nodes])

    inv_map = np.zeros(total_nodes, dtype=int)
    inv_map[reordered] = np.arange(total_nodes)
    return inv_map


def compute_matrix_bandwidth(K: np.ndarray) -> Tuple[int, int]:
    n = K.shape[0]
    ml = 0
    mu = 0
    tol = 1e-14 * np.max(np.abs(K))
    for i in range(n):
        for j in range(n):
            if abs(K[i, j]) > tol:
                ml = max(ml, i - j)
                mu = max(mu, j - i)
    return ml, mu
