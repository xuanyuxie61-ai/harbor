
import numpy as np
from typing import Tuple
from system_utils import EPS, TOL_RANK, MAX_ITER, check_finite






def annulus_sample(n: int, pc: np.ndarray, r1: float, r2: float) -> np.ndarray:
    if r1 < 0 or r2 <= r1:
        raise ValueError("Require 0 ≤ r1 < r2.")
    pc = np.asarray(pc, dtype=float)
    v = np.random.rand(n)
    u = np.random.rand(n)
    r = np.sqrt((1.0 - v) * r1 * r1 + v * r2 * r2)
    theta = 2.0 * np.pi * u
    pts = np.column_stack((pc[0] + r * np.cos(theta),
                           pc[1] + r * np.sin(theta)))
    return pts


def gaussian_random_matrix(m: int, n: int, seed: int = None) -> np.ndarray:
    if seed is not None:
        np.random.seed(seed)
    return np.random.randn(m, n)


def subsampled_random_fourier_transform(n: int, l: int, seed: int = None) -> np.ndarray:
    if seed is not None:
        np.random.seed(seed)
    D = np.random.choice([-1.0, 1.0], size=n)


    idx = np.random.choice(n, size=l, replace=False)
    Omega = np.zeros((l, n), dtype=complex)
    for i, row in enumerate(idx):
        Omega[i, row] = 1.0


    return np.random.randn(l, n) / np.sqrt(l)






def randomized_range_finder(A: np.ndarray, k: int, p: int = 5,
                            seed: int = None) -> np.ndarray:
    A = np.asarray(A, dtype=float)
    m, n = A.shape
    l = k + p
    if seed is not None:
        np.random.seed(seed)
    Omega = np.random.randn(n, l)

    Y = A @ Omega
    Y = A @ (A.T @ Y)
    Y = A @ (A.T @ Y)
    Q, _ = np.linalg.qr(Y, mode='reduced')
    return Q


def randomized_svd(A: np.ndarray, k: int, p: int = 5,
                   seed: int = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    A = np.asarray(A, dtype=float)
    Q = randomized_range_finder(A, k, p=p, seed=seed)
    B = Q.T @ A
    U_b, s, Vt = np.linalg.svd(B, full_matrices=False)
    U = Q @ U_b

    U = U[:, :k]
    s = s[:k]
    Vt = Vt[:k, :]
    return U, s, Vt






def hilbert_matrix(m: int, n: int) -> np.ndarray:
    i = np.arange(1, m + 1, dtype=float).reshape(-1, 1)
    j = np.arange(1, n + 1, dtype=float).reshape(1, -1)
    H = 1.0 / (i + j - 1.0)
    return H


def low_rank_test_matrix(m: int, n: int, rank: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    U, _ = np.linalg.qr(rng.standard_normal((m, rank)), mode='reduced')
    V, _ = np.linalg.qr(rng.standard_normal((n, rank)), mode='reduced')
    sigma = np.exp(-np.linspace(0.0, 3.0, rank))
    A = U * sigma @ V.T
    return A






def adaptive_rank_threshold(s: np.ndarray, tol: float = 1e-10) -> int:
    s = np.asarray(s, dtype=float)
    if s.size == 0 or s[0] < EPS:
        return 0
    thresh = tol * s[0]
    r = int(np.sum(s > thresh))
    return max(r, 1)


from typing import Tuple
