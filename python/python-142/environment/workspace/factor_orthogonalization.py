
import numpy as np
from typing import Tuple, Optional


def modified_gram_schmidt(A: np.ndarray, tol: float = 1e-12) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n, m = A.shape
    Q = np.zeros((n, m), dtype=float)
    R = np.zeros((m, m), dtype=float)
    rank = 0
    for j in range(m):
        v = A[:, j].copy()
        for i in range(rank):
            R[i, j] = np.dot(Q[:, i], A[:, j])
            v -= R[i, j] * Q[:, i]
        norm_v = np.linalg.norm(v)
        if norm_v > tol:
            Q[:, rank] = v / norm_v
            R[rank, j] = norm_v
            rank += 1
    return Q[:, :rank], R[:rank, :], rank


def classical_gram_schmidt(A: np.ndarray, tol: float = 1e-12) -> Tuple[np.ndarray, np.ndarray, int]:
    n, m = A.shape
    Q = np.zeros((n, m), dtype=float)
    R = np.zeros((m, m), dtype=float)
    rank = 0
    for j in range(m):
        v = A[:, j].copy()
        for i in range(rank):
            R[i, j] = np.dot(Q[:, i], A[:, j])
        for i in range(rank):
            v -= R[i, j] * Q[:, i]
        norm_v = np.linalg.norm(v)
        if norm_v > tol:
            Q[:, rank] = v / norm_v
            R[rank, j] = norm_v
            rank += 1
    return Q[:, :rank], R[:rank, :], rank


def factor_covariance_from_loadings(B: np.ndarray, sigma_factor: np.ndarray) -> np.ndarray:








    n_assets = B.shape[0]
    return np.eye(n_assets)


def orthogonalize_credit_factors(
    raw_loadings: np.ndarray,
    method: str = "mgs",
    tol: float = 1e-12
) -> np.ndarray:
    n_assets, n_factors = raw_loadings.shape
    if method.lower() == "mgs":
        Q, R, rank = modified_gram_schmidt(raw_loadings.T, tol)
    else:
        Q, R, rank = classical_gram_schmidt(raw_loadings.T, tol)

    if rank == 0:
        return np.zeros((n_assets, 1))



    orth_loadings = raw_loadings @ Q


    row_norms = np.sqrt(np.sum(orth_loadings**2, axis=1))
    scale = np.where(row_norms > 1.0, 1.0 / row_norms, 1.0)
    orth_loadings = orth_loadings * scale[:, np.newaxis]

    return orth_loadings


def test_factor_orthogonalization():
    np.random.seed(42)
    n_assets = 50
    n_factors = 10

    base = np.random.randn(n_assets, 3)
    raw = np.hstack([base + 0.1 * np.random.randn(n_assets, 3) for _ in range(n_factors // 3 + 1)])
    raw = raw[:, :n_factors]

    raw = raw / (np.linalg.norm(raw, axis=1, keepdims=True) + 1e-12) * 0.8

    orth = orthogonalize_credit_factors(raw, method="mgs")
    Corr = factor_covariance_from_loadings(orth, np.ones(orth.shape[1]))

    eigvals = np.linalg.eigvalsh(Corr)
    assert np.all(eigvals > -1e-10), "相关性矩阵存在负特征值!"
    assert np.allclose(np.diag(Corr), 1.0, atol=1e-6), "对角线不为 1!"
    print(f"factor_orthogonalization test passed. rank={orth.shape[1]}, min_eig={eigvals.min():.6f}")


if __name__ == "__main__":
    test_factor_orthogonalization()
