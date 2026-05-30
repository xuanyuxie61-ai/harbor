
import numpy as np
from typing import Tuple, List, Optional


def sample_covariance(X: np.ndarray) -> np.ndarray:
    n, p = X.shape
    if n < 2:
        raise ValueError("样本数 n 必须至少为 2 才能计算协方差。")
    Xc = X - np.mean(X, axis=0, keepdims=True)
    S = Xc.T @ Xc / n

    S = 0.5 * (S + S.T)
    return S


def soft_threshold(M: np.ndarray, tau: float) -> np.ndarray:
    if tau < 0.0:
        raise ValueError("阈值 tau 必须非负。")
    res = np.sign(M) * np.maximum(np.abs(M) - tau, 0.0)
    np.fill_diagonal(res, np.diag(M))
    return res


def graphical_lasso(S: np.ndarray,
                    lam: float = 0.05,
                    max_iter: int = 200,
                    tol: float = 1e-6,
                    eta: float = 0.5,
                    verbose: bool = False) -> np.ndarray:
    p = S.shape[0]
    if p == 0:
        raise ValueError("协方差矩阵维度不能为 0。")

    Theta = np.diag(1.0 / (np.diag(S) + 1e-8))








    raise NotImplementedError("Hole 1: Graphical Lasso 迭代循环待实现")

    return Theta


def threshold_precision(Theta: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    Theta_s = Theta.copy()
    Theta_s[np.abs(Theta_s) < eps] = 0.0
    np.fill_diagonal(Theta_s, np.diag(Theta))
    return Theta_s


def dense_to_csr(A: np.ndarray, tol: float = 1e-12) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    m, n = A.shape
    data_list: List[float] = []
    indices_list: List[int] = []
    indptr = np.zeros(m + 1, dtype=int)

    for i in range(m):
        row_nnz = 0
        for j in range(n):
            val = A[i, j]
            if np.abs(val) > tol:
                data_list.append(float(val))
                indices_list.append(j)
                row_nnz += 1
        indptr[i + 1] = indptr[i] + row_nnz

    data = np.array(data_list, dtype=float)
    indices = np.array(indices_list, dtype=int)
    return data, indices, indptr


def csr_to_dense(data: np.ndarray, indices: np.ndarray, indptr: np.ndarray, ncols: int) -> np.ndarray:
    m = indptr.shape[0] - 1
    A = np.zeros((m, ncols), dtype=float)
    for i in range(m):
        for idx in range(indptr[i], indptr[i + 1]):
            j = indices[idx]
            A[i, j] = data[idx]
    return A


def extract_causal_skeleton(Theta_sparse: np.ndarray) -> Tuple[List[Tuple[int, int, float]], int]:
    p = Theta_sparse.shape[0]
    edges = []
    for i in range(p):
        for j in range(i + 1, p):
            w = Theta_sparse[i, j]
            if w != 0.0:
                edges.append((i, j, float(w)))
    return edges, p


def demo():
    np.random.seed(42)
    p = 12
    n = 500

    Theta_true = np.eye(p) * 2.0
    for i in range(p - 1):
        Theta_true[i, i + 1] = 0.4
        Theta_true[i + 1, i] = 0.4
    Theta_true[0, 3] = 0.3
    Theta_true[3, 0] = 0.3
    Theta_true[5, 8] = -0.25
    Theta_true[8, 5] = -0.25


    Sigma_true = np.linalg.inv(Theta_true)
    X = np.random.multivariate_normal(np.zeros(p), Sigma_true, size=n)

    S = sample_covariance(X)
    Theta_est = graphical_lasso(S, lam=0.08, max_iter=300, verbose=True)
    Theta_sparse = threshold_precision(Theta_est, eps=5e-3)

    edges, _ = extract_causal_skeleton(Theta_sparse)
    print(f"[sparse_sem_matrix] 估计的因果骨架边数: {len(edges)}")


    data, indices, indptr = dense_to_csr(Theta_sparse)
    Theta_recover = csr_to_dense(data, indices, indptr, p)
    rec_err = np.linalg.norm(Theta_sparse - Theta_recover, 'fro')
    print(f"[sparse_sem_matrix] CSR 重构误差: {rec_err:.2e}")
    return Theta_sparse, edges


if __name__ == "__main__":
    demo()
