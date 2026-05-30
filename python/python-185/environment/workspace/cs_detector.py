
import numpy as np
from typing import Optional, Tuple


def soft_thresholding(x: np.ndarray, lambda_: float) -> np.ndarray:
    if lambda_ < 0:
        raise ValueError("阈值 lambda 必须非负")
    return np.sign(x) * np.maximum(np.abs(x) - lambda_, 0.0)


def ista_reconstruction(A: np.ndarray, y: np.ndarray,
                        lambda_: float,
                        max_iter: int = 1000,
                        tol: float = 1e-6,
                        x0: Optional[np.ndarray] = None) -> np.ndarray:
    A = np.asarray(A, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    m, N = A.shape

    if len(y) != m:
        raise ValueError(f"测量向量维度 {len(y)} 与矩阵行数 {m} 不匹配")


    L = np.linalg.norm(A.T @ A, 2)
    if L < 1e-14:
        raise ValueError("感知矩阵 A 的奇异值过小，问题病态")

    step = 1.0 / L

    if x0 is None:
        x = np.zeros(N, dtype=float)
    else:
        x = np.asarray(x0, dtype=float).ravel().copy()
        if len(x) != N:
            raise ValueError("初始向量维度与 A 的列数不匹配")

    for k in range(max_iter):
        x_old = x.copy()

        grad = A.T @ (A @ x - y)

        x = soft_thresholding(x - step * grad, lambda_ * step)


        if np.linalg.norm(x - x_old) < tol * max(1.0, np.linalg.norm(x_old)):
            break

    return x


def fista_reconstruction(A: np.ndarray, y: np.ndarray,
                         lambda_: float,
                         max_iter: int = 1000,
                         tol: float = 1e-6,
                         x0: Optional[np.ndarray] = None) -> np.ndarray:
    A = np.asarray(A, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    m, N = A.shape

    if len(y) != m:
        raise ValueError("测量向量维度与矩阵行数不匹配")

















    raise NotImplementedError("Hole_2: FISTA 算法待实现")


def orthogonal_matching_pursuit(A: np.ndarray, y: np.ndarray,
                                sparsity: int,
                                max_iter: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    A = np.asarray(A, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    m, N = A.shape

    if max_iter is None:
        max_iter = sparsity

    if sparsity <= 0 or sparsity > N:
        raise ValueError(f"稀疏度必须在 [1, {N}] 范围内")

    residual = y.copy()
    support = []
    x = np.zeros(N, dtype=float)

    for _ in range(max_iter):

        correlations = np.abs(A.T @ residual)

        for idx in support:
            correlations[idx] = -1.0

        j = np.argmax(correlations)
        if correlations[j] < 1e-14:
            break

        support.append(j)


        A_omega = A[:, support]
        try:
            x_omega = np.linalg.lstsq(A_omega, y, rcond=None)[0]
        except np.linalg.LinAlgError:
            break


        x.fill(0.0)
        x[support] = x_omega
        residual = y - A_omega @ x_omega

        if np.linalg.norm(residual) < 1e-12 * np.linalg.norm(y):
            break

    return x, np.array(support, dtype=int)


def build_sensing_matrix_gaussian(m: int, N: int, normalize: bool = True) -> np.ndarray:
    if m <= 0 or N <= 0:
        raise ValueError("m 和 N 必须为正整数")

    Phi = np.random.randn(m, N) / np.sqrt(m)
    if normalize:
        col_norms = np.linalg.norm(Phi, axis=0)
        col_norms[col_norms == 0] = 1.0
        Phi = Phi / col_norms
    return Phi
