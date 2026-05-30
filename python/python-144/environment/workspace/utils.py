
import numpy as np
from scipy.spatial import ConvexHull
from scipy.linalg import eigh


def caesar_perturb(data: np.ndarray, k: int = 3, axis: int = -1) -> np.ndarray:
    if data.size == 0:
        raise ValueError("caesar_perturb: 输入数据不能为空。")
    shifted = np.roll(data, shift=k, axis=axis)
    sigma = np.std(data) / 10.0
    if sigma < 1e-12:
        sigma = 1e-6
    noise = np.random.normal(0.0, sigma, size=data.shape)
    return shifted + noise


def matrix_interpolation_upsample(A: np.ndarray, factor: int = 2) -> np.ndarray:
    if factor != 2:
        raise ValueError("matrix_interpolation_upsample: 当前仅支持 factor=2。")
    m, n = A.shape
    if m < 2 or n < 2:
        raise ValueError("matrix_interpolation_upsample: 矩阵维度至少为 2×2。")
    B = np.zeros((2 * m, 2 * n), dtype=A.dtype)
    B[0::2, 0::2] = A

    B[0::2, 1::2][:, :-1] = (A[:, :-1] + A[:, 1:]) / 2.0
    B[0::2, -1] = A[:, -1]

    for i in range(m - 1):
        B[2*i+1, :] = (B[2*i, :] + B[2*i+2, :]) / 2.0
    B[-1, :] = B[-2, :]
    return B


def polygonal_convex_hull(points: np.ndarray) -> dict:
    if points.ndim != 2:
        raise ValueError("polygonal_convex_hull: 输入必须是二维数组。")
    if points.shape[0] < points.shape[1] + 1:
        raise ValueError("polygonal_convex_hull: 点数量不足，无法构成非退化凸包。")
    hull = ConvexHull(points)
    return {
        "vertices": hull.vertices,
        "volume": hull.volume,
        "simplices": hull.simplices,
        "n_points": points.shape[0],
    }


def distance_to_position_mds(distance: np.ndarray, dim: int = 2,
                              max_iter: int = 200, tol: float = 1e-6) -> np.ndarray:
    n = distance.shape[0]
    if distance.shape[0] != distance.shape[1]:
        raise ValueError("distance_to_position_mds: 距离矩阵必须是方阵。")

    D2 = distance ** 2

    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ D2 @ J

    B = 0.5 * (B + B.T)
    eigvals, eigvecs = eigh(B)

    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    positive = eigvals > 1e-10
    if np.sum(positive) < dim:
        raise RuntimeError("distance_to_position_mds: 正特征值数量不足，无法嵌入到指定维度。")
    Lambda = np.diag(np.sqrt(np.maximum(eigvals[:dim], 0.0)))
    X = eigvecs[:, :dim] @ Lambda
    return X


def r8mat_condition_number(A: np.ndarray) -> float:
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("r8mat_condition_number: 输入必须是方阵。")
    s = np.linalg.svd(A, compute_uv=False)
    if s[-1] < 1e-15:
        return np.inf
    return s[0] / s[-1]
