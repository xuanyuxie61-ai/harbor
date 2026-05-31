"""
utils.py
========
通用数值工具与辅助函数。
"""

import numpy as np
from typing import Tuple


def safe_divide(a: np.ndarray, b: np.ndarray, fill_value: float = 0.0) -> np.ndarray:
    """安全除法，避免除以零。"""
    result = np.full_like(a, fill_value, dtype=float)
    mask = np.abs(b) > 1.0e-300
    result[mask] = a[mask] / b[mask]
    return result


def clamp(value: float, vmin: float, vmax: float) -> float:
    """将数值限制在 [vmin, vmax] 区间内。"""
    return max(vmin, min(vmax, value))


def clamp_array(arr: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    """将数组所有元素限制在 [vmin, vmax] 区间内。"""
    return np.clip(arr, vmin, vmax)


def spherical_volume(r_inner: float, r_outer: float) -> float:
    """计算球壳体积 V = 4/3 * pi * (r_outer^3 - r_inner^3)。"""
    if r_outer < r_inner or r_outer < 0.0 or r_inner < 0.0:
        return 0.0
    return (4.0 / 3.0) * np.pi * (r_outer**3 - r_inner**3)


def spherical_surface_area(r: float) -> float:
    """计算球表面积 A = 4 * pi * r^2。"""
    if r < 0.0:
        return 0.0
    return 4.0 * np.pi * r * r


def log_mean(a: float, b: float) -> float:
    """
    计算对数平均数 L(a,b) = (a - b) / (ln(a) - ln(b))，用于热传导系数插值。
    当 a ≈ b 时退化为算术平均。
    """
    if a <= 0.0 or b <= 0.0:
        return 0.0
    ratio = a / b
    if np.abs(ratio - 1.0) < 1.0e-6:
        return 0.5 * (a + b)
    return (a - b) / np.log(ratio)


def vector_norm(v: np.ndarray) -> float:
    """计算向量的L2范数。"""
    return float(np.sqrt(np.sum(v * v)))


def normalize_vector(v: np.ndarray) -> np.ndarray:
    """归一化向量，若范数过小则返回零向量。"""
    norm = vector_norm(v)
    if norm < 1.0e-15:
        return np.zeros_like(v)
    return v / norm


def gauss_jacobi_quadrature_standard(n: int, alpha: float, beta: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算标准区间 [-1, 1] 上的 Gauss-Jacobi 求积节点与权重。
    使用 Golub-Welsch 算法（对称三对角矩阵特征值问题）。

    权函数: w(x) = (1-x)^alpha * (1+x)^beta

    参数
    ----
    n : int
        节点数
    alpha, beta : float
        Jacobi 参数，需满足 alpha > -1, beta > -1

    返回
    ----
    x, w : 节点与权重数组
    """
    if n < 1:
        return np.array([]), np.array([])
    if alpha <= -1.0 or beta <= -1.0:
        raise ValueError("Gauss-Jacobi 参数需满足 alpha > -1 且 beta > -1")

    # 构造对称三对角 Jacobi 矩阵
    ab = alpha + beta
    abi = 2.0 + ab

    # 对角线元素
    diag = np.zeros(n)
    diag[0] = (beta - alpha) / abi
    if n > 1:
        a2b2 = beta * beta - alpha * alpha
        for i in range(1, n):
            idx = i + 1
            ab_i = 2.0 * idx + ab
            diag[i] = a2b2 / ((ab_i - 2.0) * ab_i)

    # 次对角线元素
    subdiag = np.zeros(n - 1)
    if n > 1:
        subdiag[0] = np.sqrt(4.0 * (1.0 + alpha) * (1.0 + beta)
                             / ((abi + 1.0) * abi * abi))
        for i in range(1, n - 1):
            idx = i + 1
            ab_i = 2.0 * idx + ab
            subdiag[i] = np.sqrt(4.0 * idx * (idx + alpha) * (idx + beta) * (idx + ab)
                                 / ((ab_i * ab_i - 1.0) * ab_i * ab_i))

    # 求解特征值问题
    # 构造三对角矩阵并用 numpy 特征值求解
    T = np.diag(diag) + np.diag(subdiag, k=1) + np.diag(subdiag, k=-1)
    eigenvalues, eigenvectors = np.linalg.eigh(T)

    # 节点为特征值
    x = eigenvalues

    # 权重计算
    # 首项正交多项式归一化常数
    mu0 = 2.0**(ab + 1.0) * np.exp(
        np.math.lgamma(alpha + 1.0) + np.math.lgamma(beta + 1.0) - np.math.lgamma(ab + 2.0)
    )
    w = mu0 * (eigenvectors[0, :]**2)

    return x, w


def scale_quadrature(x: np.ndarray, w: np.ndarray, a: float, b: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    将 [-1, 1] 上的求积公式缩放至 [a, b]。
    x_new = (b-a)/2 * x + (a+b)/2
    w_new = (b-a)/2 * w
    """
    if b <= a:
        raise ValueError("区间右端点必须大于左端点")
    scale = (b - a) / 2.0
    shift = (a + b) / 2.0
    return x * scale + shift, w * scale
