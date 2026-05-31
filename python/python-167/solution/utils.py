"""
utils.py
通用工具模块：性能计时、数值稳定性检验、边界条件判定、鲁棒性处理。
融入种子项目 1417_wtime（墙钟计时）、726_matlab_mistake（数值精度与逻辑错误检测）。
"""

import time
import numpy as np
from typing import Tuple, Optional


class Timer:
    """
    基于 wtime.m 思想的高精度墙钟计时器。
    记录从首次调用开始经过的秒数。
    """
    def __init__(self):
        self._start = time.perf_counter()

    def elapsed(self) -> float:
        return time.perf_counter() - self._start

    def reset(self):
        self._start = time.perf_counter()


def check_numerical_singularity(A: np.ndarray, tol: float = 1e-12) -> bool:
    """
    基于 matlab_mistake 的数值精度意识，检测矩阵是否接近奇异。
    使用条件数与显式零对角检测双重策略。

    科学原理：
    条件数 κ(A) = ||A|| · ||A^{-1}||，当 κ >> 1 时，线性系统 Ax=b
    的相对误差满足
        ||δx||/||x|| ≤ κ(A) · ||δb||/||b||
    数值计算中若 κ(A) > 1/tol，则认为矩阵接近奇异。
    """
    if A.size == 0:
        return True
    # 显式检测零对角（常见于质量矩阵）
    if A.ndim == 2 and A.shape[0] == A.shape[1]:
        diag_abs = np.abs(np.diag(A))
        if np.any(diag_abs < tol):
            return True
    # 条件数检测
    cond = np.linalg.cond(A)
    return cond > 1.0 / tol


def safe_divide(a: float, b: float, fallback: float = 0.0) -> float:
    """
    安全除法，避免除零错误。
    参考 matlab_mistake 中对 1/(i-5) 类奇异点的处理思想。
    """
    if np.isclose(b, 0.0, atol=1e-15):
        return fallback
    return a / b


def robust_sqrt(x: float, eps: float = 1e-14) -> float:
    """
    鲁棒平方根：对负数输入返回 sqrt(max(x, eps))。
    在 Cholesky 分解等场景避免复数结果。
    """
    return np.sqrt(max(float(x), eps))


def clip_to_bounds(val: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """
    将数值裁剪到边界区间 [lower, upper]，用于关节限位与物理约束。
    """
    return np.clip(val, lower, upper)


def finite_difference_jacobian(func, x: np.ndarray, h: float = 1e-6) -> np.ndarray:
    """
    中心差分计算 Jacobian 矩阵。
    对向量函数 f: R^n → R^m，J_{ij} = ∂f_i/∂x_j。

    数学公式（中心差分，二阶精度）：
        ∂f_i/∂x_j ≈ (f_i(x + h·e_j) - f_i(x - h·e_j)) / (2h)
        截断误差 O(h^2)。
    """
    n = x.size
    fx = func(x)
    m = fx.size
    J = np.zeros((m, n), dtype=float)
    for j in range(n):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[j] += h
        x_minus[j] -= h
        J[:, j] = (func(x_plus) - func(x_minus)) / (2.0 * h)
    return J


def householder_reflection(v: np.ndarray) -> np.ndarray:
    """
    Householder 反射矩阵计算，用于 QR 分解与数值稳定性提升。

    给定非零向量 v，Householder 矩阵
        H = I - 2·(vv^T)/(v^T v)
    满足 H·v = -sign(v_1)·||v||·e_1，可将任意向量反射到坐标轴方向。
    这是数值线性代数中保证正交性的核心工具。
    """
    v = v.astype(float).copy()
    norm_v = np.linalg.norm(v)
    if norm_v < 1e-15:
        return np.eye(len(v))
    v[0] += np.sign(v[0]) * norm_v
    H = np.eye(len(v)) - 2.0 * np.outer(v, v) / np.dot(v, v)
    return H


def gershgorin_discs(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Gershgorin 圆盘定理：对矩阵 A，其特征值 λ 必位于某个圆盘
        D_i = { z ∈ C : |z - a_{ii}| ≤ R_i }
    其中 R_i = Σ_{j≠i} |a_{ij}|。

    返回值：
        centers —— 圆盘中心 (对角元)
        radii   —— 圆盘半径
    """
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be square matrix")
    n = A.shape[0]
    centers = np.diag(A).copy()
    radii = np.zeros(n)
    for i in range(n):
        radii[i] = np.sum(np.abs(A[i, :])) - np.abs(A[i, i])
    return centers, radii
