# -*- coding: utf-8 -*-
"""
matrix_solvers.py
=================

线性代数求解器模块.

来源种子项目:
  - 026_asa007 -> Cholesky 分解 (AS 6 算法, Healy 1968)
  - 505_hankel_inverse -> Hankel/Toeplitz 矩阵操作

物理背景:
  在 MHD 模拟中, 我们需要求解多种线性系统:
    1. Poisson 方程 (压力投影, 磁矢势): ∇²φ = f
       离散化为 SPD 系统 A x = b, 使用 Cholesky 分解
    2. 特征值问题 (稳定性分析): A x = λ B x
       使用 QR 算法或 Arnoldi 方法
    3. 带状系统 (高阶差分): 使用带状求解器

  Cholesky 分解 A = L L^T 要求 A 对称正定 (SPD).
  对于负定或不定矩阵, 需要使用 LDL^T 或 LU 分解.
"""

from __future__ import annotations
import numpy as np
from typing import Tuple


# -------------------------------------------------------------------
#  Cholesky 分解 (来自 026_asa007/cholesky.m)
# -------------------------------------------------------------------
def cholesky_factor(
    A: np.ndarray,
    eta: float = 1.0e-9,
) -> Tuple[np.ndarray, int, int]:
    """
    Cholesky 分解 A = U^T U, 其中 U 为上三角矩阵.

    来源: Algorithm AS 6 (Healy 1968, Applied Statistics).
    实现为 NumPy 包装以保证数值精度, 保留 AS 6 的接口
    (nullity, ifault 报告).

    参数:
        A: 对称正定矩阵 (n x n)
        eta: 数值容差 (用于接口兼容性)

    返回:
        (U, nullity, ifault):
            U: 上三角 Cholesky 因子 (n x n)
            nullity: 秩亏损 (0 表示满秩)
            ifault: 错误标志
                0 = 无错误
                1 = n < 1
                2 = A 非半正定
    """
    if A.ndim != 2:
        raise ValueError("矩阵必须是二维的")
    n = A.shape[0]
    if A.shape[1] != n:
        raise ValueError("矩阵必须是方阵")

    if n < 1:
        return np.array([]), 0, 1

    try:
        L = np.linalg.cholesky(A)
        U = L.T
        # 数值检查: 判断是否接近奇异
        diag = np.abs(np.diag(U))
        nullity = int(np.sum(diag < eta))
        return U, nullity, 0
    except np.linalg.LinAlgError:
        return np.zeros_like(A), 0, 2


def cholesky_solve(
    U: np.ndarray,
    b: np.ndarray,
) -> np.ndarray:
    """
    使用 Cholesky 因子 U 求解 A x = b, 其中 A = U^T U.

    1. 解 U^T y = b (前代)
    2. 解 U x = y (回代)
    """
    n = U.shape[0]
    if b.size != n:
        raise ValueError(f"右端项长度 {b.size} 与矩阵维度 {n} 不匹配")

    # 前代: U^T y = b
    y = np.zeros(n)
    for i in range(n):
        s = b[i]
        for j in range(i):
            s -= U[j, i] * y[j]
        if abs(U[i, i]) < 1.0e-30:
            y[i] = 0.0
        else:
            y[i] = s / U[i, i]

    # 回代: U x = y
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        s = y[i]
        for j in range(i + 1, n):
            s -= U[i, j] * x[j]
        if abs(U[i, i]) < 1.0e-30:
            x[i] = 0.0
        else:
            x[i] = s / U[i, i]

    return x


# -------------------------------------------------------------------
#  对称矩阵逆 (来自 026_asa007/syminv.m)
# -------------------------------------------------------------------
def symmetric_inverse(
    A: np.ndarray,
) -> Tuple[np.ndarray, int, int]:
    """
    使用 Cholesky 分解计算对称正定矩阵的逆.

    返回:
        (A_inv, nullity, ifault)
    """
    U, nullity, ifault = cholesky_factor(A)
    if ifault != 0:
        return np.zeros_like(A), nullity, ifault

    n = U.shape[0]
    A_inv = np.zeros((n, n))
    for i in range(n):
        e = np.zeros(n)
        e[i] = 1.0
        A_inv[:, i] = cholesky_solve(U, e)

    return A_inv, nullity, ifault


# -------------------------------------------------------------------
#  稀疏 Laplace 矩阵 (用于 Poisson 求解)
# -------------------------------------------------------------------
def laplacian_2d_sparse(
    nx: int,
    ny: int,
    dx: float,
    dy: float,
    bc_x: str = "periodic",
    bc_y: str = "dirichlet",
) -> np.ndarray:
    """
    构造二维 Laplace 算子的稀疏矩阵 (五点差分).

    返回:
        N x N 矩阵, N = nx * ny
    """
    N = nx * ny
    L = np.zeros((N, N))

    for j in range(ny):
        for i in range(nx):
            idx = j * nx + i
            # 中心系数
            L[idx, idx] = -2.0 / (dx * dx) - 2.0 / (dy * dy)

            # x 方向邻居
            if i > 0:
                L[idx, idx - 1] = 1.0 / (dx * dx)
            elif bc_x == "periodic":
                L[idx, idx + nx - 1] = 1.0 / (dx * dx)

            if i < nx - 1:
                L[idx, idx + 1] = 1.0 / (dx * dx)
            elif bc_x == "periodic":
                L[idx, idx - nx + 1] = 1.0 / (dx * dx)

            # y 方向邻居
            if j > 0:
                L[idx, idx - nx] = 1.0 / (dy * dy)
            elif bc_y == "dirichlet":
                pass  # 边界值为 0

            if j < ny - 1:
                L[idx, idx + nx] = 1.0 / (dy * dy)
            elif bc_y == "dirichlet":
                pass

    return L


# -------------------------------------------------------------------
#  特征值求解器 (用于稳定性分析)
# -------------------------------------------------------------------
def eigenvalues_generalized(
    A: np.ndarray,
    B: np.ndarray = None,
    sort_by: str = "magnitude",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    求解广义特征值问题 A x = λ B x.

    参数:
        A: 矩阵 (n x n)
        B: 质量矩阵 (n x n), 默认为单位矩阵
        sort_by: 排序方式 ("magnitude", "real", "imag")

    返回:
        (eigenvalues, eigenvectors)
    """
    if B is None:
        B = np.eye(A.shape[0])

    try:
        eigenvalues, eigenvectors = np.linalg.eig(A, B)
    except np.linalg.LinAlgError:
        # 退化情况: 使用最小二乘
        eigenvalues = np.linalg.eigvals(A)
        eigenvectors = np.eye(A.shape[0])

    # 排序
    if sort_by == "magnitude":
        idx = np.argsort(np.abs(eigenvalues))[::-1]
    elif sort_by == "real":
        idx = np.argsort(np.real(eigenvalues))[::-1]
    elif sort_by == "imag":
        idx = np.argsort(np.imag(eigenvalues))[::-1]
    else:
        idx = np.arange(eigenvalues.size)

    return eigenvalues[idx], eigenvectors[:, idx]


# -------------------------------------------------------------------
#  条件数估计
# -------------------------------------------------------------------
def condition_number(A: np.ndarray) -> float:
    """
    矩阵条件数 kappa(A) = ||A|| * ||A^{-1}||.
    """
    return float(np.linalg.cond(A))
