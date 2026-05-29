"""
fast_solver.py
==============
基于三对角矩阵存储的共轭梯度快速求解器模块

科学背景：
---------
在压缩感知重建的内层循环中，需要反复求解形如
    (A^T A + \lambda D) x = b
的线性系统。当 A 具有特殊结构（如部分傅里叶、部分 DCT）时，
A^T A 可能是带状矩阵。对于三对角矩阵，共轭梯度法（CG）具有 O(N)
的每步计算复杂度，远优于稠密矩阵的 O(N^2)。

三对角矩阵的 R83 存储格式：
    对于 N \times N 三对角矩阵 A，存储为 3 \times N 数组：
        A_r83[0, j] = A_{j, j+1}  （上对角线，j=0,...,N-2）
        A_r83[1, j] = A_{j, j}    （主对角线，j=0,...,N-1）
        A_r83[2, j] = A_{j+1, j}  （下对角线，j=0,...,N-2）

共轭梯度算法（Hestenes & Stiefel, 1952）：
    输入：对称正定矩阵 A，右端项 b，初始猜测 x_0
    初始化：r_0 = b - A x_0, p_0 = r_0
    对 k = 0, 1, 2, ...：
        \alpha_k = \frac{r_k^T r_k}{p_k^T A p_k}
        x_{k+1} = x_k + \alpha_k p_k
        r_{k+1} = r_k - \alpha_k A p_k
        \beta_k = \frac{r_{k+1}^T r_{k+1}}{r_k^T r_k}
        p_{k+1} = r_{k+1} + \beta_k p_k

理论性质：
    - 精确解在最多 N 步内得到（无舍入误差时）
    - 误差满足：\|x_k - x^*\|_A \leq 2 \left(\frac{\sqrt{\kappa} - 1}{\sqrt{\kappa} + 1}\right)^k \|x_0 - x^*\|_A
      其中 \kappa = \lambda_{\max}(A) / \lambda_{\min}(A) 为条件数。

来自项目 962_r83 的核心算法。
"""

import numpy as np
from typing import Optional


def r83_mv(m: int, n: int, a: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    R83 格式三对角矩阵与向量相乘。

    参数:
        m, n: 矩阵维度
        a: R83 存储矩阵，形状为 (3, n)
        x: 输入向量，形状为 (n,)
    返回:
        结果向量 b = A x，形状为 (m,)
    """
    a = np.asarray(a, dtype=float)
    x = np.asarray(x, dtype=float).ravel()

    if a.shape != (3, n):
        raise ValueError(f"R83 矩阵形状必须是 (3, {n})，实际为 {a.shape}")
    if len(x) != n:
        raise ValueError(f"向量维度 {len(x)} 与矩阵列数 {n} 不匹配")

    b = np.zeros(m, dtype=float)
    for j in range(n):
        i_start = max(0, j - 1)
        i_end = min(m, j + 2)
        for i in range(i_start, i_end):
            b[i] += a[i - j + 1, j] * x[j]
    return b


def r83_cg(n: int, a: np.ndarray, b: np.ndarray,
           x: Optional[np.ndarray] = None,
           max_iter: Optional[int] = None,
           tol: float = 1e-10) -> np.ndarray:
    """
    利用共轭梯度法求解 R83 三对角线性系统 A x = b。

    参数:
        n: 矩阵阶数
        a: R83 格式三对角矩阵，形状为 (3, n)
        b: 右端项向量，形状为 (n,)
        x: 初始猜测（默认零向量）
        max_iter: 最大迭代次数（默认 n）
        tol: 相对残差容差
    返回:
        近似解向量 x
    """
    b = np.asarray(b, dtype=float).ravel()
    if len(b) != n:
        raise ValueError(f"右端项维度 {len(b)} 与 n={n} 不匹配")

    if x is None:
        x = np.zeros(n, dtype=float)
    else:
        x = np.asarray(x, dtype=float).ravel().copy()
        if len(x) != n:
            raise ValueError("初始猜测维度不匹配")

    if max_iter is None:
        max_iter = n

    # 初始化
    ap = r83_mv(n, n, a, x)
    r = b - ap
    p = r.copy()

    rs_old = float(r @ r)
    b_norm = np.linalg.norm(b)
    if b_norm < 1e-14:
        b_norm = 1.0

    for it in range(max_iter):
        ap = r83_mv(n, n, a, p)
        pap = float(p @ ap)

        if abs(pap) < 1e-20:
            break

        alpha = rs_old / pap
        x += alpha * p
        r -= alpha * ap

        rs_new = float(r @ r)
        if np.sqrt(rs_new) < tol * b_norm:
            break

        beta = rs_new / rs_old
        p = r + beta * p
        rs_old = rs_new

    return x


def construct_tridiagonal_from_dense(A: np.ndarray) -> np.ndarray:
    """
    从稠密矩阵中提取三对角部分并转换为 R83 格式。

    参数:
        A: 稠密矩阵，形状为 (n, n)
    返回:
        R83 格式矩阵，形状为 (3, n)
    """
    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("输入必须是方阵")

    n = A.shape[0]
    a_r83 = np.zeros((3, n), dtype=float)

    for j in range(n):
        if j > 0:
            a_r83[2, j - 1] = A[j, j - 1]  # 下对角线
        a_r83[1, j] = A[j, j]              # 主对角线
        if j < n - 1:
            a_r83[0, j + 1] = A[j, j + 1]  # 上对角线

    return a_r83


def solve_normal_equations_cg(A: np.ndarray, y: np.ndarray,
                              lambda_reg: float = 1e-6,
                              max_iter: Optional[int] = None,
                              tol: float = 1e-10) -> np.ndarray:
    """
    利用三对角近似 + CG 快速求解正规方程 (A^T A + \lambda I) x = A^T y。

    策略：
        1. 计算 H = A^T A + \lambda I
        2. 提取 H 的三对角近似
        3. 用 R83-CG 求解

    参数:
        A: 感知矩阵，形状为 (m, N)
        y: 测量向量，形状为 (m,)
        lambda_reg: 正则化参数
        max_iter: CG 最大迭代次数
        tol: 收敛容差
    返回:
        解向量 x，形状为 (N,)
    """
    A = np.asarray(A, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    m, N = A.shape

    # 计算 H = A^T A + lambda * I
    H = A.T @ A
    H += lambda_reg * np.eye(N)

    # 提取三对角近似
    H_tri = np.zeros_like(H)
    for i in range(N):
        H_tri[i, i] = H[i, i]
        if i > 0:
            H_tri[i, i - 1] = H[i, i - 1]
            H_tri[i - 1, i] = H[i - 1, i]

    a_r83 = construct_tridiagonal_from_dense(H_tri)
    b = A.T @ y

    x = r83_cg(N, a_r83, b, max_iter=max_iter, tol=tol)
    return x
