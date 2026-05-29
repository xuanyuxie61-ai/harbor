"""
toeplitz_solver.py
================================================================================
对称Toeplitz矩阵求解模块 (来源于 999_r8sto 项目)
================================================================================
本模块实现对称Toeplitz线性系统的高效求解器，采用 Levinson-Durbin
递归算法，时间复杂度 O(N²)（优于通用的 O(N³)）。在潮汐能提取问题中，
周期性边界条件和自相关矩阵常呈现Toeplitz结构，本算法用于快速求解
流场离散化和时间序列预测中的线性系统。

核心公式:
    对称Toeplitz矩阵 A，由第一行元素 a[0:N-1] 定义:
        A_{ij} = a(|i-j|)

    Levinson-Durbin 递归:
        对于 k = 0, 1, ..., N-1:
            β_{k+1} = (1 - y_k²) β_k
            x_{k+1} = (b_{k+1} - Σ_{j=1}^{k} a_j x_{k+1-j}) / β_{k+1}
            x_j ← x_j + x_{k+1} y_{k+1-j},  j=1,...,k

    其中 y_k 为反射系数，满足 Yule-Walker 方程:
        A_k y_k = -a[1:k]
"""

import numpy as np
from typing import Tuple


def r8sto_sl(n: int, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    求解对称正定Toeplitz线性系统 A x = b。

    算法:
        Levinson-Durbin 递归 (Golub & Van Loan, Matrix Computations, 4.7.3)

    参数:
        n: 矩阵阶数
        a: 第一行元素，长度 n，要求 a[0] = 1 (已归一化)
        b: 右端项，长度 n

    返回:
        x: 解向量，长度 n
    """
    a = np.asarray(a, dtype=float).flatten()
    b = np.asarray(b, dtype=float).flatten()
    if a.size < n or b.size < n:
        raise ValueError("r8sto_sl: 输入数组长度不足")

    x = np.zeros(n)
    y = np.zeros(n)

    beta = 1.0
    x[0] = b[0] / beta
    if n > 1:
        y[0] = -a[1] / beta

    for k in range(1, n):
        beta = (1.0 - y[k - 1] * y[k - 1]) * beta
        if abs(beta) < 1e-14:
            raise RuntimeError(f"r8sto_sl: 在步骤 k={k} 处 β 接近零，矩阵可能不正定")

        # 计算 x[k]
        dot_ax = np.dot(a[1:k + 1], x[k - 1::-1])
        x[k] = (b[k] - dot_ax) / beta
        x[:k] = x[:k] + x[k] * y[k - 1::-1]

        if k < n - 1:
            dot_ay = np.dot(a[1:k + 1], y[k - 1::-1])
            y[k] = (-a[k + 1] - dot_ay) / beta
            y[:k] = y[:k] + y[k] * y[k - 1::-1]

    return x


def r8sto_yw_sl(n: int, a: np.ndarray) -> np.ndarray:
    """
    求解 Yule-Walker 方程 A y = -a[1:n]。

    参数:
        n: 矩阵阶数
        a: 第一行元素，长度 n

    返回:
        y: 反射系数向量
    """
    a = np.asarray(a, dtype=float).flatten()
    if a.size < n:
        raise ValueError("r8sto_yw_sl: 输入数组长度不足")

    y = np.zeros(n)
    beta = a[0]
    if abs(beta) < 1e-14:
        raise RuntimeError("r8sto_yw_sl: a[0] 为零")

    y[0] = -a[1] / beta
    for k in range(1, n - 1):
        beta = (1.0 - y[k - 1] * y[k - 1]) * beta
        dot_ay = np.dot(a[1:k + 1], y[k - 1::-1])
        y[k] = (-a[k + 1] - dot_ay) / beta
        y[:k] = y[:k] + y[k] * y[k - 1::-1]
    return y


def build_toeplitz_first_row(n: int, correlation_length: float = 5.0) -> np.ndarray:
    """
    构造指数衰减型Toeplitz矩阵的第一行。

    物理背景:
        海洋流速的时间自相关函数常呈指数衰减:
            R(τ) = exp(-|τ| / T_c)
        离散化后得到Toeplitz矩阵。

    公式:
        a[k] = exp(-k / correlation_length)

    参数:
        n: 矩阵阶数
        correlation_length: 相关长度

    返回:
        归一化的第一行元素 (a[0] = 1)
    """
    a = np.exp(-np.arange(n) / correlation_length)
    a[0] = 1.0
    return a


def solve_periodic_boundary_system(
    rhs: np.ndarray,
    correlation_length: float = 5.0,
) -> np.ndarray:
    """
    求解周期性边界条件对应的Toeplitz系统。

    参数:
        rhs: 右端项
        correlation_length: 相关长度参数

    返回:
        解向量
    """
    n = len(rhs)
    a = build_toeplitz_first_row(n, correlation_length)
    return r8sto_sl(n, a, rhs)
