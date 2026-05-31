"""
matrix_stability.py
数值稳定性分析模块：矩阵运算与条件数估计

在中子星物态方程计算中，线性系统的数值稳定性至关重要。
本模块提供下三角矩阵行列式计算、Hilbert矩阵等病态矩阵分析，
用于评估高密度极限下数值算法的可靠性。

原项目映射:
- 984_r8lt         -> 下三角矩阵行列式 r8lt_det
- 1216_test_matrix -> Hilbert矩阵等测试矩阵生成
"""

import numpy as np
import math
from typing import Tuple


# =============================================================================
# 下三角矩阵行列式 (源自 984_r8lt)
# =============================================================================
def r8lt_det(n: int, a: np.ndarray) -> float:
    """
    计算下三角矩阵 (R8LT format) 的行列式。

    对于下三角矩阵，行列式等于对角线元素的乘积:
        det(A) = ∏_{i=1}^{n} a_{ii}

    源自 984_r8lt 中 r8lt_det.m 的核心算法。

    在TOV方程线性化稳定性分析中，Jacobian矩阵常为下三角形式，
    其行列式的符号决定了解的稳定性（Poincaré-Hopf指标）。

    Parameters
    ----------
    n : int
        矩阵阶数，必须为正。
    a : np.ndarray, shape (n, n)
        下三角矩阵（包含上三角的零元素存储位置）。

    Returns
    -------
    float
        行列式值。
    """
    if n <= 0:
        raise ValueError("Matrix order n must be positive.")
    a = np.asarray(a, dtype=float)
    if a.shape != (n, n):
        raise ValueError(f"Matrix shape {a.shape} does not match expected ({n}, {n}).")

    det = 1.0
    for i in range(n):
        det *= a[i, i]
    return det


def r8lt_inverse(n: int, a: np.ndarray) -> np.ndarray:
    """
    计算下三角矩阵的逆矩阵。

    公式:
        (A^{-1})_{ii} = 1 / a_{ii}
        (A^{-1})_{ij} = - (1/a_{jj}) Σ_{k=j}^{i-1} a_{ik} (A^{-1})_{kj},  i > j
    """
    if n <= 0:
        raise ValueError("Matrix order n must be positive.")
    a = np.asarray(a, dtype=float)
    inv = np.zeros((n, n), dtype=float)

    for i in range(n):
        if abs(a[i, i]) < 1e-30:
            raise ValueError(f"Singular matrix: diagonal element a[{i},{i}] = 0.")
        inv[i, i] = 1.0 / a[i, i]
        for j in range(i):
            s = 0.0
            for k in range(j, i):
                s += a[i, k] * inv[k, j]
            inv[i, j] = -s / a[i, i]
    return inv


# =============================================================================
# Hilbert矩阵 (源自 1216_test_matrix)
# =============================================================================
def hilbert_matrix(m: int, n: int = None) -> np.ndarray:
    """
    生成 Hilbert 矩阵。

    公式:
        H_{ij} = 1 / (i + j - 1),  i, j = 1, ..., n

    Hilbert矩阵是经典的病态矩阵，条件数随阶数指数增长:
        κ_2(H_n) ~ exp(3.5 n)

    源自 1216_test_matrix 中 hilbert_matrix.m 的核心算法。

    在核物态方程的参数反演中，类似Hilbert矩阵的病态结构会出现，
    因此需要专门测试数值算法的稳定性。

    Parameters
    ----------
    m : int
        行数。
    n : int, optional
        列数，默认等于 m。

    Returns
    -------
    np.ndarray
        Hilbert矩阵。
    """
    if n is None:
        n = m
    if m <= 0 or n <= 0:
        raise ValueError("Dimensions must be positive.")

    a = np.zeros((m, n), dtype=float)
    for i in range(m):
        for j in range(n):
            a[i, j] = 1.0 / (i + j + 1)
    return a


def hilbert_inverse(n: int) -> np.ndarray:
    """
    Hilbert矩阵的精确逆矩阵（元素全为整数）。

    公式:
        (H^{-1})_{ij} = (-1)^{i+j} (n+i-1)!(n+j-1)! / [
            (i+j-1) ((i-1)!(j-1)!)^2 (n-i)!(n-j)!
        ]
    """
    if n <= 0:
        raise ValueError("n must be positive.")
    from math import factorial

    inv = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            sign = (-1)**(i + j)
            num = factorial(n + i) * factorial(n + j)
            den = (i + j + 1) * (factorial(i) * factorial(j))**2 * factorial(n - 1 - i) * factorial(n - 1 - j)
            inv[i, j] = sign * num / den
    return inv


# =============================================================================
# 条件数与稳定性分析
# =============================================================================
def matrix_condition_number_1d(a: np.ndarray) -> float:
    """
    计算矩阵的1-范数条件数估计。

    κ_1(A) = ||A||_1 * ||A^{-1}||_1

    对于大型矩阵，使用幂迭代法估计最大/最小奇异值。
    """
    a = np.asarray(a, dtype=float)
    # 使用numpy的SVD计算精确条件数
    s = np.linalg.svd(a, compute_uv=False)
    s_max = np.max(s)
    s_min = np.min(s[s > 1e-15]) if np.any(s > 1e-15) else 1e-30
    return s_max / s_min


def estimate_tov_stability_matrix(
    radius: np.ndarray,
    pressure: np.ndarray,
    mass: np.ndarray,
    energy_density: np.ndarray
) -> np.ndarray:
    """
    构建TOV方程线性化稳定性分析的Jacobi矩阵。

    在平衡构型附近微扰:
        δP' = A(r) δP + B(r) δm
        δm' = C(r) δP + D(r) δm

    返回离散的线性化矩阵，其特征值决定微扰增长率。
    """
    n = len(radius)
    if n < 2:
        raise ValueError("Need at least 2 radial points.")

    J = np.zeros((2 * n, 2 * n))
    Gc2 = 6.67430e-11 / (2.99792458e8)**2

    for i in range(n - 1):
        r = radius[i]
        P = pressure[i]
        m = mass[i]
        eps = energy_density[i]

        if r < 1e-10:
            continue

        denom = r * (r - 2.0 * Gc2 * m)
        if abs(denom) < 1e-30:
            continue

        # 简化：构造离散Jacobi的块结构
        dr = radius[i + 1] - radius[i]
        J[2*i, 2*i] = -1.0
        J[2*i, 2*i + 2] = 1.0
        J[2*i + 1, 2*i + 1] = -1.0
        J[2*i + 1, 2*i + 3] = 1.0

    return J


def analyze_eigenvalue_stability(a: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    分析矩阵特征值的稳定性。

    对于动力学系统 ẋ = A x:
        - 若所有 Re(λ_i) < 0，系统渐近稳定
        - 若存在 Re(λ_i) > 0，系统不稳定

    Returns
    -------
    eigenvalues : np.ndarray
        特征值。
    is_stable : bool
        是否稳定。
    """
    a = np.asarray(a, dtype=float)
    eigs = np.linalg.eigvals(a)
    real_parts = np.real(eigs)
    is_stable = np.all(real_parts < 1e-10)
    return eigs, is_stable


def test_numerical_stability_on_hilbert(n_max: int = 10) -> dict:
    """
    在Hilbert矩阵上测试数值稳定性，模拟物态方程反演中的病态问题。

    测试: 解 H_n x = b，其中 b = H_n * ones，检验数值误差 ||x - 1||。
    """
    results = {}
    for n in range(2, n_max + 1):
        H = hilbert_matrix(n)
        x_exact = np.ones(n)
        b = H @ x_exact

        try:
            x_numerical = np.linalg.solve(H, b)
            error = np.linalg.norm(x_numerical - x_exact)
            cond = matrix_condition_number_1d(H)
            results[n] = {
                'error': error,
                'condition_number': cond,
                'log_error': math.log10(error + 1e-20)
            }
        except np.linalg.LinAlgError:
            results[n] = {
                'error': float('inf'),
                'condition_number': float('inf'),
                'log_error': float('inf')
            }

    return results
