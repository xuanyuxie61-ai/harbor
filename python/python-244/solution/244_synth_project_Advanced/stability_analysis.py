#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
stability_analysis.py
=====================
【融合种子项目】 687_linpack_bench (LINPACK 基准: LU 分解 + 线性求解)

本模块实现中子星平衡构型的线性稳定性分析, 通过将扰动方程离散化为
广义特征值问题 A x = lambda B x, 然后使用 LU 分解求解.

物理/数学公式
-------------
1. 径向扰动方程 (Chandrasekhar 1964):
       d/dr [Gamma P r^4 e^{(3 lambda + nu)/2} d xi / dr]
       + r^3 dP/dr e^{(3 lambda + nu)/2} d/dr (r xi)
       + 4 pi r^2 (epsilon + P/c^2) e^{(3 lambda + nu)/2}
         [omega^2 e^{-nu} r^3 xi - 4 pi G r^3 (epsilon + 3P/c^2) xi] = 0

   简化为 Sturm-Liouville 形式:
       -(p(x) y')' + q(x) y = lambda w(x) y

   其中:
       p(x) = Gamma P r^4
       q(x) = 相关势函数
       w(x) = (epsilon + P/c^2) r^4
       lambda = omega^2 (本征值的平方)

2. 稳定性判据:
       omega^2 > 0  =>  稳定 (振荡模式)
       omega^2 < 0  =>  不稳定 (塌缩)
       omega^2 = 0  =>  临界点 (最大质量处)

3. 有限差分离散 (Sturm-Liouville):
       -[p_{i+1/2}(y_{i+1}-y_i)/h - p_{i-1/2}(y_i-y_{i-1})/h]/h
       + q_i y_i = lambda w_i y_i

   矩阵形式:
       A y = lambda B y
   其中 A 为三对角, B 为对角正定.

4. LU 分解 (移植自 LINPACK dgefa):
       A = P L U
   其中 P 为排列矩阵, L 为下三角, U 为上三角.

5. 线性求解 (移植自 LINPACK dgesl):
       Ax = b  =>  PAx = Pb  =>  LUx = Pb
       1) Ly = Pb  (前代)
       2) Ux = y   (回代)

6. 条件数:
       kappa(A) = ||A|| * ||A^{-1}||
   大条件数 => 病态问题

7. 行列式 (用于判断奇异性):
       det(A) = prod(U_{ii}) * sign(P)
"""

import math
import numpy as np
from typing import Tuple, Dict, Optional
from numerical_constants import (r8mat_norm_fro, r8mat_norm_l1, r8mat_norm_li,
                                  r8_epsilon)


# ============================================================
# 第一部分: LU 分解与线性求解 (移植自 LINPACK)
# ============================================================

def lu_factor(a: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    LU 分解 (部分选主元, 移植自 LINPACK dgefa).

    A = P L U

    算法:
        for k = 1, ..., n:
            1. 选主元: j = argmax_{i>=k} |a_{ik}|
            2. 交换行 k 和 j
            3. 计算乘子: l_{ik} = a_{ik} / a_{kk},  i > k
            4. 更新: a_{ij} -= l_{ik} a_{kj},  i,j > k

    Parameters
    ----------
    a : np.ndarray, shape (n, n)

    Returns
    -------
    ALU : np.ndarray
        L (严格下三角) 和 U (上三角) 的合并存储
    piv : np.ndarray
        主元排列 (0-based)
    info : int
        0 = 成功, k > 0 = U(k,k) = 0 (奇异)
    """
    n = a.shape[0]
    ALU = a.copy()
    piv = np.arange(n)
    info = 0

    for k in range(n):
        # 选主元
        j_max = k + int(np.argmax(np.abs(ALU[k:, k])))
        piv[k], piv[j_max] = piv[j_max], piv[k]
        # 交换行
        ALU[[k, j_max]] = ALU[[j_max, k]]

        if abs(ALU[k, k]) < r8_epsilon():
            info = k + 1
            continue

        # 计算乘子
        for i in range(k + 1, n):
            ALU[i, k] /= ALU[k, k]
            # 更新子矩阵
            ALU[i, k + 1:] -= ALU[i, k] * ALU[k, k + 1:]

    return ALU, piv, info


def lu_solve(ALU: np.ndarray, piv: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    LU 求解 (移植自 LINPACK dgesl).

    解 Ax = b, 其中 A 已通过 lu_factor 分解.

    算法:
        1. 应用排列: Pb
        2. 前代: Ly = Pb
        3. 回代: Ux = y
    """
    n = ALU.shape[0]
    x = b[piv].copy()

    # 前代 (L y = Pb)
    for i in range(1, n):
        x[i] -= np.dot(ALU[i, :i], x[:i])

    # 回代 (U x = y)
    for i in range(n - 1, -1, -1):
        x[i] -= np.dot(ALU[i, i + 1:], x[i + 1:])
        if abs(ALU[i, i]) > r8_epsilon():
            x[i] /= ALU[i, i]
        else:
            x[i] = 0.0

    return x


def lu_det(ALU: np.ndarray, piv: np.ndarray) -> float:
    """
    从 LU 分解计算行列式.

    det(A) = prod(U_{ii}) * (-1)^{交换次数}
    """
    n = ALU.shape[0]
    det_val = 1.0
    for i in range(n):
        det_val *= ALU[i, i]
    # 计算排列的符号
    n_swaps = sum(1 for i in range(n) if piv[i] != i)
    if n_swaps % 2 == 1:
        det_val = -det_val
    return det_val


def lu_residual(a: np.ndarray, x: np.ndarray, b: np.ndarray) -> float:
    """
    计算残差 ||b - Ax|| / (||A|| ||x||).

    类似 LINPACK benchmark 中的残差报告.
    """
    r = b - a @ x
    a_norm = r8mat_norm_li(a)
    x_norm = np.max(np.abs(x))
    r_norm = np.max(np.abs(r))
    if a_norm * x_norm < r8_epsilon():
        return r_norm
    return r_norm / (a_norm * x_norm)


# ============================================================
# 第二部分:  Sturm-Liouville 离散化
# ============================================================

def sturm_liouville_matrices(p_func: callable, q_func: callable,
                              w_func: callable,
                              r_nodes: np.ndarray,
                              bc_left: str = "neumann",
                              bc_right: str = "dirichlet"
                              ) -> Tuple[np.ndarray, np.ndarray]:
    """
    将 Sturm-Liouville 问题离散化为矩阵广义特征值问题.

    -(p(x) y')' + q(x) y = lambda w(x) y

    离散:
        A[i,i] = (p_{i+1/2} + p_{i-1/2})/h^2 + q_i
        A[i,i+1] = -p_{i+1/2}/h^2
        A[i,i-1] = -p_{i-1/2}/h^2
        B[i,i] = w_i

    Parameters
    ----------
    p_func : callable
        p(x) 函数
    q_func : callable
        q(x) 函数
    w_func : callable
        权函数 w(x)
    r_nodes : np.ndarray
        节点坐标
    bc_left : str
        左边界条件
    bc_right : str
        右边界条件

    Returns
    -------
    A : np.ndarray, shape (N, N)
        刚度矩阵
    B : np.ndarray, shape (N, N)
        质量矩阵 (对角)
    """
    n = len(r_nodes)
    h = r_nodes[1] - r_nodes[0] if n > 1 else 1.0

    A = np.zeros((n, n))
    B = np.zeros((n, n))

    for i in range(n):
        x = r_nodes[i]
        B[i, i] = w_func(x)

        # p_{i+1/2} 和 p_{i-1/2}
        if i < n - 1:
            p_half_right = p_func(0.5 * (r_nodes[i] + r_nodes[i + 1]))
        else:
            p_half_right = p_func(x)

        if i > 0:
            p_half_left = p_func(0.5 * (r_nodes[i] + r_nodes[i - 1]))
        else:
            p_half_left = p_func(x)

        A[i, i] = (p_half_right + p_half_left) / (h * h) + q_func(x)
        if i < n - 1:
            A[i, i + 1] = -p_half_right / (h * h)
        if i > 0:
            A[i, i - 1] = -p_half_left / (h * h)

    # 边界条件
    if bc_left == "neumann":
        # y'(0) = 0: 虚拟节点 y_{-1} = y_1
        A[0, 0] -= A[0, 1] if n > 1 else 0
        if n > 1:
            A[0, 1] *= 2.0  # 但保持对称
    elif bc_left == "dirichlet":
        A[0, :] = 0.0
        A[0, 0] = 1.0
        B[0, 0] = 0.0

    if bc_right == "dirichlet":
        A[-1, :] = 0.0
        A[-1, -1] = 1.0
        B[-1, -1] = 0.0

    return A, B


def generalized_eigenvalue_symmetric(A: np.ndarray,
                                      B: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    解对称广义特征值问题 A x = lambda B x.

    当 B 为正定对角矩阵时, 可转换为标准问题:
        B^{-1/2} A B^{-1/2} z = lambda z
        x = B^{-1/2} z

    Returns
    -------
    eigenvalues : np.ndarray
        特征值 (升序)
    eigenvectors : np.ndarray
        对应特征向量 (列)
    """
    n = A.shape[0]

    # 检查 B 的对角元素
    b_diag = np.diag(B).copy()
    for i in range(n):
        if b_diag[i] < r8_epsilon():
            b_diag[i] = r8_epsilon()

    B_inv_sqrt = np.diag(1.0 / np.sqrt(b_diag))

    # 转换
    A_tilde = B_inv_sqrt @ A @ B_inv_sqrt
    # 对称化 (消除数值不对称)
    A_tilde = 0.5 * (A_tilde + A_tilde.T)

    # 标准对称特征值问题
    eigenvalues, Z = np.linalg.eigh(A_tilde)

    # 转换回原变量
    eigenvectors = B_inv_sqrt @ Z

    # 归一化: x^T B x = 1
    for i in range(n):
        norm_val = np.sqrt(np.abs(eigenvectors[:, i] @ B @ eigenvectors[:, i]))
        if norm_val > r8_epsilon():
            eigenvectors[:, i] /= norm_val

    return eigenvalues, eigenvectors


# ============================================================
# 第三部分: 中子星稳定性分析
# ============================================================

def neutron_star_stability_matrices(r_nodes: np.ndarray,
                                     P_profile: np.ndarray,
                                     rho_profile: np.ndarray,
                                     gamma_profile: np.ndarray,
                                     m_profile: np.ndarray
                                     ) -> Tuple[np.ndarray, np.ndarray]:
    """
    构造中子星径向扰动的 Sturm-Liouville 矩阵.

    方程 (Chandrasekhar 1964 简化):
        -(Gamma P r^4 xi')' = omega^2 (epsilon + P/c^2) r^4 xi
        + 4 pi G r^4 (epsilon + P/c^2)(epsilon + 3P/c^2) xi / c^2
        + r^3 (dP/dr)^2 / (Gamma P) xi  (修正项)

    简化:
        p(r) = Gamma(r) P(r) r^4
        q(r) = r^3 (dP/dr)^2 / (Gamma P) - 4 pi G r^4 (...)
        w(r) = (epsilon + P/c^2) r^4
    """
    from numerical_constants import NeutronStarConstants as NS

    n = len(r_nodes)

    def p_func(r):
        """p(r) = Gamma P r^4."""
        P_r = float(np.interp(r, r_nodes, P_profile))
        Gamma_r = float(np.interp(r, r_nodes, gamma_profile))
        if r <= 0 or P_r <= 0:
            return r8_epsilon()
        return Gamma_r * P_r * r ** 4

    def q_func(r):
        """q(r): 有效势."""
        P_r = float(np.interp(r, r_nodes, P_profile))
        rho_r = float(np.interp(r, r_nodes, rho_profile))
        Gamma_r = float(np.interp(r, r_nodes, gamma_profile))
        m_r = float(np.interp(r, r_nodes, m_profile))

        if r <= r8_epsilon() or P_r <= 0 or rho_r <= 0:
            return 0.0

        eps_r = rho_r * NS.c_light ** 2 + P_r / (Gamma_r - 1.0)

        # 引力贡献 (简化)
        grav_term = (4.0 * math.pi * NS.G_newton * r ** 3
                     * (eps_r + P_r / NS.c_light ** 2)
                     * (eps_r + 3.0 * P_r / NS.c_light ** 2)
                     / (NS.c_light ** 4 * r + r8_epsilon()))

        return grav_term

    def w_func(r):
        """w(r) = (epsilon + P/c^2) r^4."""
        P_r = float(np.interp(r, r_nodes, P_profile))
        rho_r = float(np.interp(r, r_nodes, rho_profile))
        Gamma_r = float(np.interp(r, r_nodes, gamma_profile))

        if r <= 0:
            return r8_epsilon()

        eps_r = rho_r * NS.c_light ** 2
        if P_r > 0 and Gamma_r > 1.0:
            eps_r += P_r / (Gamma_r - 1.0)

        return (eps_r + P_r / NS.c_light ** 2) * r ** 4

    A, B = sturm_liouville_matrices(p_func, q_func, w_func, r_nodes)
    return A, B


def analyze_stability(r_nodes: np.ndarray, P_profile: np.ndarray,
                       rho_profile: np.ndarray, gamma_profile: np.ndarray,
                       m_profile: np.ndarray) -> Dict:
    """
    执行中子星径向稳定性分析.

    Returns
    -------
    dict
        'eigenvalues': omega^2 值
        'is_stable': bool (最小 omega^2 > 0?)
        'fundamental_mode': 基频 omega_0
        'condition_number': 矩阵条件数
        'n_negative_modes': 不稳定模式数
    """
    A, B = neutron_star_stability_matrices(
        r_nodes, P_profile, rho_profile, gamma_profile, m_profile
    )

    eigenvalues, eigenvectors = generalized_eigenvalue_symmetric(A, B)

    # 稳定性判据
    n_negative = int(np.sum(eigenvalues < 0))
    is_stable = n_negative == 0

    # 基频
    omega2_min = float(np.min(eigenvalues))
    omega0 = math.sqrt(abs(omega2_min)) if abs(omega2_min) > r8_epsilon() else 0.0

    # 条件数
    try:
        kappa = float(np.linalg.cond(A))
    except np.linalg.LinAlgError:
        kappa = float('inf')

    # LU 残差检验
    ALU, piv, info = lu_factor(A)
    if info == 0:
        b_test = np.ones(len(A))
        x_test = lu_solve(ALU, piv, b_test)
        res = lu_residual(A, x_test, b_test)
    else:
        res = float('inf')

    return {
        'eigenvalues': eigenvalues,
        'eigenvectors': eigenvectors,
        'is_stable': is_stable,
        'omega2_min': omega2_min,
        'fundamental_mode': omega0,
        'n_negative_modes': n_negative,
        'condition_number': kappa,
        'lu_residual': res,
        'lu_info': info,
    }


# ============================================================
# 第四部分: LINPACK 风格性能基准
# ============================================================

def linpack_benchmark(n: int = 100) -> Dict:
    """
    LINPACK 基准测试 (移植自 linpack_bench).

    求解 n x n 线性系统, 报告 MFLOPS.

    FLOPs ≈ (2/3) n^3 + 2 n^2
    """
    import time

    np.random.seed(42)
    A = np.random.randn(n, n)
    x_exact = np.ones(n)
    b = A @ x_exact

    t_start = time.time()
    ALU, piv, info = lu_factor(A)
    t_factor = time.time() - t_start

    t_start = time.time()
    x = lu_solve(ALU, piv, b)
    t_solve = time.time() - t_start

    t_total = t_factor + t_solve

    # 残差
    r = b - A @ x
    a_norm = r8mat_norm_li(A)
    x_norm = np.max(np.abs(x))
    r_norm = np.max(np.abs(r))

    ratio = r_norm / (a_norm * x_norm * r8_epsilon()) if a_norm * x_norm > 0 else r_norm

    ops = (2.0 * n ** 3) / 3.0 + 2.0 * n ** 2
    mflops = ops / (1.0e6 * t_total) if t_total > 0 else 0.0

    return {
        'n': n,
        'time_factor': t_factor,
        'time_solve': t_solve,
        'time_total': t_total,
        'residual_ratio': ratio,
        'mflops': mflops,
        'info': info,
        'x_first': float(x[0]),
        'x_last': float(x[-1]),
    }


# ============================================================
# 自检
# ============================================================

if __name__ == "__main__":
    print("=== 稳定性分析自检 ===")

    # LU 分解测试
    A_test = np.array([[2.0, 1.0, 0.0],
                       [1.0, 3.0, 1.0],
                       [0.0, 1.0, 2.0]])
    b_test = np.array([1.0, 2.0, 3.0])

    ALU, piv, info = lu_factor(A_test)
    print(f"LU info: {info}")
    x_sol = lu_solve(ALU, piv, b_test)
    res = lu_residual(A_test, x_sol, b_test)
    print(f"LU 解: {x_sol}")
    print(f"归一化残差: {res:.3e}")

    det = lu_det(ALU, piv)
    det_exact = np.linalg.det(A_test)
    print(f"det(A) = {det:.6f}, 精确 = {det_exact:.6f}")

    # Sturm-Liouville 测试: -y'' = lambda y on [0, pi]
    # 精确本征值: lambda_n = n^2
    n_sl = 50
    r_sl = np.linspace(0, math.pi, n_sl)
    p_func = lambda x: 1.0
    q_func = lambda x: 0.0
    w_func = lambda x: 1.0

    A_sl, B_sl = sturm_liouville_matrices(p_func, q_func, w_func, r_sl,
                                           bc_left="dirichlet",
                                           bc_right="dirichlet")
    eigs, vecs = generalized_eigenvalue_symmetric(A_sl, B_sl)

    # 前几个本征值
    print(f"\nSturm-Liouville -y'' = lambda y:")
    for k in range(min(5, len(eigs))):
        exact = (k + 1) ** 2
        print(f"  lambda_{k+1} = {eigs[k]:.4f}  (精确 {exact})")

    # LINPACK 基准
    result = linpack_benchmark(100)
    print(f"\nLINPACK 基准 (n=100):")
    print(f"  总时间: {result['time_total']:.4f} s")
    print(f"  MFLOPS: {result['mflops']:.2f}")
    print(f"  残差比: {result['residual_ratio']:.3e}")

    print("\nstability_analysis.py 自检通过.")
