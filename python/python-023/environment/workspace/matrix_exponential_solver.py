#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
矩阵指数时间演化求解器
================================================================================

基于 741_matrix_exponential 的Pade逼近算法，精确计算扩散算子的
矩阵指数，用于准线性Fokker-Planck方程的时间推进。

核心数学模型：

Fokker-Planck方程的半离散形式：
    df/dt = A f
    
    其中 A 为准线性扩散算子（已在 assemble_ql_diffusion_matrix 中组装）。

精确解：
    f(t + Δt) = exp(A Δt) f(t)

矩阵指数 exp(M) 的Pade逼近 (Moler & Van Loan 2003) ：

1. 缩放：找到 s 使得 ||M/2^s||_∞ ≤ 1/2
2. 对角Pade逼近（阶数 q=6）：
       exp(M/2^s) ≈ D^{-1} E
   其中
       E = I + c_1 (M/2^s) + c_2 (M/2^s)^2 + ... + c_q (M/2^s)^q
       D = I - c_1 (M/2^s) + c_2 (M/2^s)^2 - ... ± c_q (M/2^s)^q
       c_k = (q-k+1)! q! / [ (2q-k+1)! k! (q-k)! ]
3. 平方：exp(M) = [exp(M/2^s)]^{2^s}

稳定性分析：
    对于扩散算子 A（负实部特征值），exp(A Δt) 的所有特征值满足
    |exp(λ_i Δt)| ≤ 1，保证数值稳定性。

对于大矩阵，使用Krylov子空间方法：
    exp(A) v ≈ ||v|| V_m exp(H_m) e_1
    其中 V_m 为Krylov子空间基，H_m 为上Hessenberg矩阵。
================================================================================
"""

import numpy as np


def matrix_exponential_pade(A):
    """
    使用Pade逼近计算矩阵指数 exp(A)。
    
    基于 741_matrix_exponential 的 r8mat_expm1 算法。
    
    参数
    ----
    A : ndarray, shape (n, n)
        输入矩阵。
        
    返回
    ----
    E : ndarray, shape (n, n)
        exp(A) 的近似。
    """
    n = A.shape[0]
    
    # 缩放因子 s
    inf_norm = np.linalg.norm(A, ord=np.inf)
    if inf_norm < 1e-30:
        return np.eye(n)
    
    # s = max(0, floor(log2(inf_norm)) + 1)
    s = max(0, int(np.log2(inf_norm)) + 1)
    
    # 缩放矩阵
    A_scaled = A / (2.0 ** s)
    
    # Pade逼近阶数 q = 6
    q = 6
    
    # 构建分子 E 和分母 D
    I = np.eye(n)
    
    X = A_scaled.copy()
    c = 0.5
    E = I + c * A_scaled
    D = I - c * A_scaled
    
    p = True  # 交替符号
    
    for k in range(2, q + 1):
        # c_k = c_{k-1} * (q - k + 1) / [k * (2q - k + 1)]
        c = c * (q - k + 1) / (k * (2 * q - k + 1))
        X = A_scaled @ X
        cX = c * X
        E = E + cX
        if p:
            D = D + cX
        else:
            D = D - cX
        p = not p
    
    # 解线性系统 D E_result = E
    try:
        E_result = np.linalg.solve(D, E)
    except np.linalg.LinAlgError:
        # D 病态，使用伪逆
        E_result = np.linalg.lstsq(D, E, rcond=None)[0]
    
    # 平方 s 次
    for _ in range(s):
        E_result = E_result @ E_result
    
    return E_result


def arnoldi_iteration(A, b, m):
    """
    Arnoldi迭代构建Krylov子空间。
    
    输出：
        V: 正交基，shape (n, m+1)
        H: 上Hessenberg矩阵，shape (m+1, m)
    """
    n = len(b)
    V = np.zeros((n, m + 1))
    H = np.zeros((m + 1, m))
    
    V[:, 0] = b / np.linalg.norm(b)
    
    for j in range(m):
        w = A @ V[:, j]
        
        for i in range(j + 1):
            H[i, j] = V[:, i] @ w
            w = w - H[i, j] * V[:, i]
        
        H[j + 1, j] = np.linalg.norm(w)
        
        if H[j + 1, j] < 1e-30:
            H[j + 1, j] = 1e-30
        
        V[:, j + 1] = w / H[j + 1, j]
    
    return V, H


def expm_krylov(A, v, m=20):
    """
    使用Krylov子空间近似计算 exp(A) v。
    
    参数
    ----
    A : ndarray
        矩阵。
    v : ndarray
        向量。
    m : int
        Krylov子空间维度。
        
    返回
    ----
    result : ndarray
        exp(A) v 的近似。
    """
    n = len(v)
    v_norm = np.linalg.norm(v)
    if v_norm < 1e-30:
        return np.zeros(n)
    
    V, H = arnoldi_iteration(A, v, m)
    
    # 计算 exp(H_m) e_1
    e1 = np.zeros(m)
    e1[0] = 1.0
    
    # 对小的Hessenberg矩阵使用Pade
    exp_H = matrix_exponential_pade(H[:m, :m])
    
    result = v_norm * V[:, :m] @ (exp_H @ e1)
    
    return result


def evolve_diffusion_operator(A, f0, dt, n_steps, use_krylov=False):
    """
    使用矩阵指数推进扩散方程 df/dt = A f。
    
    参数
    ----
    A : ndarray
        扩散算子矩阵。
    f0 : ndarray
        初始条件。
    dt : float
        时间步长。
    n_steps : int
        推进步数。
    use_krylov : bool
        是否使用Krylov子空间方法。
        
    返回
    ----
    f_final : ndarray
        最终分布函数。
    """
    # TODO: 实现扩散算子的时间演化推进
    # 使用矩阵指数或Krylov子空间方法推进 df/dt = A f
    # 
    # 提示：
    # 1. 根据矩阵大小选择方法（n > 100 用 Krylov，否则用 Pade 逼近）
    # 2. 计算 exp(A * dt) 或直接应用矩阵指数
    # 3. 进行 n_steps 步推进
    # 4. 加入边界检查（非有限值回退）和范数监控
    # 5. 注意输入参数 A 和 f0 的格式与上游模块匹配
    
    n = len(f0)
    f = f0.copy()
    
    return f
