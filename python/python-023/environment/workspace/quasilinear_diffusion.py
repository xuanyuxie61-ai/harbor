#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
准线性扩散算子组装器
================================================================================

基于 853_pce_legendre 的随机Galerkin有限元矩阵组装思想，
将速度空间的准线性扩散系数组装为稀疏线性算子矩阵。

核心物理模型：

准线性扩散张量（ Kennel & Engelmann 1966 ）：

对于 whistler 模与电子的回旋共振 (n = -1, 电子回旋阻尼)：

    D_{∥∥}^{QL} = Σ_k (π q_e² / m_e²) |E_k|² J_1²(k_⊥ v_⊥ / Ω_e)
                  × δ(ω_k - k_∥ v_∥ - Ω_e / γ) × (1 - k_∥ v_∥ / ω_k)²

    D_{⊥⊥}^{QL} = Σ_k (π q_e² / m_e²) |E_k|² [J_0'(x_e)]²
                  × δ(ω_k - k_∥ v_∥ - Ω_e / γ) × (k_∥ v_⊥ / ω_k)²

    D_{∥⊥}^{QL} = Σ_k (π q_e² / m_e²) |E_k|² J_1(x_e) J_0'(x_e)
                  × δ(ω_k - k_∥ v_∥ - Ω_e / γ)
                  × (1 - k_∥ v_∥ / ω_k) × (k_∥ v_⊥ / ω_k)

其中 x_e = k_⊥ v_⊥ / Ω_e，J_n 为Bessel函数。

δ函数的宽化（准线性近似）：
    δ(x) → 1/(√π Δv) exp(-x²/Δv²)

速度空间扩散方程的离散形式：
    ∂f/∂t = D_{∥∥} ∂²f/∂v_∥² + D_{⊥⊥} (1/v_⊥) ∂/∂v_⊥ (v_⊥ ∂f/∂v_⊥)
          + 2 D_{∥⊥} ∂²f/∂v_∥∂v_⊥

有限差分离散（中心差分）：
    ∂²f/∂v_∥² ≈ (f_{i+1,j} - 2f_{i,j} + f_{i-1,j}) / Δv_∥²
    (1/v_⊥) ∂/∂v_⊥ (v_⊥ ∂f/∂v_⊥) ≈ 
        [v_{⊥,j+1/2}(f_{i,j+1}-f_{i,j}) - v_{⊥,j-1/2}(f_{i,j}-f_{i,j-1})]
        / (v_{⊥,j} Δv_⊥²)
================================================================================
"""

import numpy as np
from scipy.special import jv


def compute_ql_diffusion_coefficients(v_parallel, v_perp, omega_solutions, params):
    """
    计算速度空间的准线性扩散系数 D_{∥∥}, D_{⊥⊥}, D_{∥⊥}。
    
    参数
    ----
    v_parallel : ndarray, shape (nv,)
        平行速度网格 [m/s]。
    v_perp : ndarray, shape (nv,)
        垂直速度网格 [m/s]。
    omega_solutions : ndarray, shape (Nk, 2)
        (k, ω) 色散解。
    params : dict
        物理参数。
        
    返回
    ----
    D_par : ndarray, shape (nv, nv)
        D_{∥∥} 在 (v_∥, v_⊥) 网格上。
    D_perp : ndarray, shape (nv, nv)
        D_{⊥⊥} 在 (v_∥, v_⊥) 网格上。
    D_cross : ndarray, shape (nv, nv)
        D_{∥⊥} 在 (v_∥, v_⊥) 网格上。
    """
    # TODO: 实现准线性扩散系数计算
    # 需要计算 D_{∥∥}, D_{⊥⊥}, D_{∥⊥} 三个扩散系数
    # 核心物理：Kennel & Engelmann 1966 准线性扩散张量
    # 涉及 Bessel 函数、共振条件宽化、投影因子等
    # 
    # 提示：
    # 1. 对每支波模式 (k, ω) 求和
    # 2. 计算共振条件：ω_r - k_∥ v_∥ - Ω_e/γ = 0
    # 3. 使用宽化的 δ 函数近似
    # 4. 计算投影因子 P_par = (1 - k_∥ v_∥ / ω_r), P_perp = (k_∥ v_⊥ / ω_r)
    # 5. 组装 D_par, D_perp, D_cross
    
    nv = len(v_parallel)
    D_par = np.zeros((nv, nv), dtype=np.float64)
    D_perp = np.zeros((nv, nv), dtype=np.float64)
    D_cross = np.zeros((nv, nv), dtype=np.float64)
    
    return D_par, D_perp, D_cross


def assemble_ql_diffusion_matrix(v_parallel, v_perp, omega_solutions, params,
                                  n_stochastic=2, p_degree=2):
    """
    组装准线性扩散算子的稀疏矩阵。
    
    基于 853_pce_legendre 的随机Galerkin矩阵组装：
    将 (v_∥, v_⊥) 二维离散与多项式混沌展开结合，
    得到大稀疏线性系统 A f = rhs。
    
    参数
    ----
    v_parallel, v_perp : ndarray
        速度网格。
    omega_solutions : ndarray
        色散解。
    params : dict
        物理参数。
    n_stochastic : int
        随机维度。
    p_degree : int
        PCE阶数。
        
    返回
    ----
    A : ndarray, shape (N, N)
        扩散算子矩阵（稠密，用于小系统）。
    rhs : ndarray, shape (N,)
        初始条件向量。
    """
    nv = len(v_parallel)
    
    # PCE维度
    from pce_expansion import enumerate_multi_indices
    indices = enumerate_multi_indices(n_stochastic, p_degree)
    M_pce = len(indices)
    
    # 总维度
    N_total = nv * nv * M_pce
    
    # 为可行性，使用简化版本：仅空间离散，PCE退化为标量乘性因子
    # 计算扩散系数
    D_par, D_perp, D_cross = compute_ql_diffusion_coefficients(
        v_parallel, v_perp, omega_solutions, params
    )
    
    dv_par = v_parallel[1] - v_parallel[0]
    dv_perp = v_perp[1] - v_perp[0]
    
    # TODO: 组装准线性扩散算子矩阵和初始条件
    # 需要将扩散系数 D_par, D_perp, D_cross 组装为有限差分矩阵 A
    # 并构造初始条件向量 rhs
    # 
    # 提示：
    # 1. 使用中心差分离散 ∂²/∂v_∥² 和 (1/v_⊥)∂/∂v_⊥(v_⊥∂/∂v_⊥)
    # 2. 处理 v_⊥ = 0 处的奇异性
    # 3. 交叉项使用混合差分
    # 4. rhs 为 Maxwellian 初始分布
    # 5. 注意返回格式需要与调用方匹配
    
    N = nv * nv
    A = np.zeros((N, N))
    rhs = np.zeros(N)
    
    return A, rhs
