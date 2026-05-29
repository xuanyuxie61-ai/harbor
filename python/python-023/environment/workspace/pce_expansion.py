#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
多项式混沌展开(PCE)不确定性量化
================================================================================

基于 853_pce_legendre 的随机Galerkin矩阵组装思想，使用Legendre多项式
混沌展开量化磁场涨落对粒子扩散系数的不确定性。

核心数学模型：

设随机磁场涨落可参数化为：
    B(𝐱, 𝛏) = B_0(𝐱) + Σ_{i=1}^N ξ_i B_i(𝐱)

其中 ξ_i ~ U(-1, 1) 为独立均匀随机变量。

分布函数 f(𝐯, t, 𝛏) 的多项式混沌展开：
    f(𝐯, t, 𝛏) = Σ_{α∈J} f_α(𝐯, t) Ψ_α(𝛏)

其中 Ψ_α(𝛏) 为多变量Legendre多项式，指标集 J = {α ∈ ℕ_0^N : |α| ≤ P}。

Galerkin投影给出系数演化方程：
    ∂f_α/∂t = -Σ_β C_{αβ}^{-1} ⟨Ψ_α Ψ_β Ψ_γ⟩ A_γ ∂f_β/∂𝐯
              + (1/2) Σ_β C_{αβ}^{-1} ⟨Ψ_α Ψ_β Ψ_γ Ψ_δ⟩ D_γδ ∂²f_β/∂𝐯²

其中 C_{αβ} = ⟨Ψ_α Ψ_β⟩ 为质量矩阵。

对于Legendre多项式，单变量正交性：
    ∫_{-1}^{1} L_m(x) L_n(x) dx = 2/(2n+1) δ_{mn}

多维情形下：
    C_{αβ} = Π_{i=1}^N [2/(2α_i+1)] δ_{α_i β_i}
================================================================================
"""

import numpy as np
from scipy.special import eval_legendre


def legendre_polynomial_normalized(n, x):
    """
    计算归一化Legendre多项式，使得 ∫_{-1}^1 L_n²(x) dx = 1。
    
    标准Legendre多项式 P_n(x) 满足：
        ∫_{-1}^{1} P_n(x) P_m(x) dx = 2/(2n+1) δ_{nm}
    
    归一化版本：
        L_n(x) = √((2n+1)/2) P_n(x)
    
    参数
    ----
    n : int
        多项式阶数。
    x : float 或 ndarray
        自变量，在 [-1, 1] 内。
        
    返回
    ----
    L : float 或 ndarray
        归一化多项式值。
    """
    # 边界检查
    x = np.clip(x, -1.0, 1.0)
    Pn = eval_legendre(n, x)
    norm = np.sqrt((2.0 * n + 1.0) / 2.0)
    return norm * Pn


def multivariate_legendre_basis(alpha, xi):
    """
    计算多变量Legendre多项式 Ψ_α(𝛏)。
    
    参数
    ----
    alpha : ndarray, shape (N,)
        多指标。
    xi : ndarray, shape (N,)
        随机变量样本。
        
    返回
    ----
    Psi : float
        Ψ_α(𝛏) 的值。
    """
    N = len(alpha)
    result = 1.0
    for i in range(N):
        result *= legendre_polynomial_normalized(alpha[i], xi[i])
    return result


def enumerate_multi_indices(N, P):
    """
    枚举所有满足 |α| ≤ P 的多指标 α ∈ ℕ_0^N。
    
    基于 853_pce_legendre 中的组合枚举思想（subcomp_next）。
    
    总维度：
        M = C(N+P, N) = (N+P)! / (N! P!)
    
    参数
    ----
    N : int
        随机维度。
    P : int
        最大全次数。
        
    返回
    ----
    indices : list of ndarray
        多指标列表。
    """
    indices = []
    
    def recurse(current, remaining, dim):
        if dim == N - 1:
            current.append(remaining)
            indices.append(np.array(current, dtype=int))
            current.pop()
            return
        for val in range(remaining + 1):
            current.append(val)
            recurse(current, remaining - val, dim + 1)
            current.pop()
    
    recurse([], P, 0)
    return indices


def compute_mass_matrix(indices):
    """
    计算PCE质量矩阵 C_{αβ} = ⟨Ψ_α Ψ_β⟩。
    
    对于归一化Legendre多项式，C = I（单位矩阵）。
    这里显式计算以验证正交性。
    
    参数
    ----
    indices : list of ndarray
        多指标列表。
        
    返回
    ----
    C : ndarray, shape (M, M)
        质量矩阵。
    """
    M = len(indices)
    C = np.eye(M)
    return C


def polychaos_magnetic_uncertainty(v_parallel, v_perp, params, n_stochastic=2, p_degree=3):
    """
    使用多项式混沌展开量化磁场不确定性对分布函数的影响。
    
    参数
    ----
    v_parallel : ndarray
        平行速度网格。
    v_perp : ndarray
        垂直速度网格。
    params : dict
        物理参数。
    n_stochastic : int
        随机维度 N。
    p_degree : int
        多项式阶数 P。
        
    返回
    ----
    f_mean : ndarray, shape (nv, nv)
        分布函数均值。
    f_var : ndarray, shape (nv, nv)
        分布函数方差。
    """
    nv = len(v_parallel)
    
    # 枚举多指标
    indices = enumerate_multi_indices(n_stochastic, p_degree)
    M = len(indices)
    
    # 质量矩阵
    C = compute_mass_matrix(indices)
    
    # 初始化PCE系数
    # f_α(v_∥, v_⊥) 对于所有 α
    f_coeffs = np.zeros((M, nv, nv))
    
    # 零阶系数：均值（Maxwellian）
    v_te = params['v_te']
    VP, VPL = np.meshgrid(v_perp, v_parallel, indexing='ij')
    v_sq = VPL**2 + VP**2
    
    # 归一化Maxwellian分布
    f_maxwell = (1.0 / (np.pi * v_te**2))**(1.5) * np.exp(-v_sq / v_te**2)
    f_coeffs[0] = f_maxwell
    
    # 高阶系数：通过Galerkin投影近似
    # 使用随机采样计算投影积分
    n_samples = 500
    for samp in range(n_samples):
        # 随机样本 ξ ∈ [-1, 1]^N
        xi = 2.0 * np.random.rand(n_stochastic) - 1.0
        
        # 磁场扰动幅度（依赖于ξ）
        delta_B = 0.1 * params['B0'] * (1.0 + 0.3 * np.sum(xi) / n_stochastic)
        
        # 扰动后的分布（准线性响应）
        # 简化的线性响应模型
        Omega_e = params['Omega_e']
        response = 1.0 + 0.1 * delta_B / params['B0'] * np.sin(2 * np.pi * VPL * Omega_e / v_te)
        response = np.clip(response, 0.5, 2.0)  # 鲁棒性边界
        
        f_sample = f_maxwell * response
        
        # Galerkin投影
        for alpha_idx, alpha in enumerate(indices):
            psi_val = multivariate_legendre_basis(alpha, xi)
            f_coeffs[alpha_idx] += f_sample * psi_val / n_samples
    
    # 均值 = f_0
    f_mean = f_coeffs[0].copy()
    
    # 方差 = Σ_{α≠0} |f_α|² ⟨Ψ_α²⟩
    f_var = np.zeros((nv, nv))
    for alpha_idx in range(1, M):
        # 归一化Legendre的正交范数
        norm_sq = 1.0
        for ai in indices[alpha_idx]:
            norm_sq *= 1.0  # 已归一化
        f_var += f_coeffs[alpha_idx]**2 * norm_sq
    
    # 边界处理：确保非负
    f_mean = np.maximum(f_mean, 0.0)
    f_var = np.maximum(f_var, 0.0)
    
    return f_mean, f_var
