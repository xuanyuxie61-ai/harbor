#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
等离子体分布函数模型
================================================================================

基于 082_beta_nc (非中心Beta分布) 和 780_mortality (概率密度模型) 的思想，
构建空间等离子体中非热粒子（超热电子）的速度分布函数。

核心物理模型：

1. Kappa分布（用于描述非热等离子体）：
       f_κ(v) = n_0 / [ (π κ v_{th}²)^{3/2} ] × Γ(κ+1) / [Γ(κ-1/2) κ^{3/2}]
                × [1 + v² / (κ v_{th}²)]^{-(κ+1)}

   其中 Γ(z) 为Gamma函数，κ 为谱指数（κ → ∞ 退化为Maxwellian）。
   
   二阶矩（温度）：
       T_κ = T_M × κ / (κ - 3/2)    (κ > 3/2)

2. 非中心Beta分布尾巴（用于描述共振加速后的非热尾巴）：
       f_{tail}(v_∥) = f_0 × I_{v_∥/v_{max}}(a, b, λ)
   
   其中 I_x(a, b, λ) 为非中心不完全Beta函数，用于描述
   具有非零偏心率的粒子分布。

3. 复合分布模型：
       f_{total} = w_κ f_κ + w_β f_{tail}

   权重满足 w_κ + w_β = 1。

4. 逃逸概率模型（基于 mortality PDF 思想）：
   粒子从共振区逃逸的概率：
       P_{esc}(t) = exp(-t / τ_{esc})
   
   逃逸时间尺度：
       τ_{esc} ~ L_{res} / v_{th} × (δB / B_0)^{-2}
================================================================================
"""

import numpy as np
from scipy.special import gamma, gammainc


def kappa_distribution_3d(vx, vy, vz, n0, v_th, kappa):
    """
    3D Kappa分布函数。
    
    参数
    ----
    vx, vy, vz : ndarray
        速度分量 [m/s]。
    n0 : float
        数密度 [m^{-3}]。
    v_th : float
        热速度 [m/s]。
    kappa : float
        Kappa指数（κ > 3/2）。
        
    返回
    ----
    f : ndarray
        分布函数值 [s³/m⁶]。
    """
    # 边界检查
    kappa = max(kappa, 1.6)  # 确保 κ > 3/2
    
    v_sq = vx**2 + vy**2 + vz**2
    
    # 归一化常数
    norm = n0 / ((np.pi * kappa * v_th**2)**(1.5))
    norm *= gamma(kappa + 1.0) / (gamma(kappa - 0.5) * kappa**1.5)
    
    # 分布函数
    f = norm * (1.0 + v_sq / (kappa * v_th**2))**(-(kappa + 1.0))
    
    return f


def incomplete_beta_noncentral(x, a, b, lam, error_max=1e-10):
    """
    非中心不完全Beta函数 I_x(a, b, λ)。
    
    基于 082_beta_nc 的级数展开思想：
        I_x(a, b, λ) = Σ_{i=0}^∞ p_i(λ) I_x(a+i, b, 0)
    
    其中 p_i(λ) = e^{-λ/2} (λ/2)^i / i!
    
    参数
    ----
    x : float
        自变量，[0, 1]。
    a, b : float
        Beta参数。
    lam : float
        非中心参数 λ。
    error_max : float
        截断误差容限。
        
    返回
    ----
    value : float
        I_x(a, b, λ)。
    """
    x = np.clip(x, 0.0, 1.0)
    a = max(a, 1e-10)
    b = max(b, 1e-10)
    
    # 初始项
    pi_val = np.exp(-lam / 2.0)
    
    # 正则不完全Beta函数
    beta_log = np.log(gamma(a)) + np.log(gamma(b)) - np.log(gamma(a + b))
    
    # 简化为标准不完全Beta（当 λ 较小时）
    if lam < 1e-6:
        from scipy.special import betainc
        return betainc(a, b, x)
    
    # 级数展开
    p_sum = pi_val
    pb_sum = pi_val * gammainc(a, a, x)  # 简化的不完全Beta近似
    
    i = 0
    bi = gammainc(a, a, x)
    si = np.exp(a * np.log(x) + b * np.log(1.0 - x) - beta_log - np.log(a))
    
    while p_sum < 1.0 - error_max and i < 1000:
        i += 1
        pi_val = 0.5 * lam * pi_val / i
        bi = bi - si
        si = x * (a + b + i - 1.0) * si / (a + i)
        
        p_sum += pi_val
        pb_sum += pi_val * bi
    
    return pb_sum


def noncentral_beta_tail(v, v_max, a=2.0, b=5.0, lam=1.5):
    """
    基于非中心Beta分布的速度尾巴。
    
    参数
    ----
    v : ndarray
        速度 [m/s]。
    v_max : float
        最大速度。
    a, b, lam : float
        Beta参数和非中心参数。
        
    返回
    ----
    f_tail : ndarray
        尾巴分布。
    """
    # 映射到 [0, 1]
    x = np.clip(v / v_max, 0.0, 1.0)
    
    # 使用scipy的betainc（标准不完全Beta）作为基础
    from scipy.special import betainc
    
    # 非中心修正（简化模型）
    f_tail = betainc(a, b, x)
    
    # 乘以非中心因子
    f_tail *= (1.0 + 0.1 * lam * x)
    
    return f_tail


def kappa_nonthermal_distribution(n_particles, v_max, v_te, kappa=4.0, params=None):
    """
    生成非热粒子分布：Kappa分布 + 非中心Beta尾巴。
    
    参数
    ----
    n_particles : int
        粒子数。
    v_max : float
        最大速度 [m/s]。
    v_te : float
        热速度 [m/s]。
    kappa : float
        Kappa指数。
    params : dict, optional
        物理参数。
        
    返回
    ----
    f_dist : ndarray, shape (n_particles,)
        各粒子的分布函数值。
    v_grid : ndarray, shape (n_particles, 3)
        各粒子的速度向量。
    """
    # 随机采样速度（接受-拒绝法）
    v_grid = np.zeros((n_particles, 3))
    
    # 使用球坐标采样
    for i in range(n_particles):
        # 从Kappa分布采样速度大小
        u = np.random.rand()
        # 逆CDF近似
        v_mag = v_te * np.sqrt(kappa) * np.sqrt(u / (1.0 - u + 1e-10))
        v_mag = min(v_mag, v_max)
        
        # 随机方向
        cos_theta = 2.0 * np.random.rand() - 1.0
        sin_theta = np.sqrt(1.0 - cos_theta**2)
        phi = 2.0 * np.pi * np.random.rand()
        
        v_grid[i, 0] = v_mag * sin_theta * np.cos(phi)
        v_grid[i, 1] = v_mag * sin_theta * np.sin(phi)
        v_grid[i, 2] = v_mag * cos_theta
    
    # 计算分布函数值
    f_kappa = kappa_distribution_3d(
        v_grid[:, 0], v_grid[:, 1], v_grid[:, 2],
        n0=1.0, v_th=v_te, kappa=kappa
    )
    
    # 添加尾巴贡献
    v_mag = np.linalg.norm(v_grid, axis=1)
    f_tail = noncentral_beta_tail(v_mag, v_max)
    f_tail = f_tail / (np.max(f_tail) + 1e-30) * np.max(f_kappa) * 0.1
    
    # 复合分布
    w_kappa = 0.9
    w_tail = 0.1
    f_dist = w_kappa * f_kappa + w_tail * f_tail
    
    # 边界处理
    f_dist = np.maximum(f_dist, 1e-30)
    
    return f_dist, v_grid


def escape_probability(t, tau_esc):
    """
    粒子从共振区逃逸的概率。
    
    基于 780_mortality 的生存分析思想：
        P_{esc}(t) = exp(-t / τ_esc)
    
    参数
    ----
    t : float 或 ndarray
        时间 [s]。
    tau_esc : float
        逃逸时间尺度 [s]。
        
    返回
    ----
    P : float 或 ndarray
        逃逸概率。
    """
    t = np.maximum(t, 0.0)
    tau_esc = max(tau_esc, 1e-30)
    return 1.0 - np.exp(-t / tau_esc)


def survival_probability(t, tau_esc):
    """
    粒子在共振区中的存活概率。
    """
    return 1.0 - escape_probability(t, tau_esc)
