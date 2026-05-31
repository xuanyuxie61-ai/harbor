# -*- coding: utf-8 -*-
"""
growth_kinetics.py

博士级晶体生长动力学模型库

科学应用场景：
晶体生长速率受多种因素控制：过饱和度、温度、晶体尺寸、溶剂组成等。
本模块实现了多种生长动力学模型，用于人口平衡方程 (PBE) 的求解。

核心模型：
1. 幂律生长 (Power-law growth):
    G = k_g · σ^g
2. 尺寸依赖生长 (Size-dependent growth, ΔL-law):
    G(L) = k_g · σ^g · (1 + αL)^β
3. 温度活化生长 (Arrhenius-type):
    G = k_g0 · exp(-E_g / (R·T)) · σ^g
4. 扩散-表面集成联合控制 (Two-step model):
    1/G = 1/k_d + 1/(k_r·σ^g)
    其中 k_d 为传质系数，k_r 为表面集成系数
5. Burton-Cabrera-Frank (BCF) 螺旋位错生长:
    G = A · σ² · tanh(B/σ)

关键公式：
- 生长活化能 E_g 通常为 30~80 kJ/mol
- 扩散控制时 g ≈ 1，表面集成控制时 g ≈ 1~2
"""

import numpy as np


def power_law_growth(sigma, T, k_g0, E_g, g_exp, R=8.314):
    """
    温度活化的幂律生长模型。

    数学公式：
        G = k_g0 · exp(-E_g / (R·T)) · σ^g

    参数：
        sigma : float 或 ndarray
        T : float 或 ndarray
            温度 (K)
        k_g0 : float
            前置因子 (m/s)
        E_g : float
            生长活化能 (J/mol)
        g_exp : float
            过饱和度指数
        R : float
            气体常数

    返回：
        G : ndarray
            生长速率 (m/s)
    """
    sigma = np.asarray(sigma, dtype=float)
    T = np.asarray(T, dtype=float)

    T = np.where(T <= 0, 1e-6, T)
    sigma = np.where(sigma < 0, 0.0, sigma)

    G = k_g0 * np.exp(-E_g / (R * T)) * (sigma ** g_exp)
    return G


def size_dependent_growth(L, sigma, T, k_g0, E_g, g_exp, alpha, beta, R=8.314):
    """
    尺寸依赖生长模型 (ΔL-law 变体)。

    数学公式：
        G(L, σ, T) = k_g0 · exp(-E_g / (R·T)) · σ^g · (1 + αL)^β

    物理意义：
        - α > 0, β > 0: 大尺寸晶体生长更快（尺寸放大效应）
        - α > 0, β < 0: 大尺寸晶体生长更慢（扩散限制）
        - α = 0: 恢复幂律模型

    参数：
        L : float 或 ndarray
            晶体特征尺寸 (m)
        sigma : float 或 ndarray
        T : float 或 ndarray
        k_g0, E_g, g_exp : float
        alpha : float
            尺寸依赖系数 (1/m)
        beta : float
            尺寸依赖指数
        R : float

    返回：
        G : ndarray
    """
    L = np.asarray(L, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    T = np.asarray(T, dtype=float)

    T = np.where(T <= 0, 1e-6, T)
    sigma = np.where(sigma < 0, 0.0, sigma)
    L = np.where(L < 0, 0.0, L)

    base = k_g0 * np.exp(-E_g / (R * T)) * (sigma ** g_exp)
    size_factor = (1.0 + alpha * L) ** beta
    G = base * size_factor
    return G


def two_step_growth(sigma, T, k_d, k_r0, E_r, g_r, R=8.314):
    """
    扩散-表面集成联合控制的两步生长模型。

    数学公式：
        1/G = 1/k_d + 1/[k_r · σ^{g_r}]
        其中 k_r = k_r0 · exp(-E_r / (R·T))

    或等价地：
        G = (k_d · k_r · σ^{g_r}) / (k_d + k_r · σ^{g_r})

    极限情况：
        - 当 k_d << k_r·σ^{g_r} 时，G ≈ k_d （扩散控制）
        - 当 k_d >> k_r·σ^{g_r} 时，G ≈ k_r·σ^{g_r} （表面集成控制）

    参数：
        sigma : float 或 ndarray
        T : float 或 ndarray
        k_d : float
            传质系数 (m/s)
        k_r0 : float
            表面集成前置因子 (m/s)
        E_r : float
            表面集成活化能 (J/mol)
        g_r : float
            表面集成过饱和度指数
        R : float

    返回：
        G : ndarray
    """
    sigma = np.asarray(sigma, dtype=float)
    T = np.asarray(T, dtype=float)

    T = np.where(T <= 0, 1e-6, T)
    sigma = np.where(sigma < 0, 0.0, sigma)

    k_r = k_r0 * np.exp(-E_r / (R * T))
    term = k_r * (sigma ** g_r)
    denom = k_d + term
    denom = np.where(np.abs(denom) < 1e-300, 1e-300, denom)
    G = k_d * term / denom
    return G


def bcf_spiral_growth(sigma, T, A_bcf, B_bcf, E_act, R=8.314):
    """
    Burton-Cabrera-Frank (BCF) 螺旋位错生长模型。

    数学公式：
        G = A · exp(-E_act / (R·T)) · σ² · tanh(B / σ)

    极限行为：
        - 低过饱和度 (σ << B): tanh(B/σ) ≈ 1, G ≈ A·σ²
          （螺旋位错控制，抛物线律）
        - 高过饱和度 (σ >> B): tanh(B/σ) ≈ B/σ, G ≈ A·B·σ
          （台阶源饱和，线性律）

    参数：
        sigma : float 或 ndarray
        T : float 或 ndarray
        A_bcf : float
            BCF 前置因子 (m/s)
        B_bcf : float
            BCF 饱和参数 (无量纲)
        E_act : float
            活化能 (J/mol)
        R : float

    返回：
        G : ndarray
    """
    sigma = np.asarray(sigma, dtype=float)
    T = np.asarray(T, dtype=float)

    T = np.where(T <= 0, 1e-6, T)
    sigma = np.where(sigma < 0, 0.0, sigma)

    prefactor = A_bcf * np.exp(-E_act / (R * T))
    # 避免 σ 过小导致 tanh 参数过大
    arg = B_bcf / np.where(sigma < 1e-10, 1e-10, sigma)
    # tanh 大参数趋于 1，可直接截断
    tanh_val = np.tanh(arg)
    G = prefactor * (sigma ** 2) * tanh_val
    return G


def growth_rate_dispersion(G_mean, cv=0.1, n_samples=1000, rng=None):
    """
    生长速率分散模型 (Growth Rate Dispersion, GRD)。

    理论：即使在相同过饱和度下，不同晶体的生长速率也存在
    统计分散。通常假设服从对数正态分布：

        G_i ~ LogNormal(μ_G, σ_G²)
        其中 μ_G = ln(G_mean) - σ_G²/2,  σ_G = cv·G_mean

    参数：
        G_mean : float
            平均生长速率 (m/s)
        cv : float
            变异系数
        n_samples : int
            样本数
        rng : numpy.random.Generator

    返回：
        G_samples : ndarray
    """
    if rng is None:
        rng = np.random.default_rng()
    if G_mean <= 0:
        return np.zeros(n_samples)
    sigma_ln = np.sqrt(np.log(1.0 + cv ** 2))
    mu_ln = np.log(G_mean) - 0.5 * sigma_ln ** 2
    G_samples = rng.lognormal(mu_ln, sigma_ln, n_samples)
    return G_samples
