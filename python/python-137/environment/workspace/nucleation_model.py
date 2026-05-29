# -*- coding: utf-8 -*-
"""
nucleation_model.py

博士级结晶成核动力学模型库

融合原项目算法：
- 631_l4lib 的伪随机布尔生成（Park-Miller LCG）
- 436_flame_exact 的 Lambert W 函数用于解析求解

科学应用场景：
结晶过程中的成核分为初级均相/非均相成核和二级成核。
1. 初级成核：基于经典成核理论 (CNT)
    B_prim = A · exp(-16πγ³v_m² / [3k_B³T³(ln S)²])
2. 二级成核：与悬浮密度和搅拌强度相关
    B_sec = k_b · σ^b · M_T^j
3. 随机成核事件模拟：利用 LCG 生成器模拟离散成核事件

关键公式：
- 过饱和比 S = c / c_sat = 1 + σ
- 表面能 γ 与温度的关系：γ = γ_0 [1 - k_γ (T - T_ref)]
- 临界核半径：r* = 2γv_m / (k_B T ln S)
"""

import numpy as np
from special_functions import lambert_w


def lcg_park_miller(seed):
    """
    Park-Miller 最小标准线性同余生成器。

    算法：
        seed_{new} = 16807 · seed mod (2^31 - 1)

    参数：
        seed : int

    返回：
        new_seed : int
        uniform_val : float in (0, 1)
    """
    modulus = 2147483647  # 2^31 - 1
    multiplier = 16807
    seed = int(seed) % modulus
    if seed == 0:
        seed = 1
    k = seed // 127773
    seed = multiplier * (seed - k * 127773) - k * 2836
    if seed < 0:
        seed += modulus
    uniform_val = seed / modulus
    return seed, uniform_val


def classical_nucleation_rate(supersaturation, temperature,
                               gamma0=0.025, vm=1e-28, kb=1.380649e-23,
                               A_prefactor=1e20, k_gamma=0.001, T_ref=298.15):
    """
    经典成核理论 (CNT) 的成核率计算。

    数学公式：
        B = A · exp(-ΔG* / (k_B T))

        其中临界成核能：
        ΔG* = 16π γ³ v_m² / [3 (k_B T ln S)²]

        表面能温度依赖：
        γ(T) = γ_0 [1 - k_γ (T - T_ref)]

        过饱和比：
        S = 1 + σ

    参数：
        supersaturation : float 或 ndarray
            过饱和度 σ
        temperature : float 或 ndarray
            温度 T (K)
        gamma0 : float
            参考表面能 (J/m²)
        vm : float
            分子体积 (m³)
        kb : float
            玻尔兹曼常数 (J/K)
        A_prefactor : float
            前置因子 (#/(m³·s))
        k_gamma : float
            表面能温度系数 (1/K)
        T_ref : float
            参考温度 (K)

    返回：
        B : ndarray
            成核率 (#/(m³·s))
    """
    # TODO: Implement classical nucleation theory (CNT)
    # Compute the critical Gibbs free energy ΔG* and nucleation rate B.
    # B = A * exp(-ΔG* / (k_B * T))
    # ΔG* = 16π γ³ v_m² / [3 (k_B T ln S)²]
    # S = 1 + σ, γ(T) = γ_0 [1 - k_γ (T - T_ref)]
    # Remember boundary handling for T, sigma, S, lnS, and gamma.
    # When σ is very small, B should approach zero.
    raise NotImplementedError("Hole 1: classical_nucleation_rate is not implemented.")


def secondary_nucleation_rate(supersaturation, magma_density,
                               kb_sec=1e8, b_exp=2.0, j_exp=1.0):
    """
    二级成核率（接触成核/剪切成核）。

    数学公式：
        B_sec = k_b · σ^b · M_T^j

    其中 M_T 为悬浮密度 (kg 晶体 / m³ 浆料)。

    参数：
        supersaturation : float 或 ndarray
        magma_density : float 或 ndarray
            悬浮密度 M_T
        kb_sec : float
        b_exp : float
            过饱和度指数
        j_exp : float
            悬浮密度指数

    返回：
        B_sec : ndarray
    """
    sigma = np.asarray(supersaturation, dtype=float)
    MT = np.asarray(magma_density, dtype=float)

    sigma = np.where(sigma < 0, 0.0, sigma)
    MT = np.where(MT < 0, 0.0, MT)

    B = kb_sec * (sigma ** b_exp) * (MT ** j_exp)
    return B


def total_nucleation_rate(sigma, T, MT, A_prefactor=1e20, kb_sec=1e8,
                          b_exp=2.0, j_exp=1.0, gamma0=0.025):
    """
    总成核率 = 初级成核 + 二级成核。
    """
    B_prim = classical_nucleation_rate(sigma, T, gamma0=gamma0,
                                        A_prefactor=A_prefactor)
    B_sec = secondary_nucleation_rate(sigma, MT, kb_sec, b_exp, j_exp)
    return B_prim + B_sec


def critical_nucleus_radius(sigma, T, gamma0=0.025, vm=1e-28,
                            kb=1.380649e-23, k_gamma=0.001, T_ref=298.15):
    """
    计算临界核半径 r*。

    公式：
        r* = 2γv_m / (k_B T ln S)

    参数：
        sigma : float 或 ndarray
        T : float 或 ndarray

    返回：
        r_star : ndarray (m)
    """
    sigma = np.asarray(sigma, dtype=float)
    T = np.asarray(T, dtype=float)
    T = np.where(T <= 0, 1e-6, T)
    S = 1.0 + sigma
    S = np.where(S <= 1.0, 1.0 + 1e-10, S)
    lnS = np.log(S)
    lnS = np.where(np.abs(lnS) < 1e-10, 1e-10, lnS)

    gamma = gamma0 * (1.0 - k_gamma * (T - T_ref))
    gamma = np.where(gamma <= 0, 1e-6, gamma)

    r_star = 2.0 * gamma * vm / (kb * T * lnS)
    return r_star


def stochastic_nucleation_events(sigma, T, dt, volume, seed=12345,
                                  A_prefactor=1e20, gamma0=0.025):
    """
    基于 LCG 的随机成核事件模拟。

    算法：
        1. 计算期望成核事件数：N_expected = B(σ,T) · V · dt
        2. 当 N_expected < 10 时，使用泊松过程模拟离散事件：
           生成 U ~ Uniform(0,1)，若 U < 1 - exp(-N_expected) 则发生成核
        3. 当 N_expected >= 10 时，使用高斯近似：
           N_events ~ round(N(μ=N_expected, σ²=N_expected))

    参数：
        sigma : float
        T : float
        dt : float
            时间步长 (s)
        volume : float
            结晶器体积 (m³)
        seed : int
            LCG 种子
        A_prefactor, gamma0 : float

    返回：
        n_events : int
            成核事件数
        new_seed : int
    """
    B = classical_nucleation_rate(sigma, T, gamma0=gamma0,
                                   A_prefactor=A_prefactor)
    N_expected = float(B) * volume * dt

    if N_expected < 1e-12:
        return 0, seed

    new_seed = seed
    if N_expected < 10.0:
        # 泊松过程：生成一个均匀随机数判断是否发生
        new_seed, u = lcg_park_miller(new_seed)
        prob = 1.0 - np.exp(-N_expected)
        n_events = 1 if u < prob else 0
    else:
        # 高斯近似
        new_seed, u1 = lcg_park_miller(new_seed)
        new_seed, u2 = lcg_park_miller(new_seed)
        # Box-Muller 变换
        z = np.sqrt(-2.0 * np.log(max(u1, 1e-300))) * np.cos(2.0 * np.pi * u2)
        n_events = int(round(N_expected + np.sqrt(N_expected) * z))
        n_events = max(0, n_events)

    return n_events, new_seed


def analytical_size_dependent_growth_law(t, alpha, beta, k_g, sigma):
    """
    使用 Lambert W 函数求解尺寸依赖生长律的解析解。

    问题描述：
        生长速率 G(L) = k_g · σ^g · (1 + αL)^β
        特征线方程：dL/dt = G(L)
        分离变量并积分：
            ∫ dL / (1 + αL)^β = k_g·σ^g · t

    当 β = 1 时：
        ln(1 + αL) / α = k_g·σ^g·t
        ⇒ L(t) = [exp(α·k_g·σ^g·t) - 1] / α

    当 β ≠ 1 时，一般无闭式解。但对于 β = -1 的特殊情况：
        G(L) = k_g·σ^g / (1 + αL)
        dL/dt = k_g·σ^g / (1 + αL)
        (1 + αL) dL = k_g·σ^g dt
        L + αL²/2 = k_g·σ^g·t
        ⇒ αL² + 2L - 2k_g·σ^g·t = 0
        ⇒ L = [-1 + √(1 + 2α·k_g·σ^g·t)] / α

    更一般地，对于 G(L) = G_0·L/(1 + αL) 的形式，可用 Lambert W：
        dL/dt = G_0·L / (1 + αL)
        (1 + αL)/L dL = G_0 dt
        (1/L + α) dL = G_0 dt
        ln L + αL = G_0·t + C
        设 L(0) = L_0，则 C = ln L_0 + αL_0
        方程：ln(L/L_0) + α(L - L_0) = G_0·t
        令 u = αL，则：ln(u/(αL_0)) + u - αL_0 = G_0·t
        u·exp(u) = αL_0·exp(αL_0 + G_0·t)
        u = W(αL_0·exp(αL_0 + G_0·t))
        L = (1/α)·W(αL_0·exp(αL_0 + G_0·t))

    参数：
        t : float 或 ndarray
        alpha : float
        beta : float
        k_g : float
        sigma : float

    返回：
        L : ndarray
    """
    t = np.asarray(t, dtype=float)
    G0 = k_g * (sigma ** 2)  # 假设 g = 2

    if np.abs(beta + 1.0) < 1e-10:
        # G(L) = G0 / (1 + αL)，beta = -1
        # L = [-1 + sqrt(1 + 2α·G0·t)] / α
        val = 1.0 + 2.0 * alpha * G0 * t
        val = np.where(val < 0, 0.0, val)
        L = (-1.0 + np.sqrt(val)) / alpha
        L = np.where(alpha == 0, G0 * t, L)
        return L
    elif np.abs(beta - 1.0) < 1e-10:
        # G(L) = G0·(1 + αL)，beta = 1
        # L = [exp(α·G0·t) - 1] / α
        if np.abs(alpha) < 1e-12:
            return G0 * t
        L = (np.exp(alpha * G0 * t) - 1.0) / alpha
        return L
    else:
        # 一般情况：使用 Lambert W 求解 G(L) = G0·L/(1+αL) 的近似
        # 假设 L0 = 1e-9 (临界核尺寸)
        L0 = 1e-9
        arg = alpha * L0 * np.exp(alpha * L0 + G0 * t)
        arg = np.where(arg < -1.0 / np.e, -1.0 / np.e, arg)
        w = lambert_w(arg, branch=0)
        L = w / alpha
        L = np.where(alpha == 0, L0 * np.exp(G0 * t), L)
        return L
