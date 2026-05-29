# -*- coding: utf-8 -*-
"""
cooling_profile.py

博士级结晶过程冷却曲线生成库

融合原项目算法：
- 1059_sawtooth_ode 的锯齿波生成与驱动振荡器思想

科学应用场景：
结晶过程中的冷却策略直接影响过饱和度曲线，从而决定成核与生长动力学。
1. 线性冷却：最简单，但过饱和度先低后高，导致产品粒度分布不均
2. 自然冷却：遵循指数衰减，初期过饱和度过高导致大量成核
3. 锯齿波程序冷却：周期性温度波动用于诱导可控成核事件（细种子生成）
4. 最优冷却曲线：基于矩方法推导的解析最优控制策略

关键公式：
- 过饱和度 σ = (c - c_sat(T)) / c_sat(T)
- 最优冷却：T_opt(t) 使得 σ(t) = σ*（恒定最优过饱和度）
"""

import numpy as np


def linear_cooling(t, T0, Tf, t_total):
    """
    线性冷却曲线。

    T(t) = T0 + (Tf - T0) · t / t_total

    参数：
        t : float 或 ndarray
        T0 : float
            初始温度 (K)
        Tf : float
            最终温度 (K)
        t_total : float
            总时间 (s)

    返回：
        T : ndarray
    """
    t = np.asarray(t, dtype=float)
    if t_total <= 0:
        return np.full_like(t, T0)
    ratio = np.clip(t / t_total, 0.0, 1.0)
    return T0 + (Tf - T0) * ratio


def natural_cooling(t, T0, T_env, tau):
    """
    自然冷却（牛顿冷却定律）。

    数学模型：
        dT/dt = -(T - T_env) / τ
        T(t) = T_env + (T0 - T_env) · exp(-t/τ)

    参数：
        t : float 或 ndarray
        T0 : float
            初始温度 (K)
        T_env : float
            环境温度 (K)
        tau : float
            冷却时间常数 (s)
    """
    t = np.asarray(t, dtype=float)
    if tau <= 0:
        return np.full_like(t, T0)
    return T_env + (T0 - T_env) * np.exp(-t / tau)


def sawtooth_cooling(t, T_base, delta_T, period, phase=0.0):
    """
    锯齿波程序冷却曲线。

    数学模型：
        锯齿波定义为：
        s(t) = mod(t + ω·π, 2·ω·π) / (2·ω·π) - 0.5
        这里将其映射为温度扰动叠加在基础冷却曲线上。

        T(t) = T_base(t) + ΔT · s(t)

    其中 s(t) 是归一化锯齿波，周期为 period，幅值为 1。

    科学意义：周期性温度扰动可在结晶过程中产生可控的细晶种，
    用于改善最终产品的粒度分布（细种子-粗生长策略）。

    参数：
        t : float 或 ndarray
        T_base : float 或 callable
            基础温度曲线或其函数句柄
        delta_T : float
            温度波动幅值 (K)
        period : float
            锯齿波周期 (s)
        phase : float
            相位偏移 (s)

    返回：
        T : ndarray
    """
    t = np.asarray(t, dtype=float)
    if period <= 0:
        period = 1.0

    # 锯齿波生成：mod(t + phase, period) / period - 0.5，范围 [-0.5, 0.5]
    s = np.mod(t + phase, period) / period - 0.5

    if callable(T_base):
        Tb = T_base(t)
    else:
        Tb = np.full_like(t, float(T_base), dtype=float) if np.isscalar(T_base) else np.asarray(T_base, dtype=float)

    return Tb + delta_T * s


def optimal_cooling_polynomial(t, T0, Tf, t_total, order=3):
    """
    基于多项式的近似最优冷却曲线。

    理论背景：
        对于恒定生长速率 G* 和零成核的理想结晶过程，
        最优温度曲线可通过矩方法推导。

        设第三生长矩 μ_3 的期望轨迹为：
        μ_3(t) = μ_3(0) + 3·ρ_c·k_v·G*·∫_0^t μ_2(τ)dτ

        结合质量平衡和溶解度关系，可得温度应满足：
        T_opt(t) ≈ T0 + (Tf - T0)·(t/t_total)^p

        其中指数 p 取决于动力学参数。通常 p ≈ 2~3 可给出较好的
        过饱和度控制。

    参数：
        t : float 或 ndarray
        T0, Tf : float
        t_total : float
        order : float
            多项式阶数 p (通常 2~4)

    返回：
        T : ndarray
    """
    t = np.asarray(t, dtype=float)
    if t_total <= 0:
        return np.full_like(t, T0)
    ratio = np.clip(t / t_total, 0.0, 1.0)
    return T0 + (Tf - T0) * (ratio ** order)


def solubility_vanthoff(T, H_diss, S_diss, R=8.314):
    """
    基于 van't Hoff 方程的溶解度计算。

    数学公式：
        ln(c_sat) = -ΔH_diss / (R·T) + ΔS_diss / R
        c_sat = exp(-ΔH_diss/(R·T) + ΔS_diss/R)

    参数：
        T : float 或 ndarray
            温度 (K)，必须 > 0
        H_diss : float
            溶解焓 (J/mol)
        S_diss : float
            溶解熵 (J/(mol·K))
        R : float
            气体常数

    返回：
        c_sat : ndarray
            饱和浓度 (kg 溶质 / kg 溶剂)
    """
    T = np.asarray(T, dtype=float)
    T = np.where(T <= 0, 1e-6, T)  # 边界处理
    return np.exp(-H_diss / (R * T) + S_diss / R)


def supersaturation(c, T, H_diss, S_diss):
    """
    计算过饱和度。

    定义：
        σ = (c - c_sat(T)) / c_sat(T)

    参数：
        c : float 或 ndarray
            实际浓度
        T : float 或 ndarray
            温度 (K)
        H_diss, S_diss : float
            van't Hoff 参数

    返回：
        sigma : ndarray
    """
    c_sat = solubility_vanthoff(T, H_diss, S_diss)
    # 避免除以零
    c_sat = np.where(np.abs(c_sat) < 1e-300, 1e-300, c_sat)
    return (c - c_sat) / c_sat
