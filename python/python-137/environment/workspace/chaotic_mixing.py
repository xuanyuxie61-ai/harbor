# -*- coding: utf-8 -*-
"""
chaotic_mixing.py

博士级结晶器混沌混合模型

融合原项目算法：
- 168_chen_ode 的 Chen 混沌吸引子

科学应用场景：
工业结晶器中搅拌产生的流场并非完全均匀，而是存在混沌混合区域。
Chen 吸引子（Guanrong Chen & Tetsushi Ueta, 1999）是一个 Lorenz 型
三维混沌系统，其方程为：

    dx/dt = a·(y - x)
    dy/dt = (c - a)·x - x·z + c·y
    dz/dt = x·y - b·z

在结晶过程中，我们将其映射为局部过饱和度波动的动力学模型：
- x: 局部温度偏差 ΔT
- y: 局部浓度偏差 Δc
- z: 局部能量耗散率 ε

混沌混合导致局部过饱和度呈现非周期涨落，显著影响成核率（成核
对过饱和度高度敏感，指数依赖）。
"""

import numpy as np
from scipy.integrate import solve_ivp


def chen_attractor_rhs(t, state, a=40.0, b=3.0, c=28.0):
    """
    Chen 混沌吸引子的右端项。

    方程组：
        dx/dt = a·(y - x)
        dy/dt = (c - a)·x - x·z + c·y
        dz/dt = x·y - b·z

    参数：
        t : float
        state : array-like, shape (3,)
            [x, y, z]
        a, b, c : float
            Chen 系统参数

    返回：
        dstate : ndarray, shape (3,)
    """
    x, y, z = state
    dxdt = a * (y - x)
    dydt = (c - a) * x - x * z + c * y
    dzdt = x * y - b * z
    return np.array([dxdt, dydt, dzdt], dtype=float)


def generate_chaotic_mixing_trajectory(t_span, y0=None, params=None,
                                        method='RK45', rtol=1e-8, atol=1e-10):
    """
    生成混沌混合轨迹。

    参数：
        t_span : tuple (t0, tf)
        y0 : array-like, shape (3,), optional
            初始条件，默认 [-0.1, 0.5, -0.6]
        params : dict, optional
            {'a': 40.0, 'b': 3.0, 'c': 28.0}
        method : str
            ODE 求解方法
        rtol, atol : float
            相对/绝对容差

    返回：
        sol : OdeSolution
            scipy 的解对象
    """
    if y0 is None:
        y0 = np.array([-0.1, 0.5, -0.6], dtype=float)
    else:
        y0 = np.asarray(y0, dtype=float)

    if params is None:
        params = {'a': 40.0, 'b': 3.0, 'c': 28.0}

    def rhs(t, y):
        return chen_attractor_rhs(t, y, **params)

    sol = solve_ivp(rhs, t_span, y0, method=method,
                    dense_output=True, rtol=rtol, atol=atol,
                    max_step=(t_span[1] - t_span[0]) / 1000)
    return sol


def map_chen_to_supersaturation_fluctuation(t, sol, sigma_base,
                                             scale_T=0.5, scale_c=0.3):
    """
    将 Chen 吸引子状态映射为过饱和度波动。

    映射关系：
        ΔT(t) = scale_T · x(t)
        Δc(t) = scale_c · y(t)
        σ_fluct(t) = [(c_base + Δc) - c_sat(T_base + ΔT)] / c_sat(T_base + ΔT)

    由于 c_sat 随温度变化（van't Hoff），温度波动和浓度波动的
    耦合产生复杂的过饱和度动力学。

    参数：
        t : float 或 ndarray
        sol : OdeSolution
            Chen 系统的解
        sigma_base : float
            基础过饱和度
        scale_T, scale_c : float
            温度和浓度的缩放因子

    返回：
        sigma_local : ndarray
            局部过饱和度
    """
    t = np.asarray(t, dtype=float)
    # 确保 t 在求解区间内
    t0, tf = sol.t[0], sol.t[-1]
    t = np.clip(t, t0, tf)

    states = sol.sol(t)
    x = states[0, :]
    y = states[1, :]

    # 简化的线性映射：过饱和度波动与 x 和 y 的组合相关
    # 物理上：Δσ ≈ (∂σ/∂c)·Δc + (∂σ/∂T)·ΔT
    #        ∂σ/∂c = 1/c_sat,  ∂σ/∂T = -(c/c_sat)·(d ln c_sat / dT)
    # 这里使用简化模型
    d_sigma = scale_c * y - scale_T * x
    sigma_local = sigma_base + d_sigma
    # 边界处理：过饱和度不能为负（未饱和）且通常不超过某个上限
    sigma_local = np.clip(sigma_local, 0.0, 5.0)
    return sigma_local


def mixing_enhanced_nucleation_rate(sigma_base, t, sol, B0, scale_T=0.5, scale_c=0.3):
    """
    计算混沌混合增强的有效成核率。

    理论：局部过饱和度的瞬时值 σ_local(t) 远高于平均值 σ_base 时，
    会在局部区域触发爆发性成核。由于成核率对过饱和度的指数依赖，
    时间平均的成核率 <B> 显著高于 B(σ_base)。

    <B> = (1/T) ∫_0^T B(σ_local(t)) dt
        = (1/T) ∫_0^T B0 · exp(-A / [ln(1+σ_local(t))]^2) dt

    参数：
        sigma_base : float
        t : ndarray
            时间点
        sol : OdeSolution
        B0 : float
            前置因子
        scale_T, scale_c : float

    返回：
        B_avg : float
            时间平均成核率
        B_instant : ndarray
            瞬时成核率
    """
    sigma_local = map_chen_to_supersaturation_fluctuation(t, sol, sigma_base,
                                                           scale_T, scale_c)
    # 经典成核理论 (CNT)：B = B0 · exp(-A / [ln S]^2)
    # 其中 S = 1 + σ
    S = 1.0 + sigma_local
    S = np.where(S <= 1.0, 1.0 + 1e-6, S)
    lnS = np.log(S)
    # 避免 lnS 过小导致数值爆炸
    lnS = np.where(np.abs(lnS) < 1e-6, 1e-6, lnS)
    A = 16.0  # 典型值，与表面能、摩尔体积等相关
    B_instant = B0 * np.exp(-A / (lnS ** 2))
    B_avg = np.mean(B_instant)
    return B_avg, B_instant
