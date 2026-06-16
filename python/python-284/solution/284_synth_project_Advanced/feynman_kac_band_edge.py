# -*- coding: utf-8 -*-
"""
feynman_kac_band_edge.py — Feynman-Kac 路径积分计算隧穿概率
=============================================================
核心科学问题: 使用 Feynman-Kac 公式通过布朗运动模拟
计算异质结势垒的量子隧穿概率.

融合种子项目:
  - 424_feynman_kac_3d: 3D Feynman-Kac PDE 求解器

数学物理基础:
  Feynman-Kac 公式将椭圆 PDE 的解表示为随机过程的期望:
    u(x) = E_x[exp(-∫_0^τ V(B_s)/ℏ ds) · g(B_τ)]

  其中 B_s 是布朗运动, τ 是首次出口时间, g 是边界条件.

  对于隧穿问题:
    P_tunnel = |ψ_transmitted|²/|ψ_incident|²
    等价于 Feynman-Kac 泛函的指数衰减.
"""

import numpy as np
from high_order_fd import HBAR, M0, E0


def feynman_kac_tunneling(V_func, m_star, E_particle, z_start,
                          z_left, z_right, n_paths=1000,
                          dt=1e-16, max_steps=5000, seed=42):
    """
    Feynman-Kac 路径积分计算隧穿概率.

    物理模型:
      粒子能量 E 入射到势垒 V(z), 隧穿概率为:
        T(E) = E[exp(-2/ℏ · ∫_0^τ sqrt(2m*(V-E)) ds)]

    简化为 Feynman-Kac 期望:
        u(z_0) = E_{z_0}[exp(-∫_0^τ W(B_s) ds)]
      其中 W(z) = 2m*(V(z)-E)/ℏ² (当 V>E 时)

    参数
    ----
    V_func : callable
        势能函数 V(z) [eV]
    m_star : float
        有效质量 [m0]
    E_particle : float
        粒子能量 [eV]
    z_start : float
        起始位置 [m]
    z_left, z_right : float
        域边界 [m]
    n_paths : int
        蒙特卡洛路径数
    dt : float
        时间步长 [s]
    max_steps : int
        每条路径最大步数
    seed : int
        随机种子

    返回
    ----
    T_avg : float
        平均隧穿概率
    T_std : float
        标准差
    """
    rng = np.random.RandomState(seed)
    m_kg = m_star * M0

    # 被积函数: W(z) = 2m*(V(z)-E)/(ℏ²) [1/m²]
    # 当 V > E 时为正 (经典禁戒区)
    def W(z):
        Vz = V_func(z) if callable(V_func) else V_func
        diff = Vz - E_particle  # [eV]
        if diff > 0:
            return 2.0 * m_kg * diff * E0 / HBAR**2
        return 0.0

    # 蒙特卡洛模拟
    T_values = np.zeros(n_paths)

    for ip in range(n_paths):
        z = z_start
        integral = 0.0

        for step in range(max_steps):
            # 布朗运动增量: dz = sqrt(2D*dt) * N(0,1)
            # 对于薛定谔方程, D = ℏ²/(2m*)
            D = HBAR**2 / (2 * m_kg)
            sigma = np.sqrt(2 * D * dt)
            dz = sigma * rng.randn()
            z += dz

            # 累加 Feynman-Kac 泛函
            integral += W(z) * dt

            # 检查边界
            if z <= z_left:
                # 到达左边界: 反射
                T_values[ip] = np.exp(-integral)
                break
            elif z >= z_right:
                # 到达右边界: 透射
                T_values[ip] = np.exp(-integral)
                break
        else:
            # 达到最大步数: 衰减贡献
            T_values[ip] = np.exp(-integral)

    T_avg = np.mean(T_values)
    T_std = np.std(T_values) / np.sqrt(n_paths)

    return T_avg, T_std


def wkb_tunneling_estimate(V_func, m_star, E_particle, z_turn_left,
                           z_turn_right, n_points=200):
    """
    WKB 近似隧穿概率 (解析参考):

    T_WKB = exp(-2 * integral_{z1}^{z2} kappa(z) dz)

    其中:
      kappa(z) = sqrt(2m*(V(z)-E))/ℏ  [1/m]
      z1, z2 是经典回转点 (V(z)=E)

    参数
    ----
    V_func : callable
        势能函数 V(z) [eV]
    m_star : float
        有效质量 [m0]
    E_particle : float
        粒子能量 [eV]
    z_turn_left, z_turn_right : float
        经典回转点 [m]
    n_points : int
        积分点数

    返回
    ----
    T_wkb : float
        WKB 隧穿概率
    """
    m_kg = m_star * M0
    z_arr = np.linspace(z_turn_left, z_turn_right, n_points)

    kappa = np.zeros(n_points)
    for i, z in enumerate(z_arr):
        Vz = V_func(z) if callable(V_func) else V_func
        diff = max(Vz - E_particle, 0.0)  # [eV]
        kappa[i] = np.sqrt(2 * m_kg * diff * E0) / HBAR

    # Simpson 积分
    dz = z_arr[1] - z_arr[0]
    if n_points >= 3:
        integral = _simpson_integrate(kappa, dz)
    else:
        integral = np.trapz(kappa, z_arr)

    T_wkb = np.exp(-2.0 * integral)
    return T_wkb


def _simpson_integrate(f, dx):
    """Simpson 1/3 法则积分."""
    n = len(f)
    if n < 3:
        return np.trapz(f, dx=dx)
    if n % 2 == 0:
        # 偶数个点: 最后一段用梯形
        s = sum(f[i] + 4*f[i+1] + f[i+2]
                for i in range(0, n - 3, 2))
        s += f[-2] + f[-1]  # 梯形修正
        return dx / 3.0 * s + dx * 0.5 * (f[-2] + f[-1])
    else:
        s = sum(f[i] + 4*f[i+1] + f[i+2]
                for i in range(0, n - 2, 2))
        return dx / 3.0 * s


def transmission_coefficient_scan(V_func, m_star, E_range,
                                   z_start, z_left, z_right,
                                   n_paths=500, seed=42):
    """
    扫描能量范围的透射系数.

    参数
    ----
    E_range : ndarray
        能量范围 [eV]

    返回
    ----
    T_array : ndarray
        透射系数 T(E)
    """
    T_array = np.zeros(len(E_range))
    for i, E in enumerate(E_range):
        T_avg, _ = feynman_kac_tunneling(
            V_func, m_star, E, z_start, z_left, z_right,
            n_paths=n_paths, seed=seed + i)
        T_array[i] = T_avg
    return T_array


def resonant_tunneling_condition(V_max, V_min, barrier_width, m_star):
    """
    共振隧穿条件:
      E_n = V_min + n²π²ℏ²/(2m*L²)

    参数
    ----
    V_max : float
        势垒高度 [eV]
    V_min : float
        势阱底部 [eV]
    barrier_width : float
        势阱宽度 [m]
    m_star : float
        有效质量 [m0]

    返回
    ----
    E_resonances : ndarray
        共振能量 [eV]
    """
    m_kg = m_star * M0
    L = barrier_width

    # 量子化能级
    n_max = 10
    E_levels = np.zeros(n_max)
    for n in range(1, n_max + 1):
        E_levels[n - 1] = V_min + n**2 * np.pi**2 * HBAR**2 / (2 * m_kg * L**2 * E0)

    # 只保留在势垒内的能级
    E_resonances = E_levels[E_levels < V_max]
    return E_resonances
