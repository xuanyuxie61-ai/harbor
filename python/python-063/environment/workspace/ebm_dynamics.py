"""
ebm_dynamics.py
能量平衡模型（Energy Balance Model, EBM）动力学模块

基于 Budyko-Sellers 一维能量平衡方程:
    C(x) * dT/dt = Q * S(x,t) * (1 - alpha(T)) - epsilon * sigma * T^4
                   + D * Laplacian(T) + F_forcing

使用隐式梯形法（Trapezoidal Rule）进行时间积分，
融合种子项目 833_ode_trapezoidal 的隐式迭代框架。
"""

import numpy as np

# 物理常数
SIGMA = 5.670374419e-8          # Stefan-Boltzmann 常数 [W/(m^2 K^4)]
Q_SOLAR = 1361.0 / 4.0          # 球面平均太阳常数 [W/m^2]
C_OCEAN = 2.5e8                 # 海洋热容 [J/(m^2 K)]
C_LAND = 5.0e6                  # 陆地热容 [J/(m^2 K)]


def ice_albedo_feedback(T, T_ice=263.15, alpha_ocean=0.08, alpha_ice=0.65, k=0.5):
    """
    冰反照率反馈的平滑模型:
        alpha(T) = alpha_ocean + (alpha_ice - alpha_ocean) / (1 + exp(k*(T - T_ice)))
    避免 Heaviside 函数的数值不连续。
    """
    T = np.asarray(T, dtype=np.float64)
    return alpha_ocean + (alpha_ice - alpha_ocean) / (1.0 + np.exp(k * (T - T_ice)))


def outgoing_longwave_radiation(T, epsilon=0.62):
    """
    向外长波辐射: OLR = epsilon * sigma * T^4
    epsilon = 0.62 为有效发射率，使 T = 288 K 时 OLR ≈ 240 W/m^2。
    """
    T_safe = np.maximum(np.asarray(T, dtype=np.float64), 100.0)
    return epsilon * SIGMA * T_safe**4


def solar_insolation(lat, t=None, orbital_variation=False):
    """
    归一化日射分布 S(lat):
        S(lat) = 1 - 0.482 * P2(sin(lat))
    其中 P2(x) = (3x^2 - 1)/2 为二阶 Legendre 多项式。
    可选 Milanovitch 轨道调制（偏心率 100 kyr、岁差 23 kyr、倾角 41 kyr）。
    """
    sin_lat = np.sin(np.asarray(lat, dtype=np.float64))
    P2 = 0.5 * (3.0 * sin_lat**2 - 1.0)
    S = 1.0 - 0.482 * P2

    if orbital_variation and t is not None:
        ecc = 0.0167 + 0.011 * np.cos(2.0 * np.pi * t / 100000.0)
        precession = np.cos(2.0 * np.pi * t / 23000.0 + np.pi / 6.0)
        obliquity = 0.4091 + 0.025 * np.cos(2.0 * np.pi * t / 41000.0)
        modulation = 1.0 + ecc * precession * sin_lat * np.cos(obliquity)
        S = S * modulation

    return np.maximum(S, 0.1)


def spherical_laplacian(T, vertices, faces, dual_areas):
    """
    球面拉普拉斯算子的有限体积离散化。
    对每个节点 i，取相邻面顶点的加权平均差分近似:
        Lap(T)_i ≈ (neighbor_mean(T) - T_i) / dual_area_i
    在球面网格上此近似具有一阶一致性。
    """
    n_nodes = len(vertices)
    T = np.asarray(T, dtype=np.float64)
    neighbor_sum = np.zeros(n_nodes, dtype=np.float64)
    neighbor_cnt = np.zeros(n_nodes, dtype=int)

    for tri in faces:
        i, j, k = tri
        neighbor_sum[i] += T[j] + T[k]
        neighbor_sum[j] += T[i] + T[k]
        neighbor_sum[k] += T[i] + T[j]
        neighbor_cnt[i] += 2
        neighbor_cnt[j] += 2
        neighbor_cnt[k] += 2

    neighbor_cnt = np.maximum(neighbor_cnt, 1)
    neighbor_avg = neighbor_sum / neighbor_cnt
    lap = (neighbor_avg - T) / (0.1 + dual_areas)
    return lap


def compute_heat_capacity(lat):
    """
    空间非均匀热容：赤道附近海洋面积占比大 -> 热容大；
    极地陆地/冰面积占比大 -> 热容小。
    """
    return C_OCEAN * 0.5 + (C_OCEAN - C_LAND) * 0.5 * np.cos(np.asarray(lat))**2


def ebm_rhs(T, vertices, faces, areas, dual_areas, t,
            D_diff=0.55, epsilon=0.6, volcanic_forcing=0.0, solar_forcing=0.0):
    """
    能量平衡模型右端项 dT/dt = RHS(T, t)。

    方程:
        C * dT/dt = Q*S*(1 - alpha) - OLR + D*Lap(T) + F_forcing

    参数:
        T: 温度场 [K]
        vertices, faces, areas, dual_areas: 网格数据
        t: 时间 [年]
        D_diff: 热扩散系数 [W/(m^2 K)]
        epsilon: 长波辐射发射率
        volcanic_forcing: 火山气溶胶强迫 [W/m^2]
        solar_forcing: 太阳辐照度调制 [W/m^2]
    """
    # TODO: Implement the EBM right-hand side.
    # The energy balance equation is:
    #   C * dT/dt = Q*S*(1 - alpha) - OLR + D*Lap(T) + F_forcing
    # where:
    #   S      = solar_insolation(lat, t)
    #   alpha  = ice_albedo_feedback(T)
    #   OLR    = outgoing_longwave_radiation(T, epsilon)
    #   Lap(T) = spherical_laplacian(T, vertices, faces, dual_areas)
    #   C      = compute_heat_capacity(lat)
    #   F      = volcanic_forcing + solar_forcing
    # Apply physical bounds to temperature and rhs.
    # HINT: The function signature and docstring above must NOT be changed.
    pass


def implicit_trapezoidal_step(T_n, dt, rhs_func, max_iter=15, tol=1e-8):
    """
    隐式梯形法单步积分:
        T_{n+1} = T_n + dt/2 * [f(T_n) + f(T_{n+1})]

    使用 Picard 定点迭代求解非线性隐式方程:
        z^{(j+1)} = T_n + dt/2 * [f(T_n) + f(z^{(j)})]

    融合种子项目 833_ode_trapezoidal 的核心迭代框架。
    """
    T_n = np.asarray(T_n, dtype=np.float64)
    f_n = rhs_func(T_n)
    z = T_n + dt * f_n  # 显式 Euler 作为初值

    for _ in range(max_iter):
        f_z = rhs_func(z)
        z_new = T_n + 0.5 * dt * (f_n + f_z)
        diff = np.max(np.abs(z_new - z))
        z = z_new
        if diff < tol:
            break
    return z
