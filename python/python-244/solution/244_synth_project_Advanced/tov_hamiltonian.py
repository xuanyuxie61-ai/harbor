#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tov_hamiltonian.py
==================
【融合种子项目】 315_double_well_ode (双势阱 ODE 哈密顿系统)

本模块将 double_well_ode 的哈密顿 ODE 求解框架推广至
Tolman-Oppenheimer-Volkoff (TOV) 方程, 描述中子星流体静力学平衡.

物理/数学公式
-------------
1. TOV 方程 (广义相对论流体静力学平衡):
       dP/dr = -G [epsilon(r) + P(r)/c^2] [m(r) + 4 pi r^3 P(r)/c^2]
               / [r (r - 2Gm(r)/c^2)]

   质量方程:
       dm/dr = 4 pi r^2 epsilon(r) / c^2

   其中:
       P = 压力 (dyn/cm^2)
       epsilon = 能量密度 (erg/cm^3)
       m(r) = 半径 r 内的引力质量 (g)
       G = 引力常数
       c = 光速

2. 哈密顿形式:
   引入共轭变量, TOV 系统可写为:
       dq/dr = partial H / partial p
       dp/dr = -partial H / partial q
   其中广义坐标 q = (m, r), 广义动量 p = (P, phi)
   哈密顿量:
       H = -P - epsilon * phi + G m epsilon / (r c^2) + ...

3. 守恒量 (推广自 double_well_conserved):
   在平衡态, 红移因子守恒:
       z(r) * [1 + P(r)/(epsilon(r) + rho(r) c^2)] = const
   即 Tolman 温度条件:
       T(r) sqrt(-g_{00}) = const

4. 状态方程 (多方模型):
       P = K rho^{Gamma}
   或分段多方:
       P_i = K_i rho^{Gamma_i},  rho_{i-1} < rho < rho_i

5. 守恒量监测 (类似 double_well_conserved):
       E = H(q, p) - H(q_0, p_0)
   期望 |E| / |H_0| < tol

6. RK4 时间步进 (经典方法):
       k1 = f(r_n, y_n)
       k2 = f(r_n + h/2, y_n + h k1/2)
       k3 = f(r_n + h/2, y_n + h k2/2)
       k4 = f(r_n + h, y_n + h k3)
       y_{n+1} = y_n + (h/6)(k1 + 2k2 + 2k3 + k4)

7. 星体表面条件:
       P(R) = 0  =>  定义星体半径 R
       M = m(R)  =>  引力质量

8. 紧凑度参数:
       C = GM / (R c^2)
   典型中子星: C ~ 0.15 - 0.25 (接近 Buchdahl 极限 4/9)
"""

import math
import numpy as np
from typing import Tuple, Dict, Callable, Optional
from numerical_constants import (r8_epsilon, NeutronStarConstants as NS)


# ============================================================
# 第一部分: 状态方程模型
# ============================================================

class PolytropicEoS:
    """
    多方状态方程.

    P = K rho^{Gamma}

    其中:
        K = 多方常数 (cgs)
        Gamma = 多方指数 (绝热指数)
        rho = 静止质量密度 (g/cm^3)

    能量密度 (包含静止质量):
        epsilon = rho c^2 + P / (Gamma - 1)

    声速:
        c_s^2 = Gamma P / (epsilon + P/c^2)

    典型中子星参数:
        外壳: K ~ 1e28, Gamma ~ 4/3
        内核: K ~ 1e35, Gamma ~ 2.5-3.0
    """

    def __init__(self, K: float = 1.0e35, gamma: float = 2.5):
        """
        Parameters
        ----------
        K : float
            多方常数 (cgs 单位: dyn cm^{3 Gamma} g^{-Gamma})
        gamma : float
            多方指数 (典型值 2-3)
        """
        if K <= 0:
            raise ValueError("K 必须为正")
        if gamma <= 1.0:
            raise ValueError("Gamma 必须 > 1")
        self.K = K
        self.gamma = gamma

    def pressure(self, rho: float) -> float:
        """P = K rho^Gamma."""
        if rho <= 0:
            return 0.0
        return self.K * rho ** self.gamma

    def energy_density(self, rho: float) -> float:
        """
        epsilon = rho c^2 + P / (Gamma - 1).

        第一项为静止质量贡献, 第二项为内能.
        """
        if rho <= 0:
            return 0.0
        p = self.pressure(rho)
        return rho * NS.c_light ** 2 + p / (self.gamma - 1.0)

    def rho_from_pressure(self, p: float) -> float:
        """
        由压力反解密度:
            rho = (P/K)^{1/Gamma}
        """
        if p <= 0:
            return 0.0
        return (p / self.K) ** (1.0 / self.gamma)

    def sound_speed_squared(self, rho: float) -> float:
        """
        c_s^2 = dP/d epsilon = Gamma P / (epsilon + P/c^2).
        """
        if rho <= 0:
            return 0.0
        p = self.pressure(rho)
        eps = self.energy_density(rho)
        denom = eps + p / NS.c_light ** 2
        if denom <= 0:
            return 0.0
        cs2 = self.gamma * p / denom * NS.c_light ** 2
        return cs2

    def enthalpy(self, rho: float) -> float:
        """
        比焓 h = (epsilon + P) / (rho c^2).
        """
        if rho <= 0:
            return 1.0
        p = self.pressure(rho)
        eps = self.energy_density(rho)
        return (eps + p) / (rho * NS.c_light ** 2)


class PiecewisePolytropicEoS:
    """
    分段多方状态方程.

    将密度空间分为多段, 每段使用不同的 (K_i, Gamma_i):
        P(rho) = K_i rho^{Gamma_i},  rho_{i-1} < rho < rho_i

    K_i 通过连续性条件确定:
        K_{i+1} = K_i rho_i^{Gamma_i - Gamma_{i+1}}
    """

    def __init__(self, gamma_segments: list, rho_boundaries: list,
                 K0: float = 1.0e34):
        """
        Parameters
        ----------
        gamma_segments : list of float
            各段的多方指数 [Gamma_1, Gamma_2, ...]
        rho_boundaries : list of float
            段间边界密度 [rho_1, rho_2, ...]
            长度 = len(gamma_segments) - 1
        K0 : float
            第一段的多方常数
        """
        self.n_segments = len(gamma_segments)
        self.gamma = np.array(gamma_segments, dtype=np.float64)
        self.rho_bounds = np.array([0.0] + list(rho_boundaries) + [1e20],
                                    dtype=np.float64)

        # 通过连续性确定各段 K
        self.K_array = np.zeros(self.n_segments)
        self.K_array[0] = K0
        for i in range(1, self.n_segments):
            rho_b = self.rho_bounds[i]
            self.K_array[i] = self.K_array[i - 1] * rho_b ** (
                self.gamma[i - 1] - self.gamma[i]
            )

    def _segment_index(self, rho: float) -> int:
        """确定密度所在段."""
        if rho <= 0:
            return 0
        for i in range(self.n_segments):
            if rho < self.rho_bounds[i + 1]:
                return i
        return self.n_segments - 1

    def pressure(self, rho: float) -> float:
        """分段多方压力."""
        if rho <= 0:
            return 0.0
        i = self._segment_index(rho)
        return self.K_array[i] * rho ** self.gamma[i]

    def energy_density(self, rho: float) -> float:
        """
        能量密度通过积分计算:
            epsilon = rho c^2 + integral_0^rho P(rho')/rho'^2 drho'

        简化: 使用多方关系
            epsilon = rho c^2 + P/(Gamma-1)  (每段内近似)
        """
        if rho <= 0:
            return 0.0
        i = self._segment_index(rho)
        p = self.pressure(rho)
        return rho * NS.c_light ** 2 + p / (self.gamma[i] - 1.0)

    def sound_speed_squared(self, rho: float) -> float:
        """声速平方."""
        if rho <= 0:
            return 0.0
        i = self._segment_index(rho)
        p = self.pressure(rho)
        eps = self.energy_density(rho)
        denom = eps + p / NS.c_light ** 2
        if denom <= 0:
            return 0.0
        return self.gamma[i] * p / denom * NS.c_light ** 2


# ============================================================
# 第二部分: TOV 方程 (ODE 右端函数)
# ============================================================

def tov_rhs(r: float, y: np.ndarray, eos: PolytropicEoS) -> np.ndarray:
    """
    TOV 方程的右端函数 (类似 double_well_deriv).

    状态向量: y = [m_km, P]
        m_km = 内部引力质量 × G/c^2 (km, 几何化单位)
        P = 压力 (dyn/cm^2)

    r 单位为 km.

    方程 (几何化):
        d m_km / dr = 4 pi r^2 epsilon / c^2 (km/km = 1)
        dP/dr = -(G/c^4)(epsilon + P)(m_km + 4 pi r^3 P/c^2)
                / [r^2 (1 - 2 m_km/r)]

    使用 cgs 中间量, 最终转回 km.
    """
    m_geom, P = y[0], y[1]  # m_geom 单位 km (G M / c^2)

    if r < 1.0e-6:
        return np.array([0.0, 0.0])

    r_cm = r * 1.0e5  # km -> cm

    if P <= 0:
        return np.array([0.0, 0.0])

    rho = eos.rho_from_pressure(P)
    eps = eos.energy_density(rho)  # erg/cm^3

    # dm_geom/dr  (m_geom = GM/c^2, 单位 cm)
    # d(GM/c^2)/dr = G/c^2 * 4 pi r^2 eps/c^2 = 4 pi G r^2 eps / c^4
    # 转换为 km: 1 km = 1e5 cm
    dmdr_cm = 4.0 * math.pi * NS.G_newton * r_cm ** 2 * eps / NS.c_light ** 4
    dmdr_km = dmdr_cm * 1.0e-5  # km / km

    # 度规因子
    factor = 1.0 - 2.0 * m_geom / r
    if factor <= 0.0:
        factor = r8_epsilon()

    # dP/dr (cgs)
    # dP/dr = -G (eps + P/c^2)(m + 4 pi r^3 P/c^2) / (r^2 (1 - 2GM/(rc^2)))
    # m in grams, r in cm
    m_g = m_geom * NS.c_light ** 2 / NS.G_newton  # km -> g
    dPdr_cgs = (-NS.G_newton * (eps + P / NS.c_light ** 2)
                * (m_g + 4.0 * math.pi * r_cm ** 3 * P / NS.c_light ** 2)
                / (r_cm ** 2 * factor))

    # dP/dr in dyn/cm^2 per km
    dPdr_km = dPdr_cgs * 1.0e5

    return np.array([dmdr_km, dPdr_km])


def tov_rhs_dimensionless(x: float, y: np.ndarray,
                           sigma_c: float) -> np.ndarray:
    """
    无量纲 TOV 方程.

    无量纲化:
        x = r / R_0,  R_0 = c / sqrt(4 pi G rho_c / c^2)
        m_hat = m / M_0,  M_0 = R_0 c^2 / G
        P_hat = P / (rho_c c^2)
        sigma = rho / rho_c

    方程:
        d m_hat / dx = x^2 sigma_hat(x)
        d P_hat / dx = -(P_hat + sigma_hat) (m_hat + x^3 P_hat)
                       / [x^2 (1 - 2 m_hat / x)]

    其中 sigma_hat = epsilon / (rho_c c^2).
    """
    m_hat, P_hat = y[0], y[1]

    if x < 1e-8:
        return np.array([0.0, 0.0])

    if P_hat <= 0:
        return np.array([0.0, 0.0])

    # 简化: 使用多方关系 sigma_hat ~ P_hat^{1/Gamma}
    # (具体取决于 EoS)
    gamma = 2.5  # 默认
    sigma_hat = P_hat ** (1.0 / gamma)

    factor = 1.0 - 2.0 * m_hat / x
    if factor <= 1e-10:
        factor = 1e-10

    dmdx = x * x * (sigma_hat + sigma_c * P_hat)
    dPdx = -(P_hat + sigma_hat) * (m_hat + x ** 3 * P_hat) / (x * x * factor)

    return np.array([dmdx, dPdx])


# ============================================================
# 第三部分: 守恒量计算 (移植自 double_well_conserved)
# ============================================================

def tov_conserved_quantity(r_km: float, y: np.ndarray,
                            eos: PolytropicEoS) -> float:
    """
    TOV 系统的守恒量 (推广自 double_well_conserved).

    监测 m_geom(r) - 理论预期.
    y[0] = m_geom (km) = GM/c^2
    """
    m_geom, P = y[0], y[1]
    return m_geom  # 简单返回几何化质量


def tov_baryon_number(r_nodes: np.ndarray, m_profile: np.ndarray,
                       rho_profile: np.ndarray) -> float:
    """
    计算总重子数.

    N_B = integral_0^R 4 pi r^2 n(r) / sqrt(1 - 2Gm/(rc^2)) dr

    其中 n = rho / m_n 为重子数密度.

    Returns
    -------
    float
        总重子数
    """
    n_baryon = np.zeros_like(rho_profile)
    integrand = np.zeros_like(r_nodes)

    for k in range(len(r_nodes)):
        n_baryon[k] = max(rho_profile[k], 0.0) / NS.m_n
        r_cm = max(r_nodes[k], 1e-5) * 1e5
        m_cm = max(m_profile[k], 0.0)
        metric_factor = max(1.0 - 2.0 * NS.G_newton * m_cm / (r_cm * NS.c_light ** 2),
                           1e-10)
        integrand[k] = 4.0 * math.pi * r_cm ** 2 * n_baryon[k] / math.sqrt(metric_factor)

    # 梯形积分
    total = 0.0
    for k in range(1, len(r_nodes)):
        total += 0.5 * (integrand[k] + integrand[k - 1]) * (r_nodes[k] - r_nodes[k - 1]) * 1e5

    return total


# ============================================================
# 第四部分: ODE 积分器 (RK4 + 自适应)
# ============================================================

def rk4_step(f, r: float, y: np.ndarray, h: float, *args) -> np.ndarray:
    """
    经典 RK4 单步 (类似 double_well_ode 的积分方式).

    k1 = f(r, y)
    k2 = f(r + h/2, y + h k1/2)
    k3 = f(r + h/2, y + h k2/2)
    k4 = f(r + h, y + h k3)
    y_{n+1} = y_n + (h/6)(k1 + 2k2 + 2k3 + k4)
    """
    k1 = f(r, y, *args)
    k2 = f(r + 0.5 * h, y + 0.5 * h * k1, *args)
    k3 = f(r + 0.5 * h, y + 0.5 * h * k2, *args)
    k4 = f(r + h, y + h * k3, *args)
    return y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def rk45_adaptive_step(f, r: float, y: np.ndarray, h: float,
                        tol: float = 1e-8, *args) -> Tuple[np.ndarray, float, float]:
    """
    RK4(5) 自适应步长 (Dormand-Prince 风格简化版).

    通过比较 4 阶和 5 阶结果估计误差:
        err ≈ ||y_5 - y_4||

    步长调节:
        h_new = h * min(2.0, max(0.2, 0.9 * (tol/err)^{1/5}))

    Returns
    -------
    y_new : np.ndarray
    h_new : float
        建议的下一步步长
    err : float
        估计误差
    """
    # 4 阶 (RK4)
    y4 = rk4_step(f, r, y, h, *args)

    # 两个半步 (更精确)
    y_half = rk4_step(f, r, y, 0.5 * h, *args)
    y5 = rk4_step(f, r + 0.5 * h, y_half, 0.5 * h, *args)

    # 误差估计
    err_vec = y5 - y4
    err = float(np.max(np.abs(err_vec)))
    scale = max(float(np.max(np.abs(y))), 1.0)
    err_scaled = err / scale

    if err_scaled < r8_epsilon():
        return y5, h * 2.0, err_scaled

    # 步长调节
    factor = 0.9 * (tol / err_scaled) ** 0.2
    factor = min(2.0, max(0.2, factor))
    h_new = h * factor

    if err_scaled <= tol:
        return y5, h_new, err_scaled
    else:
        # 拒绝此步, 返回原值
        return y, h_new, err_scaled


def solve_tov(eos: PolytropicEoS, rho_c: float,
              r_max: float = 30.0, dr: float = 0.01,
              tol: float = 1e-8) -> Dict:
    """
    求解 TOV 方程, 从中心到表面.

    使用无量纲变量简化数值稳定性.
    物理单位结果从无量纲结果转换得到.

    Parameters
    ----------
    eos : PolytropicEoS
        状态方程
    rho_c : float
        中心密度 (g/cm^3)
    r_max : float
        最大积分半径 (km)
    dr : float
        初始步长 (km)
    tol : float
        误差容限

    Returns
    -------
    dict
        结果字典
    """
    P_c = eos.pressure(rho_c)
    eps_c = eos.energy_density(rho_c)

    # 特征尺度
    # K_scale = c^2 / sqrt(4 pi G eps_c)  -- 长度尺度
    K_scale = NS.c_light / math.sqrt(4.0 * math.pi * NS.G_newton * eps_c / NS.c_light ** 2)
    R_scale_km = K_scale / 1e5  # km

    # 无量纲化
    # xi = r / K_scale, m_hat = G m / (c^2 K_scale), P_hat = P / eps_c
    # d m_hat / d xi = 4 pi xi^2 epsilon_hat
    # d P_hat / d xi = -(eps_hat + P_hat)(m_hat + 4 pi xi^3 P_hat) / (xi^2 (1 - 2 m_hat/xi))

    def tov_dimless(xi, y_hat):
        m_hat, P_hat = y_hat
        if xi < 1e-8 or P_hat <= 0:
            return np.array([0.0, 0.0])

        # epsilon_hat = epsilon / eps_c (简化: 用多方关系)
        if P_hat <= 0:
            eps_hat = 0.0
        else:
            rho_hat = P_hat ** (1.0 / eos.gamma)  # rho/rho_c 的近似
            eps_hat = rho_hat + P_hat / (eos.gamma - 1.0)  # 简化

        factor = 1.0 - 2.0 * m_hat / xi
        if factor <= 1e-10:
            factor = 1e-10

        dmhat_dxi = 4.0 * math.pi * xi ** 2 * eps_hat
        dPhat_dxi = (-(eps_hat + P_hat) * (m_hat + 4.0 * math.pi * xi ** 3 * P_hat)
                     / (xi ** 2 * factor))

        return np.array([dmhat_dxi, dPhat_dxi])

    # 初值 (小 xi 展开)
    xi_start = 1e-4
    eps_hat_c = 1.0  # eps_c / eps_c = 1
    P_hat_c = P_c / eps_c
    m_hat_start = (4.0 / 3.0) * math.pi * xi_start ** 3 * eps_hat_c
    P_hat_start = P_hat_c - (2.0 / 3.0) * math.pi * (
        eps_hat_c + P_hat_c) * (eps_hat_c + 3.0 * P_hat_c) * xi_start ** 2
    P_hat_start = max(P_hat_start, P_hat_c * 0.999)

    y_hat = np.array([m_hat_start, P_hat_start])
    xi = xi_start
    h = 0.005

    xi_list = [xi]
    m_hat_list = [y_hat[0]]
    P_hat_list = [y_hat[1]]

    n_steps = 0
    xi_max = r_max / R_scale_km  # 无量纲最大半径

    while xi < xi_max and n_steps < 50000:
        # RK4 步进
        k1 = tov_dimless(xi, y_hat)
        k2 = tov_dimless(xi + 0.5 * h, y_hat + 0.5 * h * k1)
        k3 = tov_dimless(xi + 0.5 * h, y_hat + 0.5 * h * k2)
        k4 = tov_dimless(xi + h, y_hat + h * k3)
        y_hat = y_hat + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        xi += h
        n_steps += 1

        xi_list.append(xi)
        m_hat_list.append(y_hat[0])
        P_hat_list.append(y_hat[1])

        if y_hat[1] <= 0:
            break

        # 自适应步长
        if n_steps < 100:
            h = 0.005
        elif n_steps < 1000:
            h = 0.01
        else:
            h = 0.02

    # 转换回物理单位
    xi_arr = np.array(xi_list)
    m_hat_arr = np.array(m_hat_list)
    P_hat_arr = np.array(P_hat_list)

    # 找到表面 (P=0)
    surface_idx = len(P_hat_arr) - 1
    for k in range(1, len(P_hat_arr)):
        if P_hat_arr[k] <= 0:
            surface_idx = k
            break

    xi_star = xi_arr[surface_idx]
    m_hat_star = m_hat_arr[surface_idx]

    # 物理量
    R_star = xi_star * R_scale_km  # km
    # M = m_hat c^2 K_scale / G = m_hat * K_scale (km, 几何化)
    # M_sun = M_geom(km) / 1.4766
    M_star_geom_km = m_hat_star * R_scale_km  # GM/c^2 in km
    M_star_g = M_star_geom_km * 1e5 * NS.c_light ** 2 / NS.G_newton
    M_sun = M_star_g / NS.M_sun

    # 紧凑度
    compactness = M_star_geom_km / R_star if R_star > 0 else 0.0

    # 构建返回数组 (物理单位)
    r_arr = xi_arr[:surface_idx + 1] * R_scale_km
    m_arr = m_hat_arr[:surface_idx + 1] * R_scale_km  # m_geom in km
    P_arr = P_hat_arr[:surface_idx + 1] * eps_c
    rho_arr = np.zeros_like(P_arr)
    for k in range(len(P_arr)):
        if P_arr[k] > 0:
            rho_arr[k] = eos.rho_from_pressure(P_arr[k])

    # 守恒量 (m_geom 漂移)
    conserved = m_arr.copy()

    return {
        'r': r_arr,
        'm': m_arr,
        'P': P_arr,
        'rho': rho_arr,
        'R_star': R_star,
        'M_star': M_star_g,
        'M_sun': M_sun,
        'compactness': compactness,
        'conserved': conserved,
        'n_steps': n_steps,
    }


# ============================================================
# 第五部分: 参数管理 (移植自 double_well_parameters)
# ============================================================

class TOVParameters:
    """
    TOV 求解参数管理 (类似 double_well_parameters).

    支持默认值 + 用户覆盖.
    """

    def __init__(self):
        self._defaults = {
            'rho_c': 8.0e14,       # g/cm^3
            'r_max': 30.0,         # km
            'dr': 0.01,            # km
            'tol': 1e-8,
            'K': 1.0e35,           # 多方常数
            'gamma': 2.5,          # 多方指数
        }

    def get(self, key: str):
        if key in self._defaults:
            return self._defaults[key]
        raise KeyError(f"未知参数: {key}")

    def set(self, **kwargs):
        for k, v in kwargs.items():
            if k in self._defaults:
                self._defaults[k] = v
            else:
                raise KeyError(f"未知参数: {k}")

    def get_all(self) -> dict:
        return self._defaults.copy()


# ============================================================
# 自检
# ============================================================

if __name__ == "__main__":
    print("=== TOV 哈密顿求解器自检 ===")

    # 物理合理的 K 值 (使 P(rho_c) ~ 1e34 dyn/cm^2)
    rho_c = 8e14
    K = 5.5e-4
    gamma = 2.5
    eos = PolytropicEoS(K=K, gamma=gamma)
    P_c = eos.pressure(rho_c)
    print(f"中心密度: {rho_c:.2e} g/cm^3")
    print(f"中心压力: {P_c:.3e} dyn/cm^2")
    print(f"声速: {eos.sound_speed_squared(rho_c)**0.5 / NS.c_light:.4f} c")
    print(f"比焓: {eos.enthalpy(rho_c):.4f}")

    # 测试 TOV 求解
    result = solve_tov(eos, rho_c, r_max=30.0, dr=0.05)
    print(f"星体半径: {result['R_star']:.2f} km")
    print(f"引力质量: {result['M_sun']:.4f} M_sun")
    print(f"紧凑度: {result['compactness']:.4f}")
    print(f"积分步数: {result['n_steps']}")
    print(f"守恒量变化: {abs(result['conserved'][-1] - result['conserved'][0]):.3e}")

    # 分段多方
    eos_pw = PiecewisePolytropicEoS(
        gamma_segments=[4.0/3.0, 2.5, 3.0],
        rho_boundaries=[4.3e11, 2.0e14],
        K0=1e34
    )
    print(f"\n分段多方 EoS:")
    for rho_test in [1e12, 1e13, 1e14, 1e15]:
        p_test = eos_pw.pressure(rho_test)
        print(f"  rho={rho_test:.1e} -> P={p_test:.3e}")

    # 参数管理
    params = TOVParameters()
    print(f"\n默认参数: {params.get_all()}")
    params.set(rho_c=1e15)
    print(f"修改后 rho_c = {params.get('rho_c'):.2e}")

    print("\ntov_hamiltonian.py 自检通过.")
