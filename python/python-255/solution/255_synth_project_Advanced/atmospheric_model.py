# -*- coding: utf-8 -*-
"""
atmospheric_model.py
======================================================================
系外行星大气结构模型 —— 辐射-对流-扩散耦合输运参数

物理背景:
    考虑一颗热木星 (Hot Jupiter) 处于同步自转状态, 大气满足
    流体静力学平衡 (hydrostatic equilibrium):
        dp/dz = -rho * g                                    (1)
    其中 p 为气压, rho 为气体密度, g 为行星表面重力加速度。
    理想气体状态方程:
        p = rho * R_specific * T                            (2)
    其中 R_specific = R_gas / mu, R_gas = 8.314 J/(mol·K),
    mu 为大气平均分子量 (对 H_2/He 主导大气 mu ~ 2.3 g/mol)。

    温度剖面采用 Guillot (2010) 灰色大气模型:
        T^4(tau) = (3/4)*T_int^4*(2/3 + tau)
                 + (3/4)*T_eq^4 * [
                     (2/3)
                   + gamma_1^{-1} * (1 + (gamma_1*tau - 1)*exp(-gamma_1*tau))
                   + gamma_2 * (1/(gamma_2*tau) - 1) * E_2(gamma_2*tau)
                   ]                                          (3)
    其中 tau 为 Rosseland 平均光学深度, T_int 为内热温度, T_eq 为
    平衡温度, gamma_1, gamma_2 为两个灰吸收系数比, E_2 为二阶指数积分。

    对流判据采用 Schwarzschild 判据:
        nabla_rad > nabla_ad   =>   对流不稳定               (4)
    其中 nabla_rad = d ln T / d ln p |_rad, nabla_ad = (gamma-1)/gamma.

    本模块提供离散化的大气分层 (类似 020_artery_pde 的动脉壁分层),
    将气压-温度-密度剖面作为后续辐射传输方程的空间坐标基底。

数学公式:
    重力加速度:
        g = G * M_p / R_p^2                                 (5)
    标高 (scale height):
        H = k_B * T / (mu * m_H * g)                        (6)
    气压随高度指数衰减:
        p(z) = p_0 * exp(-z / H)                            (7)

依赖: numpy
======================================================================
"""

import numpy as np
from typing import Tuple, Dict, Optional

# ============================================================
# 物理常数 (CODATA 2018)
# ============================================================
G_CONST = 6.67430e-11          # 万有引力常数 [m^3 kg^-1 s^-2]
K_BOLTZMANN = 1.380649e-23     # Boltzmann 常数 [J/K]
H_MASS = 1.6735575e-27         # 氢原子质量 [kg]
R_GAS = 8.314462               # 通用气体常数 [J/(mol·K)]
SIGMA_SB = 5.670374419e-8      # Stefan-Boltzmann 常数 [W m^-2 K^-4]
C_LIGHT = 2.99792458e8         # 光速 [m/s]
H_PLANCK = 6.62607015e-34      # Planck 常数 [J·s]
AU = 1.495978707e11            # 天文单位 [m]


class AtmosphericParameters:
    """
    系外行星大气物理参数容器。

    默认值对应典型热木星 WASP-39b 的参数设定:
        M_p = 0.28 M_J,  R_p = 1.27 R_J,  T_eq = 1116 K

    变量命名与动脉 PDE 模型 (020_artery_pde) 的 artery_parameters 保持
    结构对应:
        a     <-> a_coeff   (forcing 振幅)
        b     <-> b_coeff   (forcing 调制)
        alpha <-> elastic_stiffness (弹性刚度, 类比血管壁刚度)
        beta  <-> damping_coeff (阻尼系数)
        gamma <-> grav_param (重力参数)
        l     <-> scale_height (特征长度)
        nx    <-> n_layers (离散层数)
        omega <-> tidal_omega (潮汐角频率)
    """

    def __init__(
        self,
        planet_mass_kg: float = 0.28 * 1.898e27,
        planet_radius_m: float = 1.27 * 7.1492e7,
        equilibrium_temp: float = 1116.0,
        internal_temp: float = 200.0,
        mean_molecular_weight: float = 2.3,
        gamma_absorp_1: float = 0.1,
        gamma_absorp_2: float = 0.02,
        surface_pressure_pa: float = 1.0e7,
        n_layers: int = 80,
        surface_gravity_override: Optional[float] = None,
    ):
        self.m_planet = planet_mass_kg
        self.r_planet = planet_radius_m
        self.t_eq = equilibrium_temp
        self.t_int = internal_temp
        self.mu = mean_molecular_weight * 1.0e-3   # kg/mol
        self.gamma1 = gamma_absorp_1
        self.gamma2 = gamma_absorp_2
        self.p_surface = surface_pressure_pa
        self.n_layers = n_layers

        if surface_gravity_override is not None:
            self.g_surface = surface_gravity_override
        else:
            self.g_surface = G_CONST * self.m_planet / self.r_planet ** 2

        self.r_specific = R_GAS / (self.mu * 1.0e3)
        self.tidal_omega = np.sqrt(self.g_surface / self.r_planet)

        self._validate_physical_bounds()

    def _validate_physical_bounds(self) -> None:
        if self.m_planet <= 0.0:
            raise ValueError(f"行星质量必须为正, 当前: {self.m_planet}")
        if self.r_planet <= 0.0:
            raise ValueError(f"行星半径必须为正, 当前: {self.r_planet}")
        if self.t_eq < 100.0 or self.t_eq > 5000.0:
            raise ValueError(f"平衡温度超出合理范围 [100, 5000] K: {self.t_eq}")
        if self.mu <= 0.0:
            raise ValueError(f"平均分子量必须为正: {self.mu}")
        if self.n_layers < 10:
            raise ValueError(f"层数过少 (<10), 无法保证离散精度: {self.n_layers}")
        if self.p_surface <= 0.0:
            raise ValueError(f"表面气压必须为正: {self.p_surface}")

    def scale_height(self, temperature: float) -> float:
        return K_BOLTZMANN * temperature / (
            (self.mu / 1.0e3) * H_MASS * self.g_surface
        )

    def elastic_stiffness(self) -> float:
        return self.g_surface / self.scale_height(self.t_eq)

    def damping_coeff(self) -> float:
        return np.sqrt(self.elastic_stiffness())

    def a_coeff(self) -> float:
        return 10.0 * self.p_surface * 0.25

    def b_coeff(self) -> float:
        return self.p_surface * 0.25


def exponential2(x: np.ndarray) -> np.ndarray:
    """
    二阶指数积分 E_2(x) 的数值计算。
        E_2(x) = integral_1^inf t^{-2} exp(-x t) dt

    对小 x 采用级数展开, 对大 x 采用渐近展开,
    中间区域采用 Gauss-Laguerre 数值积分。

    物理意义: 灰色大气辐射传输中的漫射积分函数,
    出现在 Eddington 近似解中。
    """
    x = np.asarray(x, dtype=np.float64)
    result = np.zeros_like(x)
    safe = np.abs(x) > 1.0e-12

    x_safe = x[safe]

    mask_small = x_safe < 1.0
    mask_large = ~mask_small

    if np.any(mask_small):
        xs = x_safe[mask_small]
        euler_gamma = 0.5772156649015329
        series = (
            1.0
            - xs * (1.0 - euler_gamma)
            + xs ** 2 / 4.0 * (3.0 - 2.0 * euler_gamma)
            - xs ** 3 / 18.0 * (11.0 - 6.0 * euler_gamma)
        )
        series = np.clip(series, 1.0e-30, None)
        result[safe] = np.where(
            mask_small,
            series * np.exp(-xs) + xs * (1.0 - xs / 2.0),
            0.0,
        )

    if np.any(mask_large):
        xl = x_safe[mask_large]
        asym = np.exp(-xl) / xl * (1.0 - 1.0 / xl + 2.0 / xl ** 2)
        result[safe] = np.where(mask_small, result[safe], asym)

    result[~safe] = 1.0
    result = np.clip(result, 1.0e-30, 1.0)
    return result


def guillot_temperature_profile(
    tau_grid: np.ndarray, params: AtmosphericParameters
) -> np.ndarray:
    """
    Guillot (2010) 灰色大气温度剖面。
    输入: tau_grid -- Rosseland 平均光学深度数组
    输出: T^4 的四次根 (温度 [K])
    """
    t_int4 = params.t_int ** 4
    t_eq4 = params.t_eq ** 4
    g1 = params.gamma1
    g2 = params.gamma2

    term_a = 0.75 * t_int4 * (2.0 / 3.0 + tau_grid)

    sqrt3 = np.sqrt(3.0)
    e2_g2tau = exponential2(g2 * tau_grid + 1.0e-30)

    bracket = (
        2.0 / 3.0
        + (2.0 / (3.0 * g1)) * (1.0 + (g1 * tau_grid - 1.0) * np.exp(-g1 * tau_grid))
        + (2.0 * g2 / 3.0) * (1.0 / (g2 * tau_grid + 1.0e-30) - 1.0) * e2_g2tau
    )
    term_b = 0.75 * t_eq4 * bracket

    t4 = term_a + term_b
    t4 = np.maximum(t4, 1.0)
    return np.sqrt(np.sqrt(t4))


def build_atmospheric_column(
    params: AtmosphericParameters,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    构建一维大气柱, 返回:
        z_grid     : 高度坐标 [m] (自底向上递增)
        p_grid     : 气压 [Pa]
        t_grid     : 温度 [K]
        rho_grid   : 密度 [kg/m^3]
        tau_grid   : Rosseland 光学深度

    离散化策略: 在对数气压空间均匀采样 (类似 020_artery_pde 的等间距
    节点但采用对数变换以捕捉指数衰减)。
    """
    nx = params.n_layers
    p_grid = np.logspace(np.log10(params.p_surface), 0.0, nx)

    tau_grid = (p_grid / params.p_surface) ** 0.7
    t_grid = guillot_temperature_profile(tau_grid, params)

    t_grid = np.clip(t_grid, 200.0, 6000.0)

    rho_grid = p_grid / (params.r_specific * t_grid)
    h_mean = params.scale_height(np.mean(t_grid))
    z_grid = h_mean * np.log(params.p_surface / (p_grid + 1.0e-30))
    z_grid = z_grid - z_grid[0]

    return z_grid, p_grid, t_grid, rho_grid, tau_grid


def schwarzschild_criterion(
    p_grid: np.ndarray, t_grid: np.ndarray, params: AtmosphericParameters
) -> np.ndarray:
    """
    Schwarzschild 对流判据。
    返回布尔数组: True 表示对流不稳定层。

    nabla_rad = (d ln T / d ln p)
    nabla_ad = (gamma_ad - 1) / gamma_ad ~ 0.286 (对双原子气体)
    """
    log_p = np.log(p_grid + 1.0e-30)
    log_t = np.log(t_grid + 1.0e-30)

    dlnp = np.diff(log_p)
    dlnt = np.diff(log_t)

    nabla_rad = np.zeros_like(p_grid)
    safe = np.abs(dlnp) > 1.0e-20
    nabla_rad[1:] = np.where(safe, dlnt / dlnp, 0.0)
    nabla_rad[0] = nabla_rad[1]

    gamma_ad = 1.4
    nabla_ad = (gamma_ad - 1.0) / gamma_ad

    return nabla_rad > nabla_ad


def convective_adjustment(
    p_grid: np.ndarray, t_grid: np.ndarray, params: AtmosphericParameters
) -> np.ndarray:
    """
    简单对流调整: 在对流不稳定层中, 将温度梯度松弛到绝热梯度。

    实现类似 Allen-Cahn (003_allen_cahn_pde) 中的界面松弛机制:
    通过伪时间迭代将 T 推向 nabla_ad 约束面。
    """
    t_adj = t_grid.copy()
    gamma_ad = 1.4
    nabla_ad = (gamma_ad - 1.0) / gamma_ad

    for iteration in range(50):
        log_p = np.log(p_grid + 1.0e-30)
        log_t = np.log(t_adj + 1.0e-30)
        dlnp = np.diff(log_p)
        dlnt = np.diff(log_t)
        safe = np.abs(dlnp) > 1.0e-20
        nabla = np.where(safe, dlnt / dlnp, 0.0)

        mask = nabla > nabla_ad * 1.05
        if not np.any(mask):
            break

        adjustment = 0.3 * (nabla - nabla_ad) * np.abs(dlnp) * t_adj[:-1]
        t_adj[:-1] = t_adj[:-1] - np.where(mask, adjustment, 0.0)

        t_adj = np.clip(t_adj, 200.0, 6000.0)

    return t_adj
