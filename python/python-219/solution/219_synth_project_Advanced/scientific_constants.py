"""
scientific_constants.py
=======================

科学常数、量纲参数与物理公式模块。

该模块为 Pontryagin 多阶段最优控制框架提供底层物理常量与公式定义。
所有公式均严格遵循国际单位制 (SI), 并在航天器轨迹优化与热扩散 PDE
约束的交叉背景下使用。

核心公式:
---------
1. 标准重力参数:           mu_E = G * M_E            [m^3 / s^2]
2. 地球表面重力加速度:     g_0  = mu_E / R_E^2       [m / s^2]
3. Tsiolkovsky 理想火箭方程:
       Delta_v = I_sp * g_0 * ln(m_0 / m_f)
4. 特征时间尺度 (轨道周期):
       T_orb = 2 * pi * sqrt(a^3 / mu_E)
5. 热扩散 PDE 系数 (Fourier 导热):
       rho * c_p * dT/dt = kappa * Laplacian(T) + Q
6. 比冲与有效排气速度:
       c = I_sp * g_0
7. 无量纲数: Biot 数 Bi = h * L_c / kappa_s
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Tuple


# ---------------------------------------------------------------------------
# 1. 基本物理常数 (CODATA 2018)
# ---------------------------------------------------------------------------
G_NEWTON: float = 6.67430e-11             # 万有引力常数      [m^3 kg^-1 s^-2]
C_LIGHT: float = 2.99792458e8             # 真空光速          [m / s]
K_BOLTZMANN: float = 1.380649e-23         # Boltzmann 常数    [J / K]
H_PLANCK: float = 6.62607015e-34          # Planck 常数       [J s]
SIGMA_SB: float = 5.670374419e-8          # Stefan-Boltzmann  [W m^-2 K^-4]
R_GAS: float = 8.314462618                # 理想气体常数      [J mol^-1 K^-1]
N_AVOGADRO: float = 6.02214076e23         # Avogadro 常数     [mol^-1]


# ---------------------------------------------------------------------------
# 2. 地球参数 (WGS-84)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class EarthParameters:
    """地球物理参数集合 (WGS-84 椭球模型)."""
    mass: float = 5.9722e24               # 质量              [kg]
    radius_equatorial: float = 6.378137e6 # 赤道半径          [m]
    radius_polar: float = 6.356752e6      # 极半径            [m]
    radius_mean: float = 6.371e6          # 平均半径          [m]
    J2_perturbation: float = 1.08263e-3   # 动力学形状因子 J2
    omega_rotation: float = 7.2921159e-5  # 自转角速度        [rad / s]

    @property
    def mu(self) -> float:
        """标准重力参数  mu = G * M_E  [m^3 / s^2]."""
        return G_NEWTON * self.mass

    @property
    def g0(self) -> float:
        """海平面标准重力加速度  g_0 = mu / R_mean^2."""
        return self.mu / (self.radius_mean ** 2)

    @property
    def f_flattening(self) -> float:
        """几何扁率  f = (a - b) / a."""
        return (self.radius_equatorial - self.radius_polar) / self.radius_equatorial


EARTH: EarthParameters = EarthParameters()


# ---------------------------------------------------------------------------
# 3. 航天器默认参数
# ---------------------------------------------------------------------------
@dataclass
class SpacecraftParameters:
    """航天器 / 运载火箭参数."""
    dry_mass: float = 500.0               # 干质量              [kg]
    fuel_mass: float = 1500.0             # 燃料质量            [kg]
    I_sp_vacuum: float = 310.0            # 真空比冲            [s]
    I_sp_sea: float = 265.0               # 海平面比冲          [s]
    thrust_max: float = 25000.0           # 最大推力            [N]
    thrust_min: float = 0.0               # 最小推力            [N]
    drag_coeff: float = 0.35              # 阻力系数 C_D
    ref_area: float = 2.5                 # 参考面积            [m^2]
    thermal_mass: float = 800.0           # 热容 (m * c_p)      [J / K]
    emissivity: float = 0.85              # 表面发射率
    biot_number: float = 0.12             # Biot 数  Bi = h L_c / kappa_s

    @property
    def total_mass(self) -> float:
        return self.dry_mass + self.fuel_mass

    @property
    def exhaust_velocity(self) -> float:
        """有效排气速度  c = I_sp * g_0  [m / s]."""
        return self.I_sp_vacuum * EARTH.g0

    @property
    def mass_flow_max(self) -> float:
        """最大质量流率  mdot = T_max / c  [kg / s]."""
        return self.thrust_max / self.exhaust_velocity

    def delta_v_theoretical(self) -> float:
        """Tsiolkovsky 理想 Delta_v:
              Delta_v = I_sp * g_0 * ln(m_0 / m_f)
        """
        m0 = self.total_mass
        mf = self.dry_mass
        if mf <= 0.0:
            raise ValueError("干质量必须为正")
        return self.I_sp_vacuum * EARTH.g0 * math.log(m0 / mf)


# ---------------------------------------------------------------------------
# 4. 热扩散 PDE 参数 (Fourier 导热方程)
# ---------------------------------------------------------------------------
@dataclass
class ThermalParameters:
    """热扩散方程参数:
        rho * c_p * dT/dt = kappa * Laplacian(T) + Q
    其中 alpha = kappa / (rho * c_p) 为热扩散系数.
    """
    conductivity: float = 45.0            # 热导率 kappa     [W / (m K)]
    density: float = 7800.0               # 密度 rho         [kg / m^3]
    specific_heat: float = 500.0          # 比热 c_p         [J / (kg K)]
    heat_source: float = 1.0e4            # 内热源 Q         [W / m^3]

    @property
    def diffusivity(self) -> float:
        """热扩散系数 alpha = kappa / (rho * c_p)  [m^2 / s]."""
        return self.conductivity / (self.density * self.specific_heat)

    def characteristic_time(self, length: float) -> float:
        """特征热扩散时间  t_c = L^2 / alpha  [s]."""
        if length <= 0.0:
            raise ValueError("特征长度必须为正")
        return (length ** 2) / self.diffusivity


# ---------------------------------------------------------------------------
# 5. 轨道力学辅助函数
# ---------------------------------------------------------------------------
def orbital_period(semi_major_axis: float, mu: float = EARTH.mu) -> float:
    """轨道周期  T = 2 pi sqrt(a^3 / mu).

    Parameters
    ----------
    semi_major_axis : float
        半长轴 a  [m].
    mu : float
        中心天体标准重力参数  [m^3 / s^2].

    Returns
    -------
    float
        轨道周期  [s].
    """
    if semi_major_axis <= 0.0:
        raise ValueError("半长轴必须为正")
    return 2.0 * math.pi * math.sqrt(semi_major_axis ** 3 / mu)


def vis_viva_velocity(r: float, a: float, mu: float = EARTH.mu) -> float:
    """Vis-viva 方程:  v = sqrt(mu * (2/r - 1/a)).

    Parameters
    ----------
    r : float  当前地心距 [m].
    a : float  半长轴     [m].
    """
    if r <= 0.0 or a <= 0.0:
        raise ValueError("r, a 必须为正")
    inside = mu * (2.0 / r - 1.0 / a)
    if inside < 0.0:
        raise ValueError(f"Vis-viva 方程内部为负 ({inside:.3e}), 轨道参数不一致")
    return math.sqrt(inside)


def atmospheric_density(altitude: float) -> float:
    """指数大气模型:  rho(h) = rho_0 * exp(-h / H).

    Parameters
    ----------
    altitude : float  海拔高度  [m].
    """
    rho_0 = 1.225     # 海平面空气密度  [kg / m^3]
    H = 8500.0        # 标高              [m]
    if altitude < 0.0:
        altitude = 0.0
    return rho_0 * math.exp(-altitude / H)


# ---------------------------------------------------------------------------
# 6. 无量纲化参数 (用于数值稳定性)
# ---------------------------------------------------------------------------
@dataclass
class DimensionlessScales:
    """特征尺度, 用于无量纲化最优控制问题."""
    length_scale: float = 6.371e6         # 特征长度  [m]  (R_E)
    time_scale: float = 800.0             # 特征时间  [s]
    mass_scale: float = 2000.0            # 特征质量  [kg]

    @property
    def velocity_scale(self) -> float:
        return self.length_scale / self.time_scale

    @property
    def acceleration_scale(self) -> float:
        return self.length_scale / (self.time_scale ** 2)

    def nondimensionalize_state(
        self,
        r: float, v: float, m: float, theta: float, gamma: float
    ) -> Tuple[float, float, float, float, float]:
        """无量纲化状态 (r, v, m, theta, gamma)."""
        L, T, M = self.length_scale, self.time_scale, self.mass_scale
        return (
            r / L,
            v / (L / T),
            m / M,
            theta,
            gamma,
        )


# ---------------------------------------------------------------------------
# 7. 验证与自检
# ---------------------------------------------------------------------------
def self_check() -> Dict[str, bool]:
    """执行内部物理自检, 返回各校验项通过情况."""
    results: Dict[str, bool] = {}
    results["mu_positive"] = EARTH.mu > 0
    results["g0_near_981"] = abs(EARTH.g0 - 9.81) < 0.05
    sc = SpacecraftParameters()
    results["delta_v_positive"] = sc.delta_v_theoretical() > 0
    results["exhaust_velocity_positive"] = sc.exhaust_velocity > 0
    th = ThermalParameters()
    results["diffusivity_positive"] = th.diffusivity > 0
    results["period_positive"] = orbital_period(7.0e6) > 0
    results["atmosphere_positive"] = atmospheric_density(100.0e3) > 0
    return results


if __name__ == "__main__":
    for k, v in self_check().items():
        print(f"  {k:30s} : {'PASS' if v else 'FAIL'}")
