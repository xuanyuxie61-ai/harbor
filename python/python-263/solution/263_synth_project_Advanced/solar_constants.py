# -*- coding: utf-8 -*-
"""
solar_constants.py
------------------
物理常数与日冕-太阳风无量纲参数集.

本模块集中管理所有物理常数 (SI) 以及日冕等离子体的特征无量纲参数.
参数选取参照 Aschwanden (2004), Pneuman & Kopp (1971),
Parker (1958) 的经典日冕加热-太阳风模型.

核心公式
--------
1) 等离子体频率 (electron plasma frequency):
       omega_pe = sqrt(n_e * e^2 / (epsilon_0 * m_e))

2) 离子惯性长度 (ion inertial length):
       d_i = c / omega_pi,   omega_pi = sqrt(n_i * e^2 / (epsilon_0 * m_i))

3) 热 beta (等离子体热压与磁压之比):
       beta = (2 * mu_0 * n * k_B * T) / B^2

4) Alfvén 速度:
       v_A = B / sqrt(mu_0 * rho),   rho = n_i * m_i

5) 声速:
       c_s = sqrt(gamma * k_B * T / m_i)

6) Parker 临界半径 (transonic point):
       r_c = G M_sun / (2 c_s^2)

7) 经典 Spitzer 热导率 (parallel):
       kappa_|| = kappa_0 * T^(5/2),   kappa_0 = 9.97e-7 erg/(s K^(7/2) cm)

8) 辐射损失函数 (Rosner-Tucker-Vaiana RTV 标度律):
       Q_rad = chi * n^2 * T^alpha
   其中 chi ~ 1e-19 cgs, alpha ~ -0.5 (6e5<T<3e7 K)

9) 磁场 Reynolds 数:
       R_m = mu_0 * v * L / eta

10) 等离子体 Lundquist 数:
        S = mu_0 * v_A * L / eta

所有常量使用 numpy 数组/标量以便下游模块矢量化调用.
"""
from __future__ import annotations
import numpy as np

# ---------------- SI 基本物理常量 ----------------
PI = np.pi
SPEED_OF_LIGHT = 2.99792458e8             # c [m/s]
BOLTZMANN = 1.380649e-23                  # k_B [J/K]
ELECTRON_CHARGE = 1.602176634e-19         # e [C]
VACUUM_PERMEABILITY = 4.0e-7 * PI         # mu_0 [H/m]
VACUUM_PERMITTIVITY = 1.0 / (
    SPEED_OF_LIGHT**2 * VACUUM_PERMEABILITY
)                                         # epsilon_0 [F/m]
PROTON_MASS = 1.67262192e-27              # m_p [kg]
ELECTRON_MASS = 9.1093837e-31             # m_e [kg]
GRAVITATIONAL_CONST = 6.67430e-11         # G [m^3/(kg s^2)]

# ---------------- 太阳参考参数 ----------------
SOLAR_MASS = 1.98892e30                   # M_sun [kg]
SOLAR_RADIUS = 6.96e8                     # R_sun [m]
SOLAR_LUMINOSITY = 3.828e26               # L_sun [W]

# ---------------- 典型日冕基底部参数 ----------------
CORONA_T_BASE = 1.5e6                     # T_0 [K]
CORONA_N_BASE = 2.0e15                    # n_0 [1/m^3] (2e9 cm^-3)
CORONA_B_BASE = 5.0e-2                    # B_0 [T]   (500 G => 5e-2)

# ---------------- 派生参考量 ----------------
def alven_speed(b_field: float, n_i: float) -> float:
    """v_A = B / sqrt(mu_0 * m_i * n_i)."""
    rho = PROTON_MASS * n_i
    return b_field / np.sqrt(VACUUM_PERMEABILITY * rho)

def ion_inertial_length(n_i: float) -> float:
    """d_i = c / omega_pi, omega_pi = sqrt(n_i e^2 / (eps_0 m_i))."""
    omega_pi = np.sqrt(
        n_i * ELECTRON_CHARGE**2
        / (VACUUM_PERMITTIVITY * PROTON_MASS)
    )
    return SPEED_OF_LIGHT / omega_pi

def electron_plasma_freq(n_e: float) -> float:
    """omega_pe = sqrt(n_e e^2 / (eps_0 m_e))."""
    return np.sqrt(
        n_e * ELECTRON_CHARGE**2
        / (VACUUM_PERMITTIVITY * ELECTRON_MASS)
    )

def plasma_beta(n: float, t: float, b_field: float) -> float:
    """beta = 2 mu_0 n k_B T / B^2."""
    return (
        2.0 * VACUUM_PERMEABILITY * n * BOLTZMANN * t
        / (b_field**2 + 1.0e-40)
    )

def sound_speed(t: float, gamma: float = 5.0 / 3.0) -> float:
    """c_s = sqrt(gamma k_B T / m_p)."""
    return np.sqrt(gamma * BOLTZMANN * t / PROTON_MASS)

def parker_critical_radius(t: float, gamma: float = 5.0 / 3.0) -> float:
    """r_c = G M_sun / (2 c_s^2)  (Parker 1958)."""
    cs = sound_speed(t, gamma)
    return GRAVITATIONAL_CONST * SOLAR_MASS / (2.0 * cs**2)

def spitzer_conductivity(t: float) -> float:
    """kappa_|| = kappa_0 T^(5/2)  [W/(m K)] in SI.
    kappa_0 ~ 1.84e-5 (Spitzer 1962) for fully ionized hydrogen."""
    kappa_0 = 1.84e-5
    return kappa_0 * np.maximum(t, 1.0) ** 2.5

def radiative_loss(n: float, t: float) -> float:
    """RTV 辐射冷却: Q_rad = chi n^2 T^alpha.
    chi = 1.0e-32 (SI), alpha = -0.5 (Rosner-Tucker-Vaiana 1978)."""
    chi = 1.0e-32
    alpha = -0.5
    return chi * n**2 * np.maximum(t, 1.0) ** alpha

def lundquist_number(b_field: float, n_i: float, length: float,
                     eta_spitzer: float = 1.0e-4) -> float:
    """S = mu_0 v_A L / eta.  eta 为等效电阻率."""
    va = alven_speed(b_field, n_i)
    return VACUUM_PERMEABILITY * va * length / (eta_spitzer + 1.0e-40)

def magnetic_reynolds(v_flow: float, length: float,
                      eta_spitzer: float = 1.0e-4) -> float:
    """R_m = mu_0 v L / eta."""
    return VACUUM_PERMEABILITY * v_flow * length / (eta_spitzer + 1.0e-40)


# ---------------- 无量纲归一化 ----------------
class Normalization:
    """归一化单位制: 长度 = R_sun, 时间 = R_sun/c_s0, 密度 = n0, 温度 = T0."""

    def __init__(self, t0: float = CORONA_T_BASE,
                 n0: float = CORONA_N_BASE,
                 b0: float = CORONA_B_BASE):
        self.t0 = t0
        self.n0 = n0
        self.b0 = b0
        self.cs0 = sound_speed(t0)
        self.r0 = SOLAR_RADIUS
        self.tau0 = self.r0 / self.cs0
        self.p0 = n0 * BOLTZMANN * t0
        self.rho0 = n0 * PROTON_MASS
        self.va0 = alven_speed(b0, n0)
        self.di0 = ion_inertial_length(n0)

    def to_length(self, x_m: float) -> float:
        return x_m / self.r0

    def to_time(self, t_s: float) -> float:
        return t_s / self.tau0

    def to_speed(self, v_ms: float) -> float:
        return v_ms / self.cs0

    def to_temperature(self, t_k: float) -> float:
        return t_k / self.t0

    def to_density(self, n_m3: float) -> float:
        return n_m3 / self.n0

    def to_bfield(self, b_t: float) -> float:
        return b_t / self.b0
