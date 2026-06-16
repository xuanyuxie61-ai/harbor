# -*- coding: utf-8 -*-
"""
physical_constants.py
=====================
PROJECT_254 — 计算天体物理：双中子星并合与 kilonova 辐射转移

CGS 单位制下的天体物理常数、核物理常数以及从这些基本常数
导出的无量纲组合。所有常数取自 CODATA 2018 与 IAU 2015
名义太阳/地球参数。

核心导出量
----------
- 普朗克质量  M_Pl = sqrt(hbar c / G)
- 钱德拉塞卡质量  M_Ch ≈ 1.44 M_sun
- 史瓦西半径  r_s = 2 G M / c^2
- 典型中子星结合能  E_bind ~ 0.15 M_NS c^2

映射种子项目
-----------
- 049_asa239 (alnorm/gammad) → 统计分布常量的精密计算
- 所有物理量在此集中管理, 避免下游模块硬编码
"""

from __future__ import annotations
import math
from typing import Dict


# ---------------------------------------------------------------------------
# 基础物理常数 (CODATA 2018, CGS)
# ---------------------------------------------------------------------------
C_LIGHT: float = 2.99792458e10          # 光速  [cm/s]
G_GRAV: float = 6.67430e-8              # 万有引力常数  [cm^3 g^-1 s^-2]
HBAR: float = 1.054571817e-27           # 约化普朗克常数  [erg s]
K_BOLTZ: float = 1.380649e-16           # 玻尔兹曼常数  [erg/K]
M_ELECTRON: float = 9.1093837015e-28    # 电子质量  [g]
M_PROTON: float = 1.67262192369e-24     # 质子质量  [g]
M_NEUTRON: float = 1.67492749804e-24    # 中子质量  [g]
M_HYDROGEN: float = 1.67353282340e-24   # 氢原子质量  [g]
SIGMA_THOMSON: float = 6.6524587321e-25 # 汤姆孙散射截面  [cm^2]
SIGMA_SB: float = 5.670374419e-5        # Stefan-Boltzmann 常数 [erg cm^-2 s^-1 K^-4]
A_RAD: float = 7.5657e-15               # 辐射常数 a = 4 sigma/c  [erg cm^-3 K^-4]
EV_ERG: float = 1.602176634e-12         # 1 eV = ... erg
MEV_ERG: float = EV_ERG * 1.0e6         # 1 MeV = ... erg


# ---------------------------------------------------------------------------
# 天文单位 (IAU 2015 名义值)
# ---------------------------------------------------------------------------
M_SUN: float = 1.98892e33               # 太阳质量  [g]
M_EARTH: float = 5.97219e27             # 地球质量  [g]
R_SUN: float = 6.957e10                 # 太阳半径  [cm]
AU_CM: float = 1.495978707e13           # 1 AU  [cm]
PC_CM: float = 3.085677581e18           # 1 秒差距  [cm]
KPC_CM: float = 3.085677581e21          # 1 千秒差距  [cm]
MPC_CM: float = 3.085677581e24          # 1 兆秒差距  [cm]
YEAR_S: float = 3.15576e7               # 1 儒略年  [s]
DAY_S: float = 86400.0                  # 1 天  [s]


# ---------------------------------------------------------------------------
# 典型中子星参数
# ---------------------------------------------------------------------------
M_NS_TYPICAL: float = 1.4 * M_SUN       # 典型中子星质量
R_NS_TYPICAL: float = 1.2e6             # 典型中子星半径 12 km  [cm]
RHO_NUC: float = 2.8e14                 # 核饱和密度  [g/cm^3]
RHO_NS_CENTRAL: float = 8.0e14          # 典型中心密度  [g/cm^3]


# ---------------------------------------------------------------------------
# 导出量
# ---------------------------------------------------------------------------
def planck_mass() -> float:
    """普朗克质量  M_Pl = sqrt(hbar c / G)  [g]."""
    return math.sqrt(HBAR * C_LIGHT / G_GRAV)


def planck_length() -> float:
    """普朗克长度  l_Pl = sqrt(hbar G / c^3)  [cm]."""
    return math.sqrt(HBAR * G_GRAV / C_LIGHT ** 3)


def planck_time() -> float:
    """普朗克时间  t_Pl = l_Pl / c  [s]."""
    return planck_length() / C_LIGHT


def schwarzschild_radius(mass_g: float) -> float:
    """史瓦西半径  r_s = 2 G M / c^2  [cm].

    Parameters
    ----------
    mass_g : float
        天体质量 [g].
    """
    return 2.0 * G_GRAV * mass_g / C_LIGHT ** 2


def chandrasekhar_mass(mu_e: float = 2.0) -> float:
    """钱德拉塞卡质量  M_Ch ≈ 5.83 / mu_e^2  M_sun.

    Parameters
    ----------
    mu_e : float, optional
        平均每个电子对应的核子数, 默认 2 (碳/氧白矮星).
    """
    return 5.83 / (mu_e * mu_e) * M_SUN


def gravitational_timescale(mass_g: float, radius_cm: float) -> float:
    """引力时标  t_dyn = 1 / sqrt(G rho)  ~  sqrt(R^3 / G M)  [s].

    Parameters
    ----------
    mass_g    : float  天体质量 [g]
    radius_cm : float  特征半径 [cm]
    """
    return math.sqrt(radius_cm ** 3 / (G_GRAV * mass_g))


def escape_velocity(mass_g: float, radius_cm: float) -> float:
    """逃逸速度  v_esc = sqrt(2 G M / R)  [cm/s]."""
    return math.sqrt(2.0 * G_GRAV * mass_g / radius_cm)


def compactness(mass_g: float, radius_cm: float) -> float:
    """致密度  C = G M / (R c^2), 无量纲.

    对于典型中子星 C ~ 0.15 -- 0.25; 黑洞 C = 0.5.
    """
    return G_GRAV * mass_g / (radius_cm * C_LIGHT ** 2)


def eddington_luminosity(mass_g: float, x_h: float = 0.7) -> float:
    """爱丁顿光度  L_Edd = 4 pi G M m_p c / sigma_T  [erg/s].

    Parameters
    ----------
    mass_g : float  天体质量 [g]
    x_h    : float  氢质量丰度, 默认 0.7
    """
    kappa_es = 0.2 * (1.0 + x_h)  # 电子散射不透明度 [cm^2/g]
    return 4.0 * math.pi * G_GRAV * mass_g * C_LIGHT / kappa_es


def neumann_ridgeley_factor(eta: float) -> float:
    """Neumann-Ridgeley 因子, 用于修正中子星冷却光度.

    f_NR = (1 - 2 G M / (R c^2))^{-1/2} = (1 - 2 C)^{-1/2}.

    Parameters
    ----------
    eta : float  致密度 C = G M / (R c^2)
    """
    discriminant = 1.0 - 2.0 * eta
    if discriminant <= 0.0:
        return float("inf")
    return 1.0 / math.sqrt(discriminant)


def all_derived_quantities() -> Dict[str, float]:
    """以字典形式返回所有导出量 (以典型中子星为例)."""
    M = M_NS_TYPICAL
    R = R_NS_TYPICAL
    return {
        "M_Pl [g]": planck_mass(),
        "l_Pl [cm]": planck_length(),
        "t_Pl [s]": planck_time(),
        "r_s (1.4 M_sun) [cm]": schwarzschild_radius(M),
        "M_Ch [g]": chandrasekhar_mass(),
        "t_dyn (NS) [s]": gravitational_timescale(M, R),
        "v_esc (NS) [cm/s]": escape_velocity(M, R),
        "compactness C": compactness(M, R),
        "L_Edd [erg/s]": eddington_luminosity(M),
        "f_NR": neumann_ridgeley_factor(compactness(M, R)),
    }


# ---------------------------------------------------------------------------
# 自检
# ---------------------------------------------------------------------------
def _self_check() -> bool:
    """执行一系列物理量级校验, 确保常数未出现数量级错误."""
    checks = [
        (2.0 < planck_mass() / 1e-5 < 2.3, "M_Pl ~ 2.18e-5 g"),
        (1.6e-33 < planck_length() < 1.7e-33, "l_Pl ~ 1.62e-33 cm"),
        (5.3e-44 < planck_time() < 5.5e-44, "t_Pl ~ 5.39e-44 s"),
        (2.9e5 < schwarzschild_radius(M_SUN) < 3.0e5, "r_s(Sun) ~ 2.95e5 cm"),
        (9.0e-5 < gravitational_timescale(M_NS_TYPICAL, R_NS_TYPICAL) < 2.0e-3,
            "t_dyn(NS) ~ 0.1--2 ms"),
        (0.1 < compactness(M_NS_TYPICAL, R_NS_TYPICAL) < 0.3,
            "C(NS) ~ 0.15--0.25"),
    ]
    for ok, label in checks:
        if not ok:
            raise AssertionError(f"physical_constants self-check FAILED: {label}")
    return True


if __name__ == "__main__":
    _self_check()
    for k, v in all_derived_quantities().items():
        print(f"  {k:32s} = {v:16.6e}")
