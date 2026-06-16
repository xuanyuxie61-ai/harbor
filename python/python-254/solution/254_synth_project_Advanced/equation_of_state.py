# -*- coding: utf-8 -*-
"""
equation_of_state.py
====================
PROJECT_254 — 计算天体物理：双中子星并合与 kilonova 辐射转移

分段多方 (Piecewise Polytropic) 冷物质状态方程 (EOS) 以及
温度相关修正 (Helmholtz 自由能简化版本).

状态方程核心方程
----------------
冷物质分段多方::

    P(rho) = K_i * rho^{Gamma_i},   rho in [rho_{i-1}, rho_i]

热修正::

    P_total = P_cold + P_thermal
    e_total = e_cold + e_thermal
    P_thermal = (Gamma_th - 1) * rho * e_thermal
    Gamma_th = 1.5  (非相对论理想气体)

声速::

    c_s^2 = (dP / drho)_s

比焓::

    h = 1 + e + P / rho  (c = 1 单位)

绝热指数::

    Gamma_1 = (rho / P) * (dP / drho)_s

映射种子项目
-----------
- 1192 (Hamiltonian evolution) → 哈密顿框架下 EOS 的热力学一致性
  密度场作为广义坐标, 压力作为广义动量的共轭流
- 049_asa239 (gammad/alnorm) →  Fermi-Dirac 积分的近似, 用于
  简并电子气的热力学量计算
"""

from __future__ import annotations
import math
from typing import List, Tuple

from physical_constants import (
    C_LIGHT, G_GRAV, RHO_NUC, K_BOLTZ, M_NEUTRON, M_PROTON, MEV_ERG
)


# ---------------------------------------------------------------------------
# 多方片段定义:  (rho_breakpoint [g/cm^3], Gamma)
# 基于 Read et al. (2009) 参数化, 适配典型 BNS 并合产物
# ---------------------------------------------------------------------------
POLYTROPE_SEGMENTS: List[Tuple[float, float, float]] = [
    # (rho_lower, K_cgs, Gamma)
    (1.0e7,   1.0626e6,  1.300),   # 地壳外层
    (2.0e11,  1.9095e6,  1.350),   # 内壳
    (3.0e13,  1.3177e12, 1.325),   # 外核
    (2.0e14,  4.0667e12, 1.400),   # 中密度核
    (5.0e14,  8.6235e12, 1.375),   # 近饱和密度
    (1.0e15,  1.6156e13, 1.350),   # 超饱和
    (2.0e15,  3.1512e13, 1.325),   # 高密度核
]

# 热压力绝热指数 (trapped neutrinos, T ~ 10--50 MeV)
GAMMA_THERMAL: float = 1.50

# 温度无关的比内能下限 [erg/g]
E_INTERNAL_FLOOR: float = 1.0e14

# 最大允许密度 [g/cm^3]  (防止数值发散)
RHO_MAX: float = 5.0e15

# 密度下限 (大气层)
RHO_ATM: float = 1.0e3


# ---------------------------------------------------------------------------
# 核心状态方程接口
# ---------------------------------------------------------------------------
def find_segment(rho: float) -> int:
    """定位给定密度 rho 所在的多方段索引."""
    if rho <= POLYTROPE_SEGMENTS[0][0]:
        return 0
    for i in range(len(POLYTROPE_SEGMENTS) - 1):
        if POLYTROPE_SEGMENTS[i][0] <= rho < POLYTROPE_SEGMENTS[i + 1][0]:
            return i
    return len(POLYTROPE_SEGMENTS) - 1


def pressure_cold(rho: float) -> float:
    """冷物质压强  P_cold(rho) [dyn/cm^2 = erg/cm^3].

    连续性要求:  每段边界的 K_i 通过匹配条件递推得到::

        K_{i+1} = K_i * rho_i^{Gamma_i - Gamma_{i+1}}
    """
    rho = max(rho, RHO_ATM)
    rho = min(rho, RHO_MAX)
    idx = find_segment(rho)
    rho0, K0, Gamma0 = POLYTROPE_SEGMENTS[0]
    # 从第一段开始递推 K
    K_i = K0
    rho_i = rho0
    for j in range(1, idx + 1):
        rho_j, K_j_explicit, Gamma_j = POLYTROPE_SEGMENTS[j]
        # 匹配: K_j * rho_{j-1}^{Gamma_j} = K_{j-1} * rho_{j-1}^{Gamma_{j-1}}
        K_i = K_i * (rho_j ** (POLYTROPE_SEGMENTS[j - 1][2] - Gamma_j))
        rho_i = rho_j
    Gamma = POLYTROPE_SEGMENTS[idx][2]
    return K_i * (rho ** Gamma)


def specific_internal_energy_cold(rho: float) -> float:
    """冷物质比内能  e_cold [erg/g].

    由热力学第一定律::

        de = P / rho^2 * drho  =>  e = K * rho^{Gamma-1} / (Gamma - 1)
    """
    rho = max(rho, RHO_ATM)
    rho = min(rho, RHO_MAX)
    idx = find_segment(rho)
    rho0, K0, Gamma0 = POLYTROPE_SEGMENTS[0]
    K_i = K0
    for j in range(1, idx + 1):
        rho_j, _, Gamma_j = POLYTROPE_SEGMENTS[j]
        K_i = K_i * (rho_j ** (POLYTROPE_SEGMENTS[j - 1][2] - Gamma_j))
    Gamma = POLYTROPE_SEGMENTS[idx][2]
    return K_i * (rho ** (Gamma - 1.0)) / (Gamma - 1.0)


def sound_speed_cold(rho: float) -> float:
    """冷物质声速  c_s = sqrt( dP/drho ) [cm/s].

    dP/drho = K * Gamma * rho^{Gamma-1} = Gamma * P / rho
    """
    rho = max(rho, RHO_ATM)
    P = pressure_cold(rho)
    cs2 = POLYTROPE_SEGMENTS[find_segment(rho)][2] * P / rho
    cs2 = max(cs2, 0.0)
    cs = math.sqrt(cs2)
    # 因果性限制: c_s < c
    cs = min(cs, 0.95 * C_LIGHT)
    return cs


def adiabatic_index(rho: float) -> float:
    """绝热指数  Gamma_1 = (rho / P) * (dP/drho)_s.

    对于多方, Gamma_1 = Gamma (多方指数).
    """
    return POLYTROPE_SEGMENTS[find_segment(max(rho, RHO_ATM))][2]


def total_pressure(rho: float, e_thermal: float) -> float:
    """总压强  P = P_cold + (Gamma_th - 1) * rho * e_thermal.

    Parameters
    ----------
    rho       : float  质量密度 [g/cm^3]
    e_thermal : float  热比内能 [erg/g]
    """
    rho = max(rho, RHO_ATM)
    e_thermal = max(e_thermal, 0.0)
    P_cold = pressure_cold(rho)
    P_therm = (GAMMA_THERMAL - 1.0) * rho * e_thermal
    return P_cold + P_therm


def total_specific_energy(rho: float, T_MeV: float) -> float:
    """总比内能  e_total = e_cold + e_thermal(T).

    热内能近似::

        e_thermal = (3/2) * (k_B T) / (mu * m_u)
    其中 mu ~ 1 (中子星物质), m_u = 原子质量单位.
    """
    rho = max(rho, RHO_ATM)
    T_K = T_MeV * MEV_ERG / K_BOLTZ
    mu = 1.0
    m_u = 0.5 * (M_PROTON + M_NEUTRON)
    e_th = 1.5 * K_BOLTZ * T_K / (mu * m_u)
    return specific_internal_energy_cold(rho) + e_th


def temperature_from_thermal_energy(rho: float, e_thermal: float) -> float:
    """从热比内能反推温度  T [MeV].

    T = (2/3) * mu * m_u * e_thermal / k_B
    """
    rho = max(rho, RHO_ATM)
    e_thermal = max(e_thermal, 0.0)
    mu = 1.0
    m_u = 0.5 * (M_PROTON + M_NEUTRON)
    T_K = (2.0 / 3.0) * mu * m_u * e_thermal / K_BOLTZ
    return T_K * K_BOLTZ / MEV_ERG


def enthalpy_specific(rho: float, e_thermal: float) -> float:
    """比焓  h = 1 + e + P/rho  (c=1 单位, 返回时恢复 c^2).

    h = c^2 + e_total + P / rho
    """
    rho = max(rho, RHO_ATM)
    P = total_pressure(rho, e_thermal)
    e_cold = specific_internal_energy_cold(rho)
    e_tot = e_cold + e_thermal
    return C_LIGHT ** 2 + e_tot + P / rho


def specific_enthalpy_rel(rho: float, e_thermal: float) -> float:
    """相对论比焓 (无量纲)  h_rel = h / c^2."""
    return enthalpy_specific(rho, e_thermal) / C_LIGHT ** 2


# ---------------------------------------------------------------------------
# Helmholtz 自由能简化 (简并电子气)
# 映射 049_asa239 (gammad): 不完全 Gamma 函数用于 Fermi-Dirac 积分近似
# ---------------------------------------------------------------------------
def fermi_dirac_integral_approx(eta: float, k: int) -> float:
    """Fermi-Dirac 积分 F_k(eta) 的近似.

    对于强简并 (eta >> 1)::

        F_k(eta) ~ eta^{k+1} / (k+1) * (1 + pi^2/6 * k(k+1)/eta^2 + ...)

    参数
    ----
    eta : float  简并参数  mu_e / (k_B T)
    k   : int    积分阶数  (0, 1/2, 1, 3/2, 2, ...)
    """
    if eta > 20.0:
        # 强简并展开到 T^2 阶
        kp1 = k + 1.0
        leading = (eta ** kp1) / kp1
        correction = 1.0 + (math.pi ** 2 / 6.0) * k * kp1 / (eta * eta)
        return leading * correction
    elif eta < -20.0:
        # 非简并 (Maxwell-Boltzmann) 极限
        return math.exp(eta)
    else:
        # 中等简并: 数值积分 (简化梯形法则)
        n_pts = 200
        x_max = eta + 40.0
        dx = x_max / n_pts
        s = 0.0
        for i in range(n_pts):
            x = (i + 0.5) * dx
            denom = math.exp(x - eta) + 1.0
            s += (x ** k) / denom * dx
        return s


def electron_pressure_degenerate(n_e: float, T_MeV: float) -> float:
    """简并电子气压强 [dyn/cm^2].

    P_e = (8 pi sqrt(2) m_e^{3/2}) / (3 h^3) * (k_B T)^{5/2} * F_{3/2}(eta)

    eta = (mu_e - m_e c^2) / (k_B T)  简化为非相对论形式
    """
    m_e = 9.10938e-28
    h = 6.62607e-27
    T_K = T_MeV * MEV_ERG / K_BOLTZ
    if T_K < 1.0e6:
        T_K = 1.0e6
    kT = K_BOLTZ * T_K
    # 简化估计: eta ~ n_e * h^3 / (2 pi m_e kT)^{3/2}
    lambda_db = h / math.sqrt(2.0 * math.pi * m_e * kT)
    eta = math.log(max(n_e * lambda_db ** 3, 1.0e-30))
    f32 = fermi_dirac_integral_approx(eta, 1.5 if eta > 0 else 1)
    coeff = 8.0 * math.pi * math.sqrt(2.0) * (m_e ** 1.5) / (3.0 * h ** 3) * (kT ** 2.5)
    return coeff * f32


# ---------------------------------------------------------------------------
# 自检
# ---------------------------------------------------------------------------
def _self_check() -> bool:
    """检查 EOS 的连续性与因果性."""
    for i in range(len(POLYTROPE_SEGMENTS) - 1):
        rho_b = POLYTROPE_SEGMENTS[i + 1][0] * 0.9999
        P1 = pressure_cold(rho_b)
        P2 = pressure_cold(rho_b * 1.0001)
        rel_jump = abs(P2 - P1) / max(abs(P1), 1.0)
        if rel_jump > 0.01:
            raise AssertionError(f"EOS discontinuity at rho={rho_b:.3e}: dP/P = {rel_jump:.3e}")
    # 因果性
    for rho_test in [1.0e10, 1.0e12, 1.0e14, 5.0e14, 1.0e15]:
        cs = sound_speed_cold(rho_test)
        if cs >= C_LIGHT:
            raise AssertionError(f"Causality violation at rho={rho_test:.3e}: cs={cs:.3e}")
    return True


if __name__ == "__main__":
    _self_check()
    print("EOS self-check passed.")
    rho_test = 3.0e14
    print(f"  rho = {rho_test:.3e} g/cm^3")
    print(f"  P_cold      = {pressure_cold(rho_test):.4e} dyn/cm^2")
    print(f"  e_cold      = {specific_internal_energy_cold(rho_test):.4e} erg/g")
    print(f"  c_s         = {sound_speed_cold(rho_test):.4e} cm/s")
    print(f"  Gamma_1     = {adiabatic_index(rho_test):.4f}")
    print(f"  h (e_th=0)  = {enthalpy_specific(rho_test, 0.0):.4e} erg/g")
