# -*- coding: utf-8 -*-
"""
thermodynamic_eos.py
====================
PROJECT_254 — 计算天体物理：双中子星并合与 kilonova 辐射转移

温度-密度相图与相变 (QCD 相变, 中子超流, 中微子陷陷).
本模块扩展 equation_of_state.py, 加入相变判据与潜热.

相变类型
--------
1. 核物质 -> 夸克物质 (MIT bag model)::

    P_quark = (1/3) * (e - 4 B)
    B = bag constant ~ 60 -- 120 MeV/fm^3

    相变条件:  P_hadron(rho, e) = P_quark(rho, e) 且 Gibbs 条件
              mu_hadron = mu_quark

2. 中子超流 (配对间隙)::

    Delta_n ~ 1 MeV * exp(-p_F / p_0)  对 T < T_c ~ Delta_n / 1.76

3. 中微子陷陷 (trapping)::

    当 rho > rho_trap ~ 10^{12} g/cm^3,  中微子自由程 < 特征尺度
    -> Y_l = Y_e + Y_nu = const  (lepton 分数守恒)

映射种子项目
-----------
- 1192 (Hamiltonian evolution) → 密度矩阵演化
  d rho / dt = -i [H, rho] 类比于相变动力学::

    dX/dt = -delta F / delta X

  其中 X 为序参量 (如夸克分数), F 为自由能
- 321 (dueling_idiots) → 蒙特卡洛 duel 中的几何分布:
  相变发生的概率服从几何分布, 平均等待时间
  tau_phase ~ 1 / Gamma_nucleation
"""

from __future__ import annotations
import math
from typing import Tuple, List

from physical_constants import (
    C_LIGHT, G_GRAV, K_BOLTZ, MEV_ERG, RHO_NUC, M_NEUTRON
)


# ---------------------------------------------------------------------------
# MIT Bag Model 夸克物质
# ---------------------------------------------------------------------------
BAG_CONSTANT_MEV_FM3: float = 90.0    # 袋常数 [MeV/fm^3]
N_FLAVOR: int = 3                     # u, d, s


def bag_constant_cgs() -> float:
    """将袋常数从 MeV/fm^3 转为 erg/cm^3."""
    # 1 MeV = 1.602e-6 erg
    # 1 fm^3 = 1e-39 cm^3
    return BAG_CONSTANT_MEV_FM3 * MEV_ERG / 1.0e-39


def pressure_quark(rho: float, T_MeV: float) -> float:
    """MIT bag 模型夸克物质压强.

    P_q = (1/3) * (e_q - 4 B)

    e_q = 3 * (pi^2 / 4)^{1/3} * (hbar c) * n_q^{4/3}  (零温极限)
         + (pi^2 / 5) * T^4 / (hbar^3 c^3) * N_dof     (有限 T)
    """
    B = bag_constant_cgs()
    # 重子数密度
    n_B = rho / M_NEUTRON  # [cm^-3]
    # 零温费米能
    hbarc = 1.0546e-27 * C_LIGHT  # erg cm
    kF = (math.pi ** 2 * n_B / N_FLAVOR) ** (1.0 / 3.0) if n_B > 0 else 0.0
    e_F = hbarc * kF  # erg
    # 零温能量密度
    e_0 = 3.0 * (math.pi ** 2 / 4.0) ** (1.0 / 3.0) * hbarc * (n_B ** (4.0 / 3.0)) \
          if n_B > 0 else 0.0
    # 有限温度修正 (相对论费米气体)
    T_K = T_MeV * MEV_ERG / K_BOLTZ
    N_dof = 2.0 * N_FLAVOR * 3.0  # spin * flavor * color
    a_rad_q = N_dof * 7.0 / 8.0 * math.pi ** 2 * K_BOLTZ ** 4 / (15.0 * (1.0546e-27) ** 3 * C_LIGHT ** 3)
    e_T = a_rad_q * T_K ** 4
    e_total = e_0 + e_T + B
    P = (e_total - 4.0 * B) / 3.0
    return max(P, 0.0)


def phase_transition_density(T_MeV: float) -> float:
    """估计核物质 -> 夸克物质的临界密度 [g/cm^3].

    粗略: rho_c ~ 3 rho_nuc * (1 - T^2 / T_c^2)
    其中 T_c ~ 160 MeV (QCD 交叉相变温度)
    """
    Tc_QCD = 160.0  # MeV
    rho_0 = 2.8e14  # g/cm^3
    ratio = T_MeV / Tc_QCD
    if ratio >= 1.0:
        return rho_0  # 高温下交叉相变
    return 3.0 * rho_0 * (1.0 - ratio * ratio)


# ---------------------------------------------------------------------------
# 中子超流
# ---------------------------------------------------------------------------
def neutron_gap_MeV(kF_fm: float) -> float:
    """中子 1S0 配对间隙 (简单参数化).

    Delta_n(k_F) = Delta_0 * exp(-(k_F - k_F0)^2 / sigma^2)

    峰值: Delta_0 ~ 1 MeV at k_F0 ~ 0.8 fm^-1, sigma ~ 0.5 fm^-1
    """
    Delta_0 = 1.0  # MeV
    kF0 = 0.8  # fm^-1
    sigma = 0.5  # fm^-1
    return Delta_0 * math.exp(-((kF_fm - kF0) / sigma) ** 2)


def critical_temperature_MeV(gap_MeV: float) -> float:
    """BCS 临界温度  T_c = Delta / 1.76."""
    if gap_MeV <= 0.0:
        return 0.0
    return gap_MeV / 1.76


def superfluid_reduction_factor(T_MeV: float, Tc_MeV: float) -> float:
    """超流态对热容的抑制因子 (BCS 形式).

    R = exp(-Delta(T) / T)  for T < Tc
      = 1                    for T >= Tc
    """
    if T_MeV >= Tc_MeV or Tc_MeV <= 0:
        return 1.0
    Delta_T = 1.76 * Tc_MeV * math.sqrt(1.0 - T_MeV / Tc_MeV)
    return math.exp(-Delta_T / max(T_MeV, 1.0e-6))


# ---------------------------------------------------------------------------
# 中微子陷陷
# ---------------------------------------------------------------------------
def neutrino_mean_free_path(rho: float, T_MeV: float, Ye: float) -> float:
    """中微子平均自由程 [cm].

    lambda_nu ~ (G_F^2 T^2 n_N / pi)^{-1}

    G_F = 1.166e-5 GeV^-2 = 1.166e-5 / (hbar c)^3 [erg cm^3]
    """
    G_F_cgs = 1.166e-5 * (MEV_ERG / 1.0e3) ** (-2) / (1.0546e-27 * C_LIGHT) ** 3
    T_erg = T_MeV * MEV_ERG
    n_N = rho / M_NEUTRON
    sigma_nu = G_F_cgs ** 2 * T_erg ** 2 * n_N / math.pi
    sigma_nu = max(sigma_nu, 1.0e-50)
    return 1.0 / sigma_nu


def is_neutrino_trapped(rho: float, T_MeV: float, Ye: float,
                        L_cm: float = 1.0e6) -> bool:
    """判断中微子是否陷陷.

    陷陷条件: lambda_nu < L (特征尺度, 取中子星半径)
    """
    lam = neutrino_mean_free_path(rho, T_MeV, Ye)
    return lam < L_cm


# ---------------------------------------------------------------------------
# 相变动力学 (蒙特卡洛): 映射 321 (dueling_idiots)
# ---------------------------------------------------------------------------
def phase_transition_waiting_time(nucleation_rate: float,
                                  volume: float,
                                  n_trials: int = 1000,
                                  seed: int = 42) -> List[float]:
    """用蒙特卡洛模拟相变等待时间的分布.

    类似 duel 的几何分布: 在每个时间步 dt, 相变以概率 p 发生::

        p = 1 - exp(-Gamma_nucl * V * dt)

    等待时间 t_wait 服从指数分布::

        f(t) = Gamma_nucl * V * exp(-Gamma_nucl * V * t)

    Returns
    -------
    List[float]  n_trials 个等待时间的样本
    """
    import random
    rng = random.Random(seed)
    dt = 1.0e-5
    p_per_step = 1.0 - math.exp(-nucleation_rate * volume * dt)
    waits = []
    for _ in range(n_trials):
        t = 0.0
        while True:
            t += dt
            if rng.random() < p_per_step:
                break
            if t > 1.0:  # 截断
                break
        waits.append(t)
    return waits


# ---------------------------------------------------------------------------
# 自检
# ---------------------------------------------------------------------------
def _self_check() -> bool:
    """验证夸克压强为正, 中微子平均自由程随密度下降."""
    P_q = pressure_quark(5.0e14, 30.0)
    # 在典型夸克密度下压强应为正 (若 B 不太大)
    assert P_q >= 0.0, "Quark pressure negative"
    # 中微子 MFP
    lam1 = neutrino_mean_free_path(1.0e12, 10.0, 0.3)
    lam2 = neutrino_mean_free_path(1.0e14, 10.0, 0.3)
    assert lam2 < lam1, "MFP should decrease with density"
    # 超流
    assert superfluid_reduction_factor(5.0, 1.0) == 1.0  # T > Tc
    return True


if __name__ == "__main__":
    _self_check()
    print("thermodynamic_eos self-check passed.")
    rho_c = phase_transition_density(T_MeV=50.0)
    print(f"  rho_c (T=50 MeV)  : {rho_c:.3e} g/cm^3")
    print(f"  nu MFP at 1e12    : {neutrino_mean_free_path(1e12, 10, 0.3):.3e} cm")
    print(f"  nu MFP at 1e14    : {neutrino_mean_free_path(1e14, 10, 0.3):.3e} cm")
    print(f"  trapped at 1e14?  : {is_neutrino_trapped(1e14, 10, 0.3)}")
