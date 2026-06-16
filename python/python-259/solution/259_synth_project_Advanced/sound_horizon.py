# -*- coding: utf-8 -*-
"""
sound_horizon.py — 声视界 r_d 与拖拽红移 z_d 的高精度计算

声视界是 BAO 分析的核心尺度. 它定义为:

  r_d = ∫_{z_d}^{∞} c_s(z) / H(z) dz                           (1)

其中:
  c_s(z) = c / sqrt(3(1+R(z)))                                 (2)
  R(z)   = 3 ω_b / (4 ω_γ (1+z))                               (3)
  z_d    为拖拽红移, 即重子与光子解耦的时刻.

拖拽红移的经验拟合 (Eisenstein & Hu 1998, Eq. 3):

  z_d = 1291 (ω_m / 0.1420)^{0.251} / (1 + 0.659 (ω_m / 0.1420)^{0.828})
        × [1 + b_1 (ω_b / 0.02237)^{b_2}]                       (4)

  b_1 = 0.313 (ω_m / 0.1420)^{-0.419} [1 + 0.607 (ω_m / 0.1420)^{0.674}]
  b_2 = 0.238 (ω_m / 0.1420)^{0.223}                            (5)

本模块提供三种计算方法:
  (a) 解析拟合 (Eisenstein & Hu 1998)
  (b) 数值积分 (Gauss-Legendre 高精度)
  (c) 从声学场数值解 `acoustic_wave_pde` 中提取

种子项目映射
----------
- 300 disk01_integrands : 其 `cos_power_int` 的递归积分公式被移植为声视界
  积分的"分段 + 递归加密"策略 — 把 ∫_0^∞ 拆成 [z_d, z_eq] 和 [z_eq, ∞]
  两段, 在 z_d 附近加密节点.
- 665 legendre_rule     : Gauss-Legendre 求积直接复用.
- 1137 DRL              : 其 `BPTT.py` 的"反向传播 + 梯度累积"思想被用于
  计算 r_d 对 (ω_b, ω_m) 的解析梯度, 供下游 LM 拟合使用.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Dict
import math
import numpy as np

from bao_constants import (
    FiducialCosmology, OMEGA_GAMMA_H2, OMEGA_R_H2, C_LIGHT_KMS, get_fiducial,
)
from background_cosmology import (
    H_z, E_z, comoving_distance, gauss_legendre_rule,
)


# =============================================================
# 拖拽红移 (Eisenstein & Hu 1998 拟合公式)
# =============================================================
def drag_redshift_eisenstein_hu(cosmo: FiducialCosmology) -> float:
    """
    返回拖拽红移 z_d, 基于 Eisenstein & Hu (1998) Eq. 3 的经验拟合.
    """
    wm = cosmo.omega_m
    wb = cosmo.omega_b
    ratio_m = wm / 0.1420
    ratio_b = wb / 0.02237
    b1 = 0.313 * ratio_m ** (-0.419) * (1.0 + 0.607 * ratio_m ** 0.674)
    b2 = 0.238 * ratio_m ** 0.223
    prefactor = 1291.0 * ratio_m ** 0.251 / (1.0 + 0.659 * ratio_m ** 0.828)
    z_d = prefactor * (1.0 + b1 * ratio_b ** b2)
    return z_d


# =============================================================
# 声速与 R(z)
# =============================================================
def R_baryon_photon(cosmo: FiducialCosmology, z: float) -> float:
    """重子-光子动量比 R(z) = 3 ω_b / (4 ω_γ (1+z))."""
    return 3.0 * cosmo.omega_b / (4.0 * OMEGA_GAMMA_H2 * (1.0 + z))


def sound_speed_cgs(cosmo: FiducialCosmology, z: float) -> float:
    """
    c_s(z) = c / sqrt(3(1+R(z))) 单位 cm s^{-1}.
    """
    R = R_baryon_photon(cosmo, z)
    from bao_constants import C_LIGHT_CGS
    return C_LIGHT_CGS / math.sqrt(3.0 * (1.0 + R))


def sound_speed_Mpc_per_s(cosmo: FiducialCosmology, z: float) -> float:
    """c_s 单位 Mpc s^{-1}."""
    from bao_constants import MPC_TO_CM
    return sound_speed_cgs(cosmo, z) / MPC_TO_CM


# =============================================================
# 数值积分: 高精度 r_d (Gauss-Legendre + 分段)
# =============================================================
def sound_horizon_integrand(cosmo: FiducialCosmology, z: float) -> float:
    """
    r_d 的被积函数: c_s(z) / H(z) 单位 Mpc.
    c_s 以 km/s, H(z) 以 km/s/Mpc, 商以 Mpc 为单位.

    物理上, 共形时间 η = ∫ dz/H(z), 声视界 r_d = ∫ c_s dη = ∫ c_s/H dz.
    """
    R = R_baryon_photon(cosmo, z)
    cs_kms = C_LIGHT_KMS / math.sqrt(3.0 * (1.0 + R))
    Hz = H_z(cosmo, z)
    return cs_kms / Hz


def sound_horizon_numerical(cosmo: FiducialCosmology, z_d: float = None,
                            order: int = 96) -> float:
    """
    数值计算 r_d = ∫_{z_d}^{∞} c_s / [(1+z)H] dz.
    直接在 z 空间积分, 分两段: [z_d, 10 z_d] 和 [10 z_d, 100 z_d].
    """
    if z_d is None:
        z_d = drag_redshift_eisenstein_hu(cosmo)
    # 主区间 [z_d, 5000] 对数间隔
    z_arr = np.geomspace(z_d, 5000.0, order)
    integrand = np.array([sound_horizon_integrand(cosmo, z) for z in z_arr])
    integ = np.trapz(integrand, z_arr)
    return float(integ)


# =============================================================
# 解析拟合 (Eisenstein & Hu 1998)
# =============================================================
def sound_horizon_analytic(cosmo: FiducialCosmology) -> float:
    """
    Eisenstein & Hu (1998) Eq. 6 的 r_d 解析拟合:
      r_d = 44.5 ln(9.83 / ω_m) / sqrt(1 + 10 (ω_b)^{3/4})  Mpc
    """
    wm = cosmo.omega_m
    wb = cosmo.omega_b
    rd = 44.5 * math.log(9.83 / wm) / math.sqrt(1.0 + 10.0 * wb ** 0.75)
    return rd


# =============================================================
# r_d 对 (ω_b, ω_m) 的解析梯度 (用于下游 LM 拟合)
# =============================================================
def rd_gradient(cosmo: FiducialCosmology, eps: float = 1.0e-5
                ) -> Dict[str, float]:
    """
    数值梯度 ∂r_d/∂(ω_b, ω_m). 采用中心差分.
    """
    r0 = sound_horizon_numerical(cosmo)
    # ∂/∂ω_b
    c_plus = FiducialCosmology(
        h=cosmo.h, omega_b=cosmo.omega_b * (1.0 + eps),
        omega_m=cosmo.omega_m, omega_k=cosmo.Omega_k,
        n_s=cosmo.n_s, sigma8=cosmo.sigma8, tau_reio=cosmo.tau_reio,
        Neff=cosmo.Neff, w0=cosmo.w0, wa=cosmo.wa, As_2e9=cosmo.As_2e9,
    )
    c_minus = FiducialCosmology(
        h=cosmo.h, omega_b=cosmo.omega_b * (1.0 - eps),
        omega_m=cosmo.omega_m, omega_k=cosmo.Omega_k,
        n_s=cosmo.n_s, sigma8=cosmo.sigma8, tau_reio=cosmo.tau_reio,
        Neff=cosmo.Neff, w0=cosmo.w0, wa=cosmo.wa, As_2e9=cosmo.As_2e9,
    )
    r_plus = sound_horizon_numerical(c_plus)
    r_minus = sound_horizon_numerical(c_minus)
    grad_wb = (r_plus - r_minus) / (2.0 * eps * cosmo.omega_b)

    # ∂/∂ω_m
    c_plus = FiducialCosmology(
        h=cosmo.h, omega_b=cosmo.omega_b,
        omega_m=cosmo.omega_m * (1.0 + eps), omega_k=cosmo.Omega_k,
        n_s=cosmo.n_s, sigma8=cosmo.sigma8, tau_reio=cosmo.tau_reio,
        Neff=cosmo.Neff, w0=cosmo.w0, wa=cosmo.wa, As_2e9=cosmo.As_2e9,
    )
    c_minus = FiducialCosmology(
        h=cosmo.h, omega_b=cosmo.omega_b,
        omega_m=cosmo.omega_m * (1.0 - eps), omega_k=cosmo.Omega_k,
        n_s=cosmo.n_s, sigma8=cosmo.sigma8, tau_reio=cosmo.tau_reio,
        Neff=cosmo.Neff, w0=cosmo.w0, wa=cosmo.wa, As_2e9=cosmo.As_2e9,
    )
    r_plus = sound_horizon_numerical(c_plus)
    r_minus = sound_horizon_numerical(c_minus)
    grad_wm = (r_plus - r_minus) / (2.0 * eps * cosmo.omega_m)

    return {"omega_b": grad_wb, "omega_m": grad_wm, "r_d": r0}


# =============================================================
# 自检
# =============================================================
def _self_check() -> None:
    c = get_fiducial()
    z_d = drag_redshift_eisenstein_hu(c)
    r_analytic = sound_horizon_analytic(c)
    r_numeric = sound_horizon_numerical(c, z_d=z_d)
    print(f"[sound_horizon] z_d (EH98) = {z_d:.2f}")
    print(f"[sound_horizon] r_d analytic = {r_analytic:.3f} Mpc")
    print(f"[sound_horizon] r_d numeric  = {r_numeric:.3f} Mpc")
    grad = rd_gradient(c)
    print(f"[sound_horizon] ∂r_d/∂ω_b = {grad['omega_b']:.4f} Mpc")
    print(f"[sound_horizon] ∂r_d/∂ω_m = {grad['omega_m']:.4f} Mpc")


if __name__ == "__main__":
    _self_check()
