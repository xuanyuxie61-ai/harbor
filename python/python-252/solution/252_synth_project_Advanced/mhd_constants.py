#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mhd_constants.py  ——  物理常数、黑洞参数、无量纲化单位

本项目核心科学问题: Kerr 黑洞磁层中 Blandford–Znajek 喷流的高阶有限差分
MHD 模拟及其线性稳定性分析 (博士级可复现小规模实验).

融合种子项目:
  - 1057_Bio-Inspired-Navigation : Config 风格的全局参数管理
  - 1374_unstable_ode           : unstable_parameters 风格的可切换参数集
  - 1008_C15StabilityDataW      : Arrhenius 风格的物理常数分组
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Dict


# ============================================================
# 基础物理常数 (CGS)
# ============================================================
C_LIGHT     = 2.99792458e10          # cm / s
G_NEWTON    = 6.67430e-8             # cm^3 g^-1 s^-2
M_SUN       = 1.98892e33             # g
K_BOLTZ     = 1.380649e-16           # erg K^-1
M_PROTON    = 1.67262192e-24         # g
SIGMA_T     = 6.6524587158e-25       # cm^2  Thomson
E_CHARGE    = 4.80320451e-10         # esu
H_PLANCK    = 6.62607015e-27         # erg s
MU_0_SI     = 1.25663706212e-6       # H m^-1
EPS_0_SI    = 8.8541878128e-12       # F m^-1


# ============================================================
# 典型 AGN 黑洞参数  (默认: M = 1e8 M_sun)
# ============================================================
@dataclass
class BlackHoleParameters:
    """Kerr 黑洞参数, 几何单位制 G = c = 1, 长度单位 = M."""
    M_BH_Msun:   float = 1.0e8           # 黑洞质量 (太阳质量)
    a_star:      float = 0.9375          # 无量纲自旋 a* = J/M^2, |a*| < 1
    M_bh_g:      float = field(init=False)   # 质量 (克)
    M_geom_cm:   float = field(init=False)   # 几何化长度 GM/c^2 (cm)
    r_plus:      float = field(init=False)   # 外事件视界 r_+ = M + sqrt(M^2 - a^2)
    r_minus:     float = field(init=False)   # 内事件视界 r_- = M - sqrt(M^2 - a^2)
    r_ergo_eq:   float = field(init=False)   # 赤道能层半径
    omega_h:     float = field(init=False)   # 视界角速度  a / (2 M r_+)
    a_geom:      float = field(init=False)   # 几何化自旋参数 a = a* M

    def __post_init__(self) -> None:
        if not 0.0 <= self.a_star < 1.0:
            raise ValueError(f"a_star 必须在 [0, 1), 给定 {self.a_star}")
        self.M_bh_g    = self.M_BH_Msun * M_SUN
        self.M_geom_cm = G_NEWTON * self.M_bh_g / (C_LIGHT * C_LIGHT)
        # 几何单位制: M = 1
        self.a_geom    = self.a_star
        disc = math.sqrt(max(1.0 - self.a_star * self.a_star, 0.0))
        self.r_plus    = 1.0 + disc
        self.r_minus   = 1.0 - disc
        # 赤道能层  r_ergo(theta=pi/2) = M + sqrt(M^2 - a^2 cos^2 theta) = 2 M
        self.r_ergo_eq = 2.0
        self.omega_h   = self.a_star / (2.0 * self.r_plus)


# ============================================================
# 吸积盘/喷流等离子体参数
# ============================================================
@dataclass
class PlasmaParameters:
    """磁化等离子体参数 (几何单位制)."""
    gamma_ad:       float = 5.0 / 3.0      # 绝热指数 (单原子非相对论)
    rho_floor:      float = 1.0e-4         # 密度地板 (防止真空)
    p_floor:        float = 1.0e-6         # 压强地板
    eta_resist:     float = 1.0e-4         # 电阻率 (电阻 MHD)
    nu_visc:        float = 0.0            # 动力学粘性 (默认理想 MHD)
    beta_init:      float = 0.1            # 初始等离子体 beta = p / (B^2/2)
    mach_pol:       float = 0.5            # 极向马赫数
    sigma_hotspot:  float = 30.0           # 磁化度 sigma = B^2/(rho h)

    def sound_speed(self, p, rho):
        """相对论声速  cs^2 = gamma p / (h), h = rho + gamma p/(gamma-1)."""
        import numpy as _np
        h = rho + self.gamma_ad * p / (self.gamma_ad - 1.0)
        cs2 = self.gamma_ad * p / _np.maximum(h, 1.0e-30)
        cs2 = _np.maximum(_np.minimum(cs2, 1.0 / 3.0), 0.0) if hasattr(cs2, '__len__') else max(min(cs2, 1.0/3.0), 0.0)
        return _np.sqrt(cs2) if hasattr(cs2, '__len__') else math.sqrt(cs2)

    def alfven_speed(self, b2, rho_h):
        """相对论 Alfvén 速度  vA^2 = b^2 / (b^2 + rho h)."""
        import numpy as _np
        va2 = b2 / _np.maximum(b2 + rho_h, 1e-30)
        return _np.sqrt(va2) if hasattr(va2, '__len__') else math.sqrt(va2)

    def fast_magnetosonic(self, p, rho, b2):
        """相对论快磁声速.  cs^2 + vA^2 - cs^2 vA^2 (近似)."""
        import numpy as _np
        cs2 = self.gamma_ad * p / _np.maximum(rho + self.gamma_ad * p / (self.gamma_ad - 1.0), 1e-30)
        va2 = b2 / _np.maximum(b2 + rho, 1e-30)
        cf2 = cs2 + va2 - cs2 * va2
        cf2 = _np.maximum(_np.minimum(cf2, 1.0), 0.0) if hasattr(cf2, '__len__') else max(min(cf2, 1.0), 0.0)
        return _np.sqrt(cf2) if hasattr(cf2, '__len__') else math.sqrt(cf2)


# ============================================================
# 数值参数
# ============================================================
@dataclass
class NumericalParameters:
    """有限差分 / 时间积分 / 网格参数."""
    # 空间离散
    nr:             int   = 64            # 径向格点数
    ntheta:         int   = 32            # 极角格点数
    nphi:           int   = 1             # 方位角 (轴对称=1)
    r_in:           float = 1.2           # 内半径 (以 M 为单位, > r_+)
    r_out:          float = 50.0          # 外半径
    fd_order:       int   = 6             # 有限差分阶数 (2/4/6/8)

    # 时间积分
    cfl:            float = 0.3           # CFL 数
    t_end:          float = 100.0         # 终止时间 (M 单位)
    integrator:     str   = "rk3-ssp"     # euler / rk2-heun / rk3-ssp / rk4

    # 隐式径向求解
    implicit_radial: bool  = True
    newton_tol:      float = 1.0e-10
    newton_maxiter:  int   = 30

    # 滤波
    dct_filter_strength: float = 0.01     # DCT 高频滤波强度

    # 稳定性分析
    n_modes_kept:  int   = 12             # 保留的最不稳定模态数
    omega_scan_pts: int  = 200            # 频率扫描点数


# ============================================================
# 聚合配置 (参考 1057 的 Config 模式)
# ============================================================
@dataclass
class MHDConfig:
    bh:     BlackHoleParameters = field(default_factory=BlackHoleParameters)
    plasma: PlasmaParameters    = field(default_factory=PlasmaParameters)
    num:    NumericalParameters = field(default_factory=NumericalParameters)

    def validate(self) -> Dict[str, bool]:
        """检查参数合法性 (边界鲁棒性)."""
        checks = {
            "a_star_in_range":    0.0 <= self.bh.a_star < 1.0,
            "r_in_outside_horizon": self.num.r_in > self.bh.r_plus,
            "r_out_beyond_r_in":  self.num.r_out > self.num.r_in,
            "fd_order_even":      self.num.fd_order in (2, 4, 6, 8),
            "cfl_positive":       self.num.cfl > 0.0,
            "gamma_positive":     self.plasma.gamma_ad > 1.0,
            "rho_floor_positive": self.plasma.rho_floor > 0.0,
            "nr_positive":        self.num.nr >= 8,
            "ntheta_positive":    self.num.ntheta >= 4,
        }
        for k, v in checks.items():
            if not v:
                raise ValueError(f"MHDConfig 校验失败: {k}")
        return checks


# ============================================================
# Blandford–Znajek 功率估算 (用于初始猜测)
# ============================================================
def blandford_znajek_power(cfg: MHDConfig, B_polar_cgs: float = 1.0e4) -> float:
    """
    BZ 喷流功率 (近似):
        P_BZ ≈ (1/6 c) Omega_H^2 Phi_BH^2
    其中  Phi_BH = 2 pi r_+^2 B_p  (磁通量)
    返回 erg/s.
    """
    bh = cfg.bh
    r_plus_cm = bh.r_plus * bh.M_geom_cm
    Phi = 2.0 * math.pi * r_plus_cm * r_plus_cm * B_polar_cgs
    Omega_H_cgs = bh.omega_h * C_LIGHT / bh.M_geom_cm
    P_BZ = (1.0 / (6.0 * C_LIGHT)) * Omega_H_cgs * Omega_H_cgs * Phi * Phi
    return P_BZ
