# -*- coding: utf-8 -*-
"""
bao_constants.py — 物理常数与基准宇宙学先验

本模块封装 BAO 分析中用到的全部物理常数、无量纲参数与基准宇宙学
(Planck 2018 TT,TE,EE+lowE+lensing 基准). 所有数值均以 CGS / Mpc / km s^{-1}
混合单位给出, 并显式提供单位换算因子, 防止下游模块出现量纲错误.

主要映射的种子项目
-----------------
- 1137 DRL-for-Personalized-Energy-Trading : Parameters_final_David.py 中
  "全局参数 + 先验字典" 的工程组织方式, 在此被移植为 `FiducialCosmology`
  数据类, 把 (Ω_m, Ω_b, h, n_s, σ_8) 与它们的 1-σ 先验一并收纳.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Tuple
import math

# =============================================================
# 基本物理常数 (CODATA 2018)
# =============================================================
C_LIGHT_CGS        : float = 2.99792458e10      # cm s^{-1}
C_LIGHT_KMS        : float = 2.99792458e5       # km s^{-1}
H_PLANCK_CGS       : float = 6.62607015e-27     # erg s
KBOLTZ_CGS         : float = 1.380649e-16        # erg K^{-1}
G_NEWTON_CGS       : float = 6.67430e-8          # cm^3 g^{-1} s^{-2}
SIGMA_THOMSON      : float = 6.6524587158e-25    # cm^2
M_PROTON_CGS       : float = 1.67262192e-24      # g
M_ELECTRON_CGS     : float = 9.1093837e-28       # g
EV_TO_ERG          : float = 1.602176634e-12     # erg / eV

# =============================================================
# 天文单位换算
# =============================================================
PC_TO_CM           : float = 3.0856775814913673e18
MPC_TO_CM          : float = 1000.0 * PC_TO_CM
MPC_TO_KM          : float = MPC_TO_CM / 1.0e5
YR_TO_S            : float = 3.15576e7
GYR_TO_S           : float = 1.0e9 * YR_TO_S

# =============================================================
# CMB / 复合时期关键温度
# =============================================================
T_CMB_0_K          : float = 2.7255              # Fixsen 2009
OMEGA_GAMMA_H2     : float = 2.4693788e-5        # γ 密度参数 (T_CMB=2.7255)
OMEGA_NU_REL_H2    : float = (
    OMEGA_GAMMA_H2 * 3.046 * (7.0 / 8.0) * (4.0 / 11.0) ** (4.0 / 3.0)
)                                                 # 有效相对论中微子 (Neff=3.046)
OMEGA_R_H2         : float = OMEGA_GAMMA_H2 + OMEGA_NU_REL_H2


@dataclass
class FiducialCosmology:
    """基准 Planck-2018 ΛCDM 参数, 作为 BAO 拟合的起点."""
    h           : float = 0.6766
    omega_b     : float = 0.02237       # Ω_b h^2
    omega_m     : float = 0.1420        # Ω_m h^2
    omega_k     : float = 0.0
    n_s         : float = 0.9649
    sigma8      : float = 0.8111
    tau_reio    : float = 0.0544
    Neff        : float = 3.046
    w0          : float = -1.0
    wa          : float = 0.0
    As_2e9      : float = 2.097         # 10^9 A_s

    # ---------- 派生量 ----------
    @property
    def H0_km_s_Mpc(self) -> float:
        return 100.0 * self.h

    @property
    def H0_inv_Mpc(self) -> float:
        """H0^{-1} 以 Mpc 为单位 (c=1 约定下即 DH0)."""
        return C_LIGHT_KMS / self.H0_km_s_Mpc

    @property
    def Omega_b(self) -> float:
        return self.omega_b / (self.h * self.h)

    @property
    def Omega_m(self) -> float:
        return self.omega_m / (self.h * self.h)

    @property
    def Omega_r(self) -> float:
        return OMEGA_R_H2 / (self.h * self.h)

    @property
    def Omega_k(self) -> float:
        return self.omega_k

    @property
    def Omega_Lambda(self) -> float:
        return 1.0 - self.Omega_m - self.Omega_k - self.Omega_r

    @property
    def R_eq(self) -> float:
        """物质-辐射相等时刻的重子-光子动量比 R(z_eq)."""
        z_eq = OMEGA_R_H2 / self.omega_m - 1.0
        return 3.0 * self.omega_b / (4.0 * self.omega_m)

    def summary(self) -> Dict[str, float]:
        return {
            "h"             : self.h,
            "Omega_m"       : self.Omega_m,
            "Omega_b"       : self.Omega_b,
            "Omega_Lambda"  : self.Omega_Lambda,
            "Omega_r"       : self.Omega_r,
            "DH0_Mpc"       : self.H0_inv_Mpc,
        }


@dataclass
class ParameterPriors:
    """BAO + 大尺度结构联合分析的一维高斯先验 (均值, σ)."""
    omega_b     : Tuple[float, float] = (0.02237, 0.00015)
    omega_m     : Tuple[float, float] = (0.1420,  0.0010)
    h           : Tuple[float, float] = (0.6766,  0.0042)
    n_s         : Tuple[float, float] = (0.9649,  0.0042)
    sigma8      : Tuple[float, float] = (0.8111,  0.0060)
    w0          : Tuple[float, float] = (-1.0,    0.05)
    wa          : Tuple[float, float] = (0.0,     0.20)

    def log_prior(self, theta: Dict[str, float]) -> float:
        """返回对数先验 p(theta) 的对数值 (非归一化)."""
        lp = 0.0
        for name, (mu, sig) in (
            ("omega_b", self.omega_b), ("omega_m", self.omega_m),
            ("h", self.h), ("n_s", self.n_s), ("sigma8", self.sigma8),
            ("w0", self.w0), ("wa", self.wa),
        ):
            if name not in theta:
                continue
            z = (theta[name] - mu) / sig
            lp += -0.5 * z * z
        # 物理边界
        if theta.get("omega_b", 0.0) < 0.018 or theta.get("omega_b", 0.0) > 0.026:
            return -math.inf
        if theta.get("omega_m", 0.0) < 0.10 or theta.get("omega_m", 0.0) > 0.20:
            return -math.inf
        if theta.get("h", 0.0) < 0.55 or theta.get("h", 0.0) > 0.85:
            return -math.inf
        return lp


# =============================================================
# 基准红shift 采样 (BOSS DR12 + eBOSS + DESI-like 壳层)
# =============================================================
DEFAULT_BAO_SHELLS: Tuple[Tuple[float, float, str], ...] = (
    (0.38, 0.10, "CMASS-Low"),
    (0.51, 0.10, "CMASS-High"),
    (0.85, 0.15, "eBOSS-LRG"),
    (1.48, 0.20, "eBOSS-QSO"),
)


def get_fiducial() -> FiducialCosmology:
    return FiducialCosmology()


def get_priors() -> ParameterPriors:
    return ParameterPriors()
