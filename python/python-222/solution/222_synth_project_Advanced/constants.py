# -*- coding: utf-8 -*-
"""
constants.py
============

计算高能物理中的基本物理常数与唯象参数。

本模块集中定义了 parton shower 与强子化模型计算所需的全部物理常量、
耦合常数、夸克质量、色因子以及模型参数，确保整个项目的数值基准统一。

物理基准:
    - 强耦合常数 alpha_s 在单圈近似下由 Lambda_QCD 跑动
    - 夸克 MSbar 质量 m_q(mu) 按重整化群演化
    - 色因子 C_F = (N_c^2 - 1) / (2 N_c),  C_A = N_c,  T_R = 1/2
    - Lund 强子化参数 a_Lund, b_Lund, sigma_string
"""

from __future__ import annotations
import math
from typing import Tuple

# ----------------------------------------------------------------------
# 基本物理常数 (PDG 2024)
# ----------------------------------------------------------------------
HBAR_C: float = 0.1973269804          # GeV * fm
HBAR_C2: float = HBAR_C * HBAR_C      # (GeV * fm)^2
C_LIGHT: float = 2.99792458e8         # m/s
ALPHA_EM: float = 1.0 / 137.035999084 # 精细结构常数 (QED)
PI: float = math.pi
EULER_GAMMA: float = 0.5772156649     # Euler-Mascheroni 常数
ZETA3: float = 1.2020569031           # Riemann zeta(3)

# ----------------------------------------------------------------------
# QCD 色因子 (SU(3))
# ----------------------------------------------------------------------
N_C: int = 3                          # 色自由度
C_F: float = (N_C ** 2 - 1) / (2.0 * N_C)   # 4/3
C_A: float = float(N_C)                       # 3
T_R: float = 0.5
N_F: int = 5                          # 活跃夸克味数 (mu ~ m_Z)

# ----------------------------------------------------------------------
# Lambda_QCD (单圈跑动耦合基准)
# ----------------------------------------------------------------------
LAMBDA_QCD_5: float = 0.213           # GeV, 5-flavor
LAMBDA_QCD_4: float = 0.294           # GeV, 4-flavor
LAMBDA_QCD_3: float = 0.340           # GeV, 3-flavor

# ----------------------------------------------------------------------
# 夸克 MSbar 质量 (GeV, mu = 2 GeV 除非特殊说明)
# ----------------------------------------------------------------------
M_U: float = 0.00216      # u 夸克
M_D: float = 0.00467      # d 夸克
M_S: float = 0.093        # s 夸克
M_C: float = 1.27         # c 夸克
M_B: float = 4.18         # b 夸克
M_T: float = 172.76       # t 夸克 (pole mass)

# 夸克味阈值 (GeV)
FLAVOUR_THRESHOLDS: Tuple[float, ...] = (
    0.0,    # 虚拟味 0
    M_U,    # 味 1
    M_D,    # 味 2
    M_S,    # 味 3
    M_C,    # 味 4
    M_B,    # 味 5
    M_T,    # 味 6
)

# ----------------------------------------------------------------------
# 强耦合常数单圈跑动
# ----------------------------------------------------------------------
def beta0_nf(nf: int) -> float:
    """单圈 beta 函数系数 beta_0 = (11 C_A - 4 T_R n_f) / (12 pi)"""
    return (11.0 * C_A - 4.0 * T_R * nf) / (12.0 * PI)


def alpha_s_one_loop(mu: float, nf: int = 5) -> float:
    """
    单圈跑动强耦合常数:
        alpha_s(mu) = 1 / [ beta_0 * ln(mu^2 / Lambda^2) ]

    其中 beta_0 = (11 C_A - 4 T_R n_f) / (12 pi)

    Parameters
    ----------
    mu : float
        重整化标度 (GeV)
    nf : int
        活跃味数

    Returns
    -------
    float
        alpha_s(mu)
    """
    if mu <= 0.0:
        raise ValueError(f"重整化标度 mu = {mu} 必须为正")
    lam = {3: LAMBDA_QCD_3, 4: LAMBDA_QCD_4, 5: LAMBDA_QCD_5}.get(nf, LAMBDA_QCD_5)
    b0 = beta0_nf(nf)
    log_ratio = math.log(mu * mu / (lam * lam))
    if log_ratio <= 0.0:
        # 进入非微扰区域，冻结耦合
        return 0.8  # 典型冻结值
    return 1.0 / (b0 * log_ratio)


def alpha_s_two_loop(mu: float, nf: int = 5) -> float:
    """
    两圈跑动强耦合常数:
        alpha_s(mu) = 1/(b0 L) - b1 ln(L) / (b0^3 L^2)
    其中 L = ln(mu^2/Lambda^2), b1 = (3 C_A - 2 n_f) / (24 pi^2) ... 简化表达
    """
    als1 = alpha_s_one_loop(mu, nf)
    if als1 >= 0.8:
        return 0.8
    lam = {3: LAMBDA_QCD_3, 4: LAMBDA_QCD_4, 5: LAMBDA_QCD_5}.get(nf, LAMBDA_QCD_5)
    b0 = beta0_nf(nf)
    b1 = (3.0 * C_A - 2.0 * nf) / (24.0 * PI * PI)
    L = math.log(mu * mu / (lam * lam))
    return als1 - b1 * math.log(L) / (b0 * b0 * L * L)


# ----------------------------------------------------------------------
# Lund 弦强子化模型参数
# ----------------------------------------------------------------------
A_LUND: float = 0.3                 # Lund 参数 a (弦碎裂偏置)
B_LUND: float = 0.8                 # Lund 参数 b (GeV^-2)
SIGMA_STRING: float = 1.0           # 弦张力 (GeV/fm) ~ 0.2 GeV^2 量级
PARALLEL_MASS: float = 0.33         # 组分夸克横向质量 (GeV)
STRING_KAPPA: float = 1.0           # 弦张力系数 (GeV^2)

# ----------------------------------------------------------------------
# Parton Shower 控制参数
# ----------------------------------------------------------------------
Q2_MIN: float = 0.5                 # 最小虚拟度截断 (GeV^2)
Q2_MAX: float = 1e6                 # 最大虚拟度 (GeV^2)
Z_CUT: float = 1e-3                 # 能量分割截断
ALPHA_S_FROZEN: float = 0.8         # 冻结耦合上限
EPSILON_REG: float = 1e-12          # 正则化小量
ALPHA_S_MAX: float = 4.0 * PI       # 微扰极限

# ----------------------------------------------------------------------
# 物理质量: 强子 (GeV) — 用于强子化输出
# ----------------------------------------------------------------------
M_PION: float = 0.13957             # pi^+-
M_RHO: float = 0.77526              # rho
M_PROTON: float = 0.93827           # p
M_KAON: float = 0.49368             # K^+-
M_ETA: float = 0.54786              # eta

# ----------------------------------------------------------------------
# 辅助
# ----------------------------------------------------------------------
def running_nf(mu: float) -> int:
    """根据标度 mu 返回活跃味数"""
    if mu < M_C:
        return 3
    elif mu < M_B:
        return 4
    elif mu < M_T:
        return 5
    else:
        return 6


def safe_alpha_s(mu: float) -> float:
    """
    数值安全的 alpha_s 获取, 冻结于非微扰区, 截断于微扰极限.
    """
    if mu <= 0.0:
        return ALPHA_S_FROZEN
    nf = running_nf(mu)
    als = alpha_s_two_loop(mu, nf)
    if not math.isfinite(als):
        return ALPHA_S_FROZEN
    return min(max(als, 0.05), ALPHA_S_FROZEN)
