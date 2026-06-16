# -*- coding: utf-8 -*-
"""
plasma_constants.py
===================

物理常数和磁流体力学 (MHD) 无量纲参数模块.

本模块定义了模拟 resistive MHD 撕裂模不稳定性 (tearing mode instability)
所需的全部物理参数, 包括:

  - 等离子体密度 rho, 电阻率 eta, 磁扩散时间 tau_R
  - Alfven 速度 v_A, 等离子体 beta 值
  - Lundquist 数 S = mu_0 L v_A / eta
  - Harris 电流片平衡参数 (B0, L_cs, j0)
  - 无量纲化常数

Harris 电流片平衡:
    B_x(y) = B0 * tanh(y / L_cs),
    j_z(y) = -(B0 / (mu_0 L_cs)) * sech^2(y / L_cs),
    p(y)   = p0 - B0^2 / (2 mu_0) * sech^2(y / L_cs).

撕裂模色散关系 (Furth-Killeen-Rosenbluth, 1963):
    gamma * tau_R ~ S^{3/5} * (k L_cs)^{2/5} * Delta'^{2/5},
其中 Delta' 为跳跃参数 (tearing stability index).
"""

from __future__ import annotations
import math


# ---------------------------------------------------------------
#  SI 基本常数
# ---------------------------------------------------------------
MU_0 = 4.0e-7 * math.pi       # 真空磁导率 [H/m]
EPS_0 = 8.854187817e-12       # 真空介电常数 [F/m]
C_LIGHT = 2.99792458e8        # 光速 [m/s]
K_B = 1.380649e-23            # 玻尔兹曼常数 [J/K]
Q_E = 1.602176634e-19         # 电子电荷 [C]
M_P = 1.67262192e-27          # 质子质量 [kg]
M_E = 9.1093837e-31           # 电子质量 [kg]


# ---------------------------------------------------------------
#  典型托卡马克等离子体参数 (简化模型)
# ---------------------------------------------------------------
#  特征长度 L (电流片半宽) [m]
L_CS = 1.0e-2

#  平衡磁场强度 B0 [T]
B0 = 0.1

#  等离子体数密度 n0 [m^-3]
N0 = 1.0e19

#  离子质量 (氘)
M_ION = 2.0 * M_P

#  等离子体质量密度 rho = n0 * M_ION [kg/m^3]
RHO0 = N0 * M_ION

#  Alfven 速度 v_A = B0 / sqrt(mu_0 * rho0) [m/s]
V_ALFVEN = B0 / math.sqrt(MU_0 * RHO0)

#  等离子体 beta = 2 mu_0 p / B0^2
#  这里假设 p = 0.5 * n0 * k_B * T, T = 1 keV
T_EV = 1.0e3                         # 温度 [eV]
T_J = T_EV * Q_E                     # 温度 [J]
P0 = N0 * T_J                        # 压强 [Pa]
BETA_PLASMA = 2.0 * MU_0 * P0 / (B0 * B0)

#  特征 Alfven 时间 tau_A = L / v_A [s]
TAU_ALFVEN = L_CS / V_ALFVEN


# ---------------------------------------------------------------
#  电阻与 Lundquist 数
# ---------------------------------------------------------------
#  Spitzer 电阻率 (简化) eta [Ohm*m]
ETA_SPITZER = 1.0e-5

#  磁扩散时间 tau_R = mu_0 L^2 / eta [s]
TAU_R = MU_0 * L_CS * L_CS / ETA_SPITZER

#  Lundquist 数 S = tau_R / tau_A = mu_0 L v_A / eta
LUNDQUIST_S = TAU_R / TAU_ALFVEN


# ---------------------------------------------------------------
#  撕裂模线性理论参数
# ---------------------------------------------------------------
#  特征波数 k (归一化: k * L_cs)
K_PARALLEL_NORM = 0.5

#  绝对波数 k [1/m]
K_PARALLEL = K_PARALLEL_NORM / L_CS

#  Harris 片峰值电流密度 j0 = B0 / (mu_0 L_cs) [A/m^2]
J0 = B0 / (MU_0 * L_CS)

#  离子惯性长度 d_i = c / omega_pi [m]
OMEGA_PI = math.sqrt(N0 * Q_E * Q_E / (EPS_0 * M_ION))
D_ION = C_LIGHT / OMEGA_PI

#  离子声速 c_s = sqrt(k_B T / M_ION) [m/s]
C_SOUND = math.sqrt(T_J / M_ION)

#  离子拉莫尔半径 rho_s = c_s / omega_ci [m]
OMEGA_CI = Q_E * B0 / M_ION
RHO_S = C_SOUND / OMEGA_CI


# ---------------------------------------------------------------
#  归一化参数包
# ---------------------------------------------------------------
def get_dimensionless_params() -> dict:
    """返回无量纲参数集合, 供稳定性分析使用."""
    return {
        "S": LUNDQUIST_S,
        "beta": BETA_PLASMA,
        "kL": K_PARALLEL_NORM,
        "d_i_over_L": D_ION / L_CS,
        "rho_s_over_L": RHO_S / L_CS,
        "c_s_over_vA": C_SOUND / V_ALFVEN,
    }


# ---------------------------------------------------------------
#  撕裂模 FKR 标度律 (Furth-Killeen-Rosenbluth, 1963)
# ---------------------------------------------------------------
def fkr_growth_rate(delta_prime: float) -> float:
    """
    经典 FKR 撕裂模增长率标度:
        gamma * tau_R ~ 0.6 * (kL)^{2/5} * S^{3/5} * Delta'^{4/5}

    参数:
        delta_prime: 无量纲跳跃参数 Delta' * L (符号约定: 不稳定为正)

    返回:
        gamma * tau_R 的无量纲增长率
    """
    S = LUNDQUIST_S
    kL = K_PARALLEL_NORM
    coeff = 0.6
    if delta_prime <= 0.0:
        return 0.0
    return coeff * (kL ** (2.0 / 3.0)) * (S ** 0.6) * (delta_prime ** 0.8)


# ---------------------------------------------------------------
#  Copling parameter (Copson 参数, 衡量 resistive-inertial 区间)
# ---------------------------------------------------------------
def copling_parameter() -> float:
    """
    耦合参数 Lambda = (gamma tau_R)^{1/2} * (kL)^{-1/2} * S^{-1/2}
    用于判断撕裂模处于常-psi 还是非常-psi 区域.
    """
    return math.sqrt(LUNDQUIST_S * K_PARALLEL_NORM)


# ---------------------------------------------------------------
#  打印参数汇总
# ---------------------------------------------------------------
def summary() -> str:
    p = get_dimensionless_params()
    lines = [
        "=" * 60,
        "  等离子体参数汇总 (Harris 电流片平衡)",
        "=" * 60,
        f"  B0            = {B0:.3e}  T",
        f"  L_cs          = {L_CS:.3e}  m",
        f"  n0            = {N0:.3e}  m^-3",
        f"  rho0          = {RHO0:.3e}  kg/m^3",
        f"  v_Alfven      = {V_ALFVEN:.3e}  m/s",
        f"  tau_Alfven    = {TAU_ALFVEN:.3e}  s",
        f"  beta          = {BETA_PLASMA:.3e}",
        f"  eta_Spitzer   = {ETA_SPITZER:.3e}  Ohm*m",
        f"  tau_R         = {TAU_R:.3e}  s",
        f"  Lundquist S   = {LUNDQUIST_S:.3e}",
        f"  j0            = {J0:.3e}  A/m^2",
        f"  kL            = {p['kL']:.3e}",
        f"  d_i/L         = {p['d_i_over_L']:.3e}",
        f"  rho_s/L       = {p['rho_s_over_L']:.3e}",
        f"  c_s/v_A       = {p['c_s_over_vA']:.3e}",
        "=" * 60,
    ]
    return "\n".join(lines)
