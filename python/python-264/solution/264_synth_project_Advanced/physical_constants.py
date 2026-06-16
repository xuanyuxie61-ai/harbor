# -*- coding: utf-8 -*-
"""
physical_constants.py
=====================

物理常数与磁层基本参数模块.

本模块集中存储所有与地球磁层粒子输运相关的物理常数, 包括:
  - 国际单位制 (SI) 基本常数
  - 地球偶极磁场参数
  - 辐射带典型粒子参数
  - 无量纲化参考尺度

这些参数来源于:
  [1] Kivelson & Russell, "Introduction to Space Physics", Cambridge (1995)
  [2] Roederer & Tian, "Dynamics of Magnetically Trapped Particles", Springer (2016)
  [3] Schulz & Lanzerotti, "Particle Diffusion in the Radiation Belts", Springer (1974)

核心物理量:
  地球半径:       R_E = 6.371e6 m
  地球偶极矩:     M_E = 7.94e15 T*m^3 (或 M_E = 8.0e22 A*m^2)
  电子静止质量:   m_e = 9.109e-31 kg
  电子电荷:       q_e = 1.602e-19 C
  光速:           c = 2.998e8 m/s
  真空磁导率:     mu_0 = 4*pi*1e-7 H/m

磁层特征尺度:
  等离子体层顶:     L_pp ~ 3 ~ 5 R_E
  同步轨道:          L_sync = 6.6 R_E
  磁顶日下点:        L_mp ~ 10 R_E (太阳风动压依赖)

偶极磁场赤道强度:
  B_0 = mu_0 * M_E / (4*pi*R_E^3) ~ 3.07e-5 T = 30700 nT
"""

import numpy as np


# =============================================================================
#  基本物理常数 (SI 单位制, CODATA 2018 推荐值)
# =============================================================================

# 光速 [m/s]
C_LIGHT = 2.99792458e8

# 真空介电常数 [F/m]
EPSILON_0 = 8.8541878128e-12

# 真空磁导率 [H/m]
MU_0 = 4.0e-7 * np.pi  # 精确值: 1.25663706212e-6

# 电子静止质量 [kg]
M_ELECTRON = 9.1093837015e-31

# 质子静止质量 [kg]
M_PROTON = 1.67262192369e-27

# 电子电荷绝对值 [C]
Q_E = 1.602176634e-19

# 玻尔兹曼常数 [J/K]
K_BOLTZMANN = 1.380649e-23

# 普朗克常数 [J*s]
H_PLANCK = 6.62607015e-34

# 阿伏伽德罗常数 [1/mol]
N_AVOGADRO = 6.02214076e23


# =============================================================================
#  地球参数
# =============================================================================

# 地球平均半径 [m]
R_EARTH = 6.371e6

# 地球自转角速度 [rad/s]
OMEGA_EARTH = 7.2921150e-5

# 地球偶极矩 [A*m^2] (中心偶极近似, IGRF-13)
DIPOLE_MOMENT = 8.0e22

# 偶极倾角 [rad] (~11.5 度)
DIPOLE_TILT = 0.2007

# 地球表面赤道磁场 [T]
# B_eq = mu_0 * M / (4*pi*R_E^3)
B_EQUATORIAL = MU_0 * DIPOLE_MOMENT / (4.0 * np.pi * R_EARTH**3)


# =============================================================================
#  磁层特征参数
# =============================================================================

# McIlwain L 参数典型范围 (无量纲, 以 R_E 为单位)
L_MIN = 1.5    # 内边界: 低 L 壳层, 稳定捕获区
L_MAX = 7.5    # 外边界: 近磁顶区域, 强损失区
L_SHELL_NUM = 61  # L 壳层网格数

# 等离子体层顶 L 值 (日侧/夜侧平均值)
L_PLASMAPAUSE = 4.0

# 同步轨道 L 值
L_GEOSTATIONARY = 6.617

# 磁顶日下点 L 值 (典型值; 实际取决于太阳风动压)
L_MAGNETOPAUSE = 10.0

# 特征磁场强度 [T] (L = 4 赤道面)
B_REFERENCE = B_EQUATORIAL / (L_PLASMAPAUSE**3)

# 特征电子回旋频率 [rad/s] (L = 4)
# Omega_ce = q_e * B / (m_e * c)  [相对论修正前]
OMEGA_CE_REF = Q_E * B_REFERENCE / M_ELECTRON

# 特征电子回旋周期 [s]
TAU_CE_REF = 2.0 * np.pi / OMEGA_CE_REF


# =============================================================================
#  辐射带电子典型能量范围
# =============================================================================

# 动能范围 [MeV]
ENERGY_MIN_MeV = 0.1    # 低能种子种群
ENERGY_MAX_MeV = 8.0    # 超相对论电子 (ULO)
ENERGY_NUM = 41         # 能量网格数

# 对应的静止能量 [MeV]: m_e * c^2 = 0.511 MeV
E_REST_MeV = M_ELECTRON * C_LIGHT**2 / Q_E * 1.0e-6

# 对应的洛伦兹因子范围
GAMMA_MIN = 1.0 + ENERGY_MIN_MeV / E_REST_MeV
GAMMA_MAX = 1.0 + ENERGY_MAX_MeV / E_REST_MeV

# 对应的动量范围 [kg*m/s]
# p = m_e*c*sqrt(gamma^2 - 1)
P_MIN = M_ELECTRON * C_LIGHT * np.sqrt(GAMMA_MIN**2 - 1.0)
P_MAX = M_ELECTRON * C_LIGHT * np.sqrt(GAMMA_MAX**2 - 1.0)


# =============================================================================
#  投掷角参数
# =============================================================================

# 赤道投掷角 [rad]
ALPHA_EQ_MIN = 1.0e-3 * np.pi     # 接近损失锥
ALPHA_EQ_MAX = 0.5 * np.pi        # 90 度 (横向)

# 损失锥投掷角 (L = 4, 偶极场近似):
# sin^2(alpha_LC) = B_eq / B_atm
# B_atm ~ 5e-5 T (100 km 高度)
B_ATMOSPHERE = 5.0e-5
SIN2_ALPHA_LC_REF = B_EQUATORIAL / (L_PLASMAPAUSE**3) / B_ATMOSPHERE
ALPHA_LC_REF = np.arcsin(np.sqrt(np.clip(SIN2_ALPHA_LC_REF, 0.0, 1.0)))


# =============================================================================
#  波-粒子相互作用参数 (用于准线性扩散系数)
# =============================================================================

# 典型chorus波频率范围 [Hz]
F_CHORUS_MIN = 0.1 * OMEGA_CE_REF / (2.0 * np.pi) * 0.05
F_CHORUS_MAX = 0.1 * OMEGA_CE_REF / (2.0 * np.pi) * 0.5

# 典型chorus波振幅 [pT] (活跃期)
B_CHORUS_AMPLITUDE = 100.0e-12

# 等离子体密度 [cm^-3] (等离子体层外典型值)
N_PLASMA_REF = 10.0e6  # 10 /cm^3 -> 10^7 /m^3

# 等离子体频率 [rad/s]
# omega_pe = sqrt(n_e * e^2 / (epsilon_0 * m_e))
OMEGA_PE_REF = np.sqrt(N_PLASMA_REF * Q_E**2 / (EPSILON_0 * M_ELECTRON))


# =============================================================================
#  径向扩散系数参考值 (L = 4, 活跃期)
# =============================================================================

# D_LL ~ 10^{-10} L^{10} s^{-1} (经验公式, Brautigam & Albert 2000)
# 在 L = 4: D_LL ~ 10^{-10} * 4^10 ~ 1.05e-4 s^{-1}  (无量纲形式需要转换)
D_LL_REFERENCE = 1.0e-10 * (L_PLASMAPAUSE**10)

# 特征扩散时间尺度 [s]
# tau_diff = L^2 / D_LL
TAU_DIFFUSION = L_PLASMAPAUSE**2 / D_LL_REFERENCE


# =============================================================================
#  无量纲化参数
# =============================================================================

# 时间归一化: 用 tau_diff 归一化
TIME_NORM = TAU_DIFFUSION

# 空间归一化: 用 R_E 归一化 (L 参数本身已无量纲)
LENGTH_NORM = R_EARTH

# 能量归一化: 用 m_e * c^2 归一化
ENERGY_NORM = M_ELECTRON * C_LIGHT**2

# 速度归一化: c
VELOCITY_NORM = C_LIGHT

# 磁场归一化: B_eq
B_NORM = B_EQUATORIAL


# =============================================================================
#  数值参数
# =============================================================================

# 小量 (防止除零)
EPSILON_NUM = 1.0e-30

# 最大洛伦兹因子 (防止溢出)
GAMMA_CEILING = 1.0e4

# 默认 CFL 数
CFL_DEFAULT = 0.45

# WENO 重构 epsilon (防止分母为零)
WENO_EPSILON = 1.0e-6

# 多项式混沌阶数
PCE_ORDER = 4

# 随机样本数
MONTE_CARLO_SAMPLES = 512


# =============================================================================
#  工具函数
# =============================================================================

def kinetic_to_lorentz(E_kin_MeV):
    """
    将动能 (MeV) 转换为洛伦兹因子 gamma.

    物理公式:
      gamma = 1 + E_kin / (m_e * c^2)

    参数
    ----
    E_kin_MeV : float 或 ndarray
        动能, 单位 MeV

    返回
    ----
    gamma : float 或 ndarray
        洛伦兹因子 (无量纲)
    """
    return 1.0 + np.asarray(E_kin_MeV) / E_REST_MeV


def lorentz_to_kinetic(gamma):
    """
    将洛伦兹因子 gamma 转换为动能 (MeV).

    物理公式:
      E_kin = (gamma - 1) * m_e * c^2

    参数
    ----
    gamma : float 或 ndarray
        洛伦兹因子 (无量纲)

    返回
    ----
    E_kin_MeV : float 或 ndarray
        动能, 单位 MeV
    """
    return (np.asarray(gamma) - 1.0) * E_REST_MeV


def momentum_from_gamma(gamma):
    """
    从洛伦兹因子计算相对论动量.

    物理公式:
      p = m_e * c * sqrt(gamma^2 - 1)

    参数
    ----
    gamma : float 或 ndarray
        洛伦兹因子

    返回
    ----
    p : float 或 ndarray
        动量 [kg*m/s]
    """
    gamma = np.asarray(gamma)
    gamma = np.clip(gamma, 1.0, GAMMA_CEILING)
    return M_ELECTRON * C_LIGHT * np.sqrt(gamma**2 - 1.0)


def dipole_field_magnitude(L, theta=np.pi / 2.0):
    """
    偶极磁场模长 (偶极近似).

    物理公式 (偶极场):
      B(L, theta) = (M_E / r^3) * sqrt(1 + 3*cos^2(theta))
                  = B_0 / L^3 * sqrt(1 + 3*cos^2(theta))

    其中 theta 为余纬 (磁余纬, 赤道 theta = pi/2).
    在赤道面上 (theta = pi/2): B_eq = B_0 / L^3

    参数
    ----
    L : float 或 ndarray
        McIlwain L 参数
    theta : float 或 ndarray
        磁余纬 (rad), 默认赤道面

    返回
    ----
    B : float 或 ndarray
        磁场强度 [T]
    """
    L = np.asarray(L, dtype=np.float64)
    L = np.maximum(L, 1.0 + EPSILON_NUM)  # 防止 L=0 奇点
    B_eq = B_EQUATORIAL / (L**3)
    angular_factor = np.sqrt(1.0 + 3.0 * np.cos(theta)**2)
    return B_eq * angular_factor


def gyrofrequency(B):
    """
    相对论电子回旋频率.

    物理公式:
      Omega_ce = q_e * B / (gamma * m_e)

    参数
    ----
    B : float 或 ndarray
        磁场强度 [T]

    返回
    ----
    Omega_ce : float 或 ndarray
        回旋频率 [rad/s]
    """
    return Q_E * np.abs(np.asarray(B)) / M_ELECTRON


def first_adiabatic_moment(gamma, B, alpha_eq):
    """
    第一绝热不变量 mu (磁矩).

    物理公式:
      mu = p_perp^2 / (2 * m_e * B_mirror)
         = gamma * m_e * c^2 * sin^2(alpha) / (2 * B)

    对于赤道投掷角 alpha_eq, 镜点磁场:
      B_m = B_eq / sin^2(alpha_eq)

    参数
    ----
    gamma : float
        洛伦兹因子
    B : float
        磁场强度 [T]
    alpha_eq : float
        赤道投掷角 [rad]

    返回
    ----
    mu : float
        第一绝热不变量 [J/T]
    """
    p = momentum_from_gamma(gamma)
    p_perp = p * np.sin(alpha_eq)
    return p_perp**2 / (2.0 * M_ELECTRON * max(B, EPSILON_NUM))


if __name__ == "__main__":
    # 简单自检
    print(f"地球赤道磁场: B_eq = {B_EQUATORIAL:.4e} T")
    print(f"电子静止能量: {E_REST_MeV:.4f} MeV")
    print(f"L=4 赤道磁场: {dipole_field_magnitude(4.0):.4e} T")
    print(f"L=4 回旋频率: {gyrofrequency(dipole_field_magnitude(4.0)):.4e} rad/s")
    print(f"L=4 参考扩散系数: D_LL = {D_LL_REFERENCE:.4e}")
    print(f"参考扩散时间: tau_diff = {TAU_DIFFUSION:.4e} s ~ {TAU_DIFFUSION/86400:.1f} days")
