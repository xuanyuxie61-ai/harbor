"""
parameters.py
全局托卡马克物理参数与物理常量定义。
所有参数以国际单位制(SI)为基准，数值参考ITER-like托卡马克。
"""

import numpy as np

# ============================
# 基本物理常量
# ============================
MU0 = 4.0 * np.pi * 1e-7          # 真空磁导率 [H/m]
EPS0 = 8.854187817e-12            # 真空介电常数 [F/m]
KB = 1.380649e-23                 # 玻尔兹曼常数 [J/K]
QE = 1.602176634e-19              # 元电荷 [C]
ME = 9.10938356e-31               # 电子质量 [kg]
MP = 1.6726219e-27                # 质子质量 [kg]
MD = 3.343583719e-27              # 氘核质量 [kg]
MT = 5.006412e-27                 # 氚核质量 [kg]
HE3_MASS = 5.006412e-27 * 1.5     # He3近似质量 [kg]
C_LIGHT = 2.99792458e8            # 光速 [m/s]

# ============================
# ITER-like 托卡马克几何参数
# ============================
R0 = 6.2          # 大半径 [m]
a_minor = 2.0     # 小半径 [m]
B0 = 5.3          # 环向磁场 on axis [T]
KAPPA = 1.7       # 拉长比 (elongation)
DELTA = 0.33      # 三角形变 (triangularity)
q0 = 1.0          # 磁轴安全因子
q_edge = 3.0      # 边界安全因子

# ============================
# 等离子体参数
# ============================
N_E_AXIS = 1.0e20       # 电子密度轴心值 [m^-3]
N_E_PED = 0.4e20        # 电子密度台基值 [m^-3]
T_E_AXIS = 20.0e3       # 电子温度轴心值 [eV]
T_E_PED = 3.0e3         # 电子温度台基值 [eV]
T_I_AXIS = 20.0e3       # 离子温度轴心值 [eV]
Z_EFF = 1.8             # 有效电荷数

# ============================
# 聚变反应参数
# ============================
DT_ENERGY_FUS = 17.6e6   # D+T -> He4 + n 每次反应释放能量 [eV]
DT_CROSS_PEAK = 5.0e-22  # <sigma*v> 在 64 keV 附近的峰值近似 [m^3/s]

# ============================
# 数值计算参数
# ============================
NR_EQUIL = 129            # 径向网格数
NTHETA_EQUIL = 129        # 极向网格数
N_DDE_STEPS = 2000        # DDE积分步数
N_FFT = 1024              # FFT点数
N_FEKETE = 16             # Fekete点数量
N_GAUSS = 64              # Gauss-Legendre求积阶数

# ============================
# 输运与延迟参数 (Mackey-Glass类比)
# ============================
TAU_TRANSPORT = 0.15       # 能量输运延迟时间 [s] (能量约束时间的分数)
BETA_TRANSPORT = 2.0       # 输运反馈增益
gamma_transport = 1.0      # 输运阻尼率
N_TRANSPORT = 9.65         # 非线性指数 (Mackey-Glass型)

# ============================
# 刚体类比参数 (用于引导中心漂移)
# ============================
I1_DRIFT = 1.6             # 类比刚体主惯性矩1
I2_DRIFT = 1.0             # 类比刚体主惯性矩2
I3_DRIFT = 2.0 / 3.0       # 类比刚体主惯性矩3

# ============================
# 矩阵市场与HB格式默认
# ============================
MM_TITLE_DEFAULT = "Tokamak MHD Stiffness Matrix"
MM_KEY_DEFAULT = "TOKAMAK1"
MM_TYPE_DEFAULT = "RUA"
MM_IFMT_DEFAULT = 8
MM_JOB_DEFAULT = 2


def get_equilibrium_params():
    """返回Grad-Shafranov平衡求解参数。"""
    return {
        "R0": R0,
        "a_minor": a_minor,
        "B0": B0,
        "kappa": KAPPA,
        "delta": DELTA,
        "q0": q0,
        "q_edge": q_edge,
        "nr": NR_EQUIL,
        "ntheta": NTHETA_EQUIL,
    }


def get_transport_params():
    """返回延迟输运模型参数。"""
    return {
        "gamma": gamma_transport,
        "beta": BETA_TRANSPORT,
        "n": N_TRANSPORT,
        "tau": TAU_TRANSPORT,
        "t0": 0.0,
        "y0": np.array([0.5]),
        "tstop": 10.0,
    }


def get_fusion_params():
    """返回聚变反应动力学参数。"""
    return {
        "k": 1.0e-18,   # 有效反应速率 [m^3/s]，简化模型
        "t0": 0.0,
        "y0": np.array([1.0e20, 1.0e20, 0.0]),   # n_D, n_T, n_He [m^-3]
        "tstop": 100.0,
    }


def get_drift_params():
    """返回引导中心漂移类比参数。"""
    return {
        "i1": I1_DRIFT,
        "i2": I2_DRIFT,
        "i3": I3_DRIFT,
        "t0": 0.0,
        "y0": np.array([np.cos(0.9), 0.0, np.sin(0.9)]),
        "tstop": 50.0,
    }
