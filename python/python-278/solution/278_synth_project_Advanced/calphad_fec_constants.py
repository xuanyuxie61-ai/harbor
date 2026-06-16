"""
calphad_fec_constants.py
========================
Fe-C 二元合金体系的 CALPHAD 热力学参数与物理常数。
所有能量单位为 J/mol，温度单位为 K，摩尔分数为无量纲量。

参考文献:
  [1] Gustafson, G. (1985). TRITA-MAC 0244.
  [2] Lee, B.-J. (1992). CALPHAD, 16(2), 121-134.
  [3] Andersson, J. et al. (2020). SSOL5 thermodynamic database.
"""

import math

# ============================================================
# 基本物理常数
# ============================================================
R_GAS = 8.3145          # 理想气体常数 J/(mol·K)
LN2_CORR = math.log(2.0)  # 磁有序修正中的 ln(2)
TAU_MAGIC = 0.40        # BCC 铁磁 s 参数经验值
TAU_FCC_MAGIC = 0.28    # FCC 铁磁 s 参数经验值

# ============================================================
# Fe-C 体系 CALPHAD 参数 (SSOL5 兼容)
# ============================================================

# ----- 纯组元: Fe -----
# G_Fe_LIQUID(T): 液相纯铁晶格稳定性 (J/mol)
# 采用 δ-Fe → L → γ-Fe 的统一描述
G_FE_LIQUID = [
    # T 的多项式系数: a0 + a1*T + a2*T*ln(T) + a3*T^2 + a4*T^3 + a5*T^(-1) + a6*T^7 + a9*T^(-9)
    # 简化为有效多项式
    13265.57,     # a0
    117.57557,    # a1
    -23.76114,    # a2  (T*lnT 系数)
    -0.015964688, # a3  (T^2)
    -2.68572e-6,  # a4  (T^3)
    -1.03144e9,   # a5  (T^-1, 对应 H_298 修正)
]
# 分段温度范围边界 (K)
T_FE_LIQUID_RANGE = [(298.15, 1811.0), (1811.0, 6000.0)]

# G_Fe_BCC(T): α/δ-Fe (BCC_A2)
G_FE_BCC = [
    0.0,            # a0 (参考态偏移)
    0.0,            # a1 (已归入 HSER)
    0.0,            # a2
    0.0,            # a3
    0.0,            # a4
    0.0,            # a5
]
# BCC 磁性参数 (Fe)
TC_FE_BCC = 1043.0   # Curie 温度 (K)
BMAG_FE_BCC = 2.22   # Bohr 磁子

# G_Fe_FCC(T): γ-Fe (FCC_A1)
G_FE_FCC = [
    -1462.40,       # a0
    8.282,          # a1
    -1.15e-4,       # a2  (T^2)
    -6.4e-8,        # a3  (T^3, 修正)
]
# FCC 磁性参数 (Fe)
TC_FE_FCC = -201.0   # Néel 温度 (反铁磁, K, 实际为负值表示 AFM)
BMAG_FE_FCC = 0.62

# G_Fe_CEMENT(T): 渗碳体中 Fe 的贡献 (纯虚部, 实际为零)
G_FE_CEMENT = 0.0

# ----- 纯组元: C -----
# G_C_LIQUID(T): 液相纯碳 (石墨溶解外推)
G_C_LIQUID = [
    117369.0,       # a0
    -24.63,         # a1
    0.0,            # a2
    0.0,            # a3
    -4.3105e18,     # a5  (T^-9)
]

# G_C_FCC(T): FCC 中 C 的晶格稳定性 (石墨 → FCC-C 外推)
G_C_FCC = [
    17336.1,        # a0
    8.837,          # a1
]

# G_C_BCC(T): BCC 中 C 的晶格稳定性 (极稀间隙固溶)
G_C_BCC = [
    37133.24,       # a0
    1.832,          # a1
]

# G_C_CEMENT(T): 渗碳体中 C 的贡献 (相对于石墨)
G_C_CEMENT = 4624.0  # Fe3C 中 C 的化学势偏移 J/mol

# ----- 相互作用参数 (Redlich-Kister) -----
# L_FCC_FeC(T) = L0 + L1*(2x_C-1) + L2*(2x_C-1)^2
# L_FCC_FeC 同时控制 FCC 相的混溶隙 (miscibility gap)

# FCC Fe-C 交互参数
L_FCC_FE_C = [
    -17737.5,       # L^(0)_Fe,C^FCC
    16954.4,        # L^(1)_Fe,C^FCC
    3219.4,         # L^(2)_Fe,C^FCC
]

# BCC Fe-C 交互参数
L_BCC_FE_C = [
    52790.0,        # L^(0)_Fe,C^BCC
    12140.0,        # L^(1)_Fe,C^BCC
]

# 液相 Fe-C 交互参数
L_LIQUID_FE_C = [
    77260.0,        # L^(0)_Fe,C^LIQ
    -44520.0,       # L^(1)_Fe,C^LIQ
    13200.0,        # L^(2)_Fe,C^LIQ
]

# ----- 渗碳体 Fe3C 参数 -----
# Fe3C 为化学计量化合物, 仅在极窄范围存在
G_CEMENT_FE3C = [
    # G_Fe3C(T) = a + b*T (相对于 3*G_Fe + G_C)
    24530.0,        # a (J/mol-formula)
    -3.145,         # b (J/(mol·K))
]

# ============================================================
# 数值计算参数
# ============================================================

# 成分空间
X_C_MIN = 1.0e-8       # 最小碳摩尔分数
X_C_MAX = 0.25         # 最大碳摩尔分数 (Fe-C 体系实用范围)
N_X_GRID = 201         # 成分网格点数
X_GRID_TYPE = "chebyshev"  # "uniform" 或 "chebyshev" (Chebyshev-Gauss-Lobatto)

# 温度空间
T_MIN = 700.0          # 最低温度 (K)
T_MAX = 1800.0         # 最高温度 (K)
N_T_GRID = 51          # 温度网格点数
T_GRID_TYPE = "uniform"

# Cahn-Hilliard 方程参数
CH_MOBILITY_0 = 1.0e-15   # 基准迁移率 M_0 (m²/(J·s))
CH_KAPPA = 1.5e-9         # 梯度能系数 κ (J·m/mol)
CH_LX = 1.0e-6            # 模拟域长度 L_x (m)
CH_NX = 128               # 空间网格点数 (必须为 2 的幂)
CH_DT = 1.0e-4            # 时间步长 (s)
CH_N_STEPS = 500          # 时间步数
CH_OMEGA_FD = 0.45        # 高阶紧致差分权重参数

# Newton-Maehly 求解器参数
NM_MAX_ITER = 80            # 最大迭代次数 (小规模实验)
NM_TOL = 1.0e-8             # 收敛容差
NM_DEF_TOL = 1.0e-10        # 消去因子容差

# Backward Euler 参数
BE_MAX_PICARD = 30          # Picard 迭代最大次数
BE_TOL = 1.0e-8             # Picard 收敛容差

# 蒙特卡罗参数
MC_N_SAMPLES = 500          # 蒙特卡罗样本数 (小规模实验)
MC_N_DIM = 6                # 参数空间维度 (对应 N_X_GRID 子集)
MC_RANDOM_SEED = 278        # 随机种子 (可复现性)

# MOEA/D 参数 (小规模可复现实验)
MOEAD_POP_SIZE = 12         # 种群规模 (小规模实验)
MOEAD_N_GEN = 6             # 进化代数 (小规模实验)
MOEAD_N_OBJ = 2             # 目标函数数
MOEAD_ETA = 20.0            # 多项式变异分布指数
MOEAD_PBI_THETA = 5.0       # PBI 罚参数

# 稳定性分析参数
SA_N_WAVENUM = 64         # 波数分析点数
SA_OMEGA_MAX = 1.0e8      # 最大角频率 (rad/m)

# 分岔检测参数
BF_TOL = 1.0e-6           # 分岔检测容差
BF_DJUMP_THRESH = 1.0e-4  # Jacobian 行列式跳跃阈值

# ============================================================
# 辅助函数
# ============================================================

def evaluate_poly(coeffs, T):
    """
    计算 T 的多项式: sum_k coeffs[k] * phi_k(T)
    其中 phi_k 依次为: 1, T, T*ln(T), T^2, T^3, T^(-1), T^7, T^(-9)

    Parameters
    ----------
    coeffs : list[float]
        多项式系数 (长度不超过 8)
    T : float
        温度 (K)

    Returns
    -------
    float
        多项式值 (J/mol)
    """
    if T <= 0.0:
        T = 1.0  # 边界保护
    basis = [
        1.0,
        T,
        T * math.log(T),
        T * T,
        T * T * T,
        1.0 / T if abs(T) > 1e-30 else 0.0,
        T ** 7 if T < 1e10 else 0.0,
        1.0 / (T ** 9) if abs(T) > 1e-30 else 0.0,
    ]
    result = 0.0
    n = min(len(coeffs), len(basis))
    for k in range(n):
        result += coeffs[k] * basis[k]
    return result


def redlich_kister_sum(L_params, x):
    """
    计算 Redlich-Kister 多项式:
        Omega^E = sum_{n=0}^{N} L^{(n)} * (2x - 1)^n

    这是 CALPHAD 中描述二元过剩 Gibbs 能的标准形式。

    Parameters
    ----------
    L_params : list[float]
        交互参数 L^(0), L^(1), ..., L^(N)
    x : float
        溶质摩尔分数

    Returns
    -------
    float
        过剩 Gibbs 能贡献 (J/mol)
    """
    xi = 2.0 * x - 1.0  # xi = (x_B - x_A), 范围 [-1, 1]
    result = 0.0
    xi_power = 1.0
    for n, Ln in enumerate(L_params):
        result += Ln * xi_power
        xi_power *= xi
    return result


def rk_derivative(L_params, x, order=1):
    """
    计算 Redlich-Kister 多项式对 x 的 n 阶导数:
        d/dx [sum L^(n) * (2x-1)^n]

    用于化学势计算。

    Parameters
    ----------
    L_params : list[float]
        交互参数
    x : float
        溶质摩尔分数
    order : int
        导数阶数 (1 或 2)

    Returns
    -------
    float
        导数值
    """
    xi = 2.0 * x - 1.0
    result = 0.0
    if order == 1:
        for n in range(1, len(L_params)):
            coeff = L_params[n] * n * (2.0 ** n)
            # d/dx [(2x-1)^n] = n * 2 * (2x-1)^(n-1)
            power = 1.0
            for _ in range(n - 1):
                power *= xi
            result += L_params[n] * n * 2.0 * power
    elif order == 2:
        for n in range(2, len(L_params)):
            power = 1.0
            for _ in range(n - 2):
                power *= xi
            result += L_params[n] * n * (n - 1) * 4.0 * power
    return result
