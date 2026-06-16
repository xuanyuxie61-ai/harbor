"""
sei_parameters.py
=================

物理常数、材料参数与电化学参数集中管理模块。

本模块为固态电解质界面（SEI）高阶有限差分反应-扩散-电迁移耦合求解器
提供统一的参数接口。所有国际单位制（SI）量纲。

核心公式：
    Faraday 常数：        F = N_A * e
    气体常数：            R = N_A * k_B
    热电压：              V_T = R * T / F
    Butler-Volmer 交换电流密度：
        j0 = F * k0 * c_li^alpha_a * c_se^alpha_c
    Nernst-Einstein 扩散系数迁移率关系：
        D_eff = D0 * exp(-E_a / (R * T))
    SEI 生长过电位：
        eta_sei = phi_s - phi_e - U_eq_sei

作者: DA-Synthesis
"""

import math


# ============================================================
#  基本物理常数（SI 单位）
# ============================================================

E_CHARGE = 1.602176634e-19           # 元电荷 e [C]
N_AVOGADRO = 6.02214076e23           # Avogadro 常数 N_A [1/mol]
K_BOLTZMANN = 1.380649e-23           # Boltzmann 常数 k_B [J/K]
FARADAY = N_AVOGADRO * E_CHARGE      # Faraday 常数 F [C/mol]
R_GAS = N_AVOGADRO * K_BOLTZMANN     # 气体常数 R [J/(mol K)]
EPSILON_0 = 8.8541878128e-12         # 真空介电常数 [F/m]


# ============================================================
#  温度与电池工况
# ============================================================

T_REF = 298.15                       # 参考温度 [K]（25 °C）
T_OPER = 298.15                      # 工作温度 [K]
V_THERMAL = R_GAS * T_OPER / FARADAY # 热电压 V_T [V] ≈ 25.69 mV


# ============================================================
#  电极/电解质材料参数（典型石墨 / EC:DMC-LiPF6）
# ============================================================

# 负极石墨
CS_MAX_GRAPHITE = 31360.0            # 石墨最大 Li 嵌入浓度 [mol/m^3]
D_S_GRAPHITE = 3.9e-14               # 固相扩散系数 [m^2/s]
EPS_S_GRAPHITE = 0.40                # 石墨颗粒体积分数 [-]

# 电解质
C_LI_INIT = 1000.0                   # 电解质初始 Li+ 浓度 [mol/m^3]
D_LI_EFF = 7.5e-10                   # 有效 Li+ 扩散系数 [m^2/s]
SIGMA_EFF = 1.1                      # 有效电导率 [S/m]
T_LI_PLUS = 0.38                     # Li+ 迁移数 [-]
EPS_ELECTROLYTE = 0.40               # 电解质体积分数 [-]
EPSILON_R_ELYTE = 22.0               # 电解质相对介电常数 [-]

# SEI 层材料
THICKNESS_SEI_INIT = 2.5e-9          # SEI 初始厚度 [m]（2.5 nm）
D_LI_SEI = 2.0e-22                   # Li+ 在 SEI 中的扩散系数 [m^2/s]
SIGMA_SEI = 1.0e-18                  # SEI 电子电导率 [S/m]
EPSILON_R_SEI = 14.0                 # SEI 相对介电常数 [-]
RHO_SEI = 2100.0                     # SEI 密度 [kg/m^3]
M_MOL_SEI = 0.160                    # SEI 产物摩尔质量 [kg/mol]（Li2CO3 等效）


# ============================================================
#  电化学动力学参数（Butler-Volmer）
# ============================================================

K0_GRAPHITE = 5.0e-6                 # 石墨嵌入反应速率常数 [m/s]
ALPHA_A_GRAPHITE = 0.5               # 阳极传递系数 [-]
ALPHA_C_GRAPHITE = 0.5               # 阴极传递系数 [-]
U_EQ_GRAPHITE_REF = 0.1              # 石墨参考平衡电位 [V vs Li/Li+]
AREA_SPECIFIC_GRAPHITE = 1.5e5       # 石墨比表面积 [1/m]

# SEI 形成反应
K0_SEI = 1.0e-11                     # SEI 形成速率常数 [m/s]
ALPHA_A_SEI = 0.5                    # SEI 反应阳极传递系数 [-]
ALPHA_C_SEI = 0.5                    # SEI 反应阴极传递系数 [-]
U_EQ_SEI = 0.4                       # SEI 反应平衡电位 [V]


# ============================================================
#  高阶有限差分离散化参数
# ============================================================

N_GRID = 81                          # 空间网格点数（奇数对称）
L_DOMAIN = 50.0e-9                   # 计算域长度 [m]（50 nm 典型 SEI 厚度）
DX = L_DOMAIN / (N_GRID - 1)         # 空间步长 [m]

# 显式格式时间步长（受 CFL 稳定性限制）
CFL_NUMBER = 0.4                     # 库朗数 [-]（显式稳定性要求 < 0.5）
DT_EXPLICIT = CFL_NUMBER * DX * DX / (2.0 * D_LI_SEI)   # 显式时间步长 [s]

# 总仿真时间
T_TOTAL = 100.0 * DT_EXPLICIT        # 仿真总时长 [s]
N_STEPS = int(round(T_TOTAL / DT_EXPLICIT))

# 高阶差分（4 阶中心差分）系数
FD4_STENCIL = [-1.0 / 12.0, 4.0 / 3.0, -5.0 / 2.0,
               4.0 / 3.0, -1.0 / 12.0]  # 四阶二阶导数模板

# Gram 多项式重构阶数
GRAM_ORDER = 4


# ============================================================
#  应力波传播参数（SEI 机械失效）
# ============================================================

YOUNG_MODULUS_SEI = 9.0e9            # SEI 弹性模量 [Pa]（9 GPa）
POISSON_RATIO_SEI = 0.28             # SEI 泊松比 [-]
PARTIAL_MOLAR_VOL_LI = 1.3e-5        # Li 偏摩尔体积 [m^3/mol]
C_WAVE_SEI = math.sqrt(
    YOUNG_MODULUS_SEI * (1.0 - POISSON_RATIO_SEI) /
    ((1.0 + POISSON_RATIO_SEI) * (1.0 - 2.0 * POISSON_RATIO_SEI) * RHO_SEI)
)                                    # SEI 中纵向弹性波速 [m/s]


# ============================================================
#  蒙特卡罗 / 随机成核参数
# ============================================================

RNG_SEED = 20260608                  # 随机数种子（可复现）
N_NUCLEATION_SITES = 200             # 候选成核位点数
NUCLEATION_ENERGY_BARRIER = 1.0      # 成核能垒 [eV]（提高以获得部分成核）
NUCLEATION_ATTEMPT_FREQ = 1.0e13     # 尝试频率 [1/s]（Debye 频率量级）


# ============================================================
#  Gauss-Laguerre / Gauss-Hermite 求积参数
# ============================================================

GL_ORDER = 8                         # Gauss-Laguerre 求积阶数
GL_ALPHA_PARAM = 0.5                 # 广义 Laguerre 权函数指数 alpha
GH_ORDER = 6                         # Gauss-Hermite 求积阶数


# ============================================================
#  SEI 形貌 Lorenz 降维模型参数
#  参考：Avramenko et al. Chaos Solitons Fractals 166 (2023)
#  状态变量：(X, Y, Z) 分别表示界面扰动幅值、溶质浓度偏差、热扰动
# ============================================================

LORENZ_SC_SEI = 10.0                 # 有效 Schmidt 数 Pr/Sc [-]
LORENZ_RAYLEIGH_SEI = 49.6           # 有效 Rayleigh 数 Ra [-]
LORENZ_B_SEI = 8.0 / 3.0             # 几何因子 b [-]


# ============================================================
#  化学计量学（不定方程）参数
#  SEI 总反应：a*EC + b*Li+ + c*e- -> d*Li2CO3 + e*C2H4
#  约束：原子守恒 + 电荷守恒 → 不定方程组
# ============================================================

DIOPHANTINE_COEFFS = [2, 1, 1, 2, 3]   # 化学计量系数向量
DIOPHANTINE_RHS = 12                    # 守恒约束右侧


# ============================================================
#  实验调度参数（Zeller 日历）
# ============================================================

EXP_YEAR = 2026
EXP_MONTH = 6
EXP_DAY = 8


# ============================================================
#  参数完整性校验
# ============================================================

def validate_parameters():
    """
    对全部参数进行物理自洽性校验。

    校验内容：
        1. 温度 > 0 K
        2. Faraday 常数与 N_A, e 自洽
        3. 扩散系数 > 0
        4. CFL 数 < 0.5（显式格式稳定性）
        5. 网格点数 >= 5（四阶差分至少需要 5 点）
        6. 浓度/模量为正值

    Returns
    -------
    bool
        True 表示通过全部校验。
    """
    checks = [
        T_OPER > 0.0,
        abs(FARADAY - N_AVOGADRO * E_CHARGE) < 1.0e-10,
        D_LI_SEI > 0.0,
        D_S_GRAPHITE > 0.0,
        CFL_NUMBER < 0.5,
        N_GRID >= 5,
        C_LI_INIT > 0.0,
        YOUNG_MODULUS_SEI > 0.0,
        RHO_SEI > 0.0,
        THICKNESS_SEI_INIT > 0.0,
        0.0 < T_LI_PLUS < 1.0,
        0.0 < POISSON_RATIO_SEI < 0.5,
    ]
    return all(checks)


if __name__ == "__main__":
    print("[sei_parameters] 参数自洽性校验:",
          "通过" if validate_parameters() else "失败")
    print(f"  Faraday 常数 F = {FARADAY:.6e} C/mol")
    print(f"  热电压     V_T = {V_THERMAL * 1000:.4f} mV")
    print(f"  SEI 波速   c   = {C_WAVE_SEI:.2f} m/s")
    print(f"  显式步长   dt  = {DT_EXPLICIT:.6e} s")
    print(f"  空间步长   dx  = {DX:.6e} m")
