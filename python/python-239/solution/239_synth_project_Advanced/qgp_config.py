"""
qgp_config.py — 全局物理常数与数值参数配置
==============================================

本模块定义夸克胶子等离子体(QGP)流体动力学模拟中所需的全部物理常数和数值参数。

物理单位制: 自然单位 (hbar = c = k_B = 1)
能量单位: GeV
长度单位: fm (1 fm = 1/(0.1973 GeV))
时间单位: fm/c

核心物理量:
  - 强耦合常数 alpha_s ~ 0.3 (RHIC/LHC 尺度)
  - 退禁闭温度 T_c ~ 155 MeV (QCD 相变临界温度)
  - 初始温度 T_0 ~ 300-600 MeV (LHC Pb+Pb 碰撞)
  - 剪切粘滞比 eta/s ~ 1/(4*pi) (AdS/CFT KSS 下限)

数值方法参数:
  - WENO5 空间重构 (五阶加权本质无振荡格式)
  - TVD-RK3 时间推进 (三阶总变差递减 Runge-Kutta)
  - Lax-Friedrichs 全局通量分裂
  - CFL 数 ~ 0.4 (满足 von Neumann 稳定性条件)
"""

import numpy as np
from typing import Dict, Any


# ============================================================
# 物理常数 (自然单位)
# ============================================================

HBAR_C = 0.1973269804  # hbar*c in GeV*fm
BOLTZMANN = 1.0         # k_B = 1 in natural units

# QCD 参数
N_COLOR = 3             # SU(3) 色群维数
N_FLAVOR = 3            # 轻夸克味数 (u, d, s)
ALPHA_S = 0.3           # 强耦合常数 (典型硬散射尺度)
SIGMA_SB = None         # Stefan-Boltzmann 常数, 见下方计算

# 基本粒子质量 (GeV)
MASS_PROTON = 0.938272
MASS_PION = 0.13957
MASS_KAON = 0.493677
MASS_PHI = 1.01946

# 碰撞能量参数
SQRT_S_NN = 200.0       # RHIC Au+Au 质心能量 sqrt(s_NN) in GeV
IMPACT_PARAM = 7.0       # 碰撞参数 b in fm (半中心碰撞)

# QCD 相变参数
T_CRITICAL = 0.155       # 退禁闭相变温度 T_c in GeV (~155 MeV)
T_SWITCH = 0.150         # 冻结温度 T_f in GeV

# 输运系数
ETA_OVER_S_KSS = 1.0 / (4.0 * np.pi)  # AdS/CFT Kovtun-Son-Starinets 下限
ZETA_OVER_S_MAX = 0.08   # 体粘滞系数上界 (非共形修正)
TAU_PI_RELAX = 0.2       # 二阶 Israel-Stewart 弛豫时间 tau_pi in fm/c


def stefan_boltzmann_constant() -> float:
    """
    计算 QGP 的 Stefan-Boltzmann 常数

    对于理想 QGP (N_f=3, N_c=3):
    sigma_SB = (pi^2/90) * g_eff
    g_eff = g_boson + (7/8)*g_fermion
    g_boson = 2*(N_c^2 - 1) = 16  (胶子: 2极化 * 8色)
    g_fermion = 2*N_c*N_f*2 = 36  (夸克+反夸克: 2自旋*3色*3味*2粒子反粒子)

    Returns:
        sigma_SB in GeV/fm^3 单位制
    """
    g_boson = 2.0 * (N_COLOR**2 - 1)         # 胶子自由度
    g_fermion = 2.0 * N_COLOR * N_FLAVOR * 2  # 夸克+反夸克自由度
    g_eff = g_boson + (7.0 / 8.0) * g_fermion
    sigma = (np.pi**2 / 90.0) * g_eff
    return sigma


# 初始化 Stefan-Boltzmann 常数
SIGMA_SB = stefan_boltzmann_constant()


# ============================================================
# 数值方法参数
# ============================================================

class NumericalParams:
    """数值方法全局参数"""

    # 空间网格
    NX = 40                   # x 方向网格点数
    NY = 40                   # y 方向网格点数
    X_MIN = -10.0             # x 范围下界 (fm)
    X_MAX = 10.0              # x 范围上界 (fm)
    Y_MIN = -10.0             # y 范围下界 (fm)
    Y_MAX = 10.0              # y 范围上界 (fm)
    GHOST_CELLS = 3           # 边界鬼单元数 (WENO5 需要 r=3)

    # 时间推进
    T_INITIAL = 0.1           # 初始时间 tau_0 in fm/c
    T_FINAL = 10.0            # 终止时间 in fm/c
    CFL_NUMBER = 0.4          # CFL 安全系数
    DT_MAX = 0.05             # 最大时间步长 (fm/c)
    DT_MIN = 1.0e-6           # 最小时间步长 (防止除零)

    # WENO5 参数
    WENO_EPSILON = 1.0e-6     # WENO 光滑度分母小量 (防止除零)
    WENO_ALPHA = 2.0          # WENO 幂次参数

    # 人工粘滞 (熵修正)
    ARTIFICIAL_VISCOSITY = 0.01  # 人工粘滞系数

    # 守恒监控
    CONSERVATION_TOL = 1.0e-8   # 守恒量监测容差
    ENERGY_FLOOR = 1.0e-10      # 能量密度下限 (GeV/fm^3)
    PRESSURE_FLOOR = 0.0        # 压力下限
    TEMPERATURE_FLOOR = 1.0e-6  # 温度下限 (GeV)

    # MCMC 参数 (DREAM 算法)
    MCMC_CHAINS = 10            # 马尔可夫链条数
    MCMC_STEPS = 200            # 每链条步数
    MCMC_CR_VALUES = [1, 3, 5]  # 交叉概率候选值
    MCMC_GAMMA_SCALE = 2.38     # DE 跳跃缩放因子 (2.38/sqrt(2*ndim) 最优)
    MCMC_BURN_IN = 50           # 预热步数
    MCMC_OUTLIER_THRESHOLD = 1.2  # Gelman-Rubin 异常链检测阈值

    # 扩散模型参数 (初始条件生成)
    DIFFUSION_STEPS = 50        # 扩散去噪步数
    DIFFUSION_SIGMA_MAX = 1.0   # 最大噪声水平
    DIFFUSION_SIGMA_MIN = 0.01  # 最小噪声水平

    # 流谐波分析
    N_HARMONICS = 6             # 分析的最高阶流谐波 v_n
    PT_BINS = 30                # 横动量分箱数
    PT_MIN = 0.2                # 最小横动量 (GeV)
    PT_MAX = 5.0                # 最大横动量 (GeV)
    PHI_BINS = 36               # 方位角分箱数

    # 谱积分
    LATTICE_POINTS = 500        # 格点积分点数
    FIBONACCI_ORDER = 20        # Fibonacci 格点阶数


# ============================================================
# 初始条件参数
# ============================================================

class InitialConditionParams:
    """QGP 初始条件参数 (Glauber 模型)"""

    # Woods-Saxon 核密度参数
    RADIUS_AU = 6.38           # Au 核半径参数 a in fm
    DIFFUSENESS_AU = 0.535     # Au 核弥散参数 a in fm

    # 初始能量沉积
    TAU_0 = 0.1                # 热化时间 tau_0 in fm/c
    T_INITIAL_MAX = 0.45       # 中心最大初始温度 (GeV)
    SMOOTHING_SIGMA = 0.5      # 初始熵密度高斯平滑宽度 (fm)

    # 流体力学初始场
    BARYON_CHEM_POT = 0.025    # 初始重子化学势 (GeV) - RHIC 下接近零
    INIT_FLOW_VELOCITY = 0.0   # 初始横向流速 (Bjorken 近似下为零)


# ============================================================
# 粒子种类数据
# ============================================================

class ParticleSpecies:
    """冻结面粒子种类属性"""

    PION_PLUS = {
        'name': 'pi^+', 'mass': 0.13957, 'charge': 1,
        'degeneracy': 1, 'is_fermion': False
    }
    PION_MINUS = {
        'name': 'pi^-', 'mass': 0.13957, 'charge': -1,
        'degeneracy': 1, 'is_fermion': False
    }
    PION_ZERO = {
        'name': 'pi^0', 'mass': 0.13498, 'charge': 0,
        'degeneracy': 1, 'is_fermion': False
    }
    KAON_PLUS = {
        'name': 'K^+', 'mass': 0.493677, 'charge': 1,
        'degeneracy': 1, 'is_fermion': False
    }
    PROTON = {
        'name': 'p', 'mass': 0.938272, 'charge': 1,
        'degeneracy': 2, 'is_fermion': True
    }
    PHI = {
        'name': 'phi', 'mass': 1.01946, 'charge': 0,
        'degeneracy': 3, 'is_fermion': False
    }

    ALL_SPECIES = [PION_PLUS, PION_MINUS, PION_ZERO,
                   KAON_PLUS, PROTON, PHI]


# ============================================================
# 辅助函数
# ============================================================

def lorentz_factor(vx: float, vy: float) -> float:
    """
    计算 Lorentz 因子 gamma = 1/sqrt(1 - v^2)

    对于相对论流体力学, 四维速度为:
        u^mu = gamma * (1, vx, vy, 0)
    其中 gamma = (1 - vx^2 - vy^2)^{-1/2}

    Args:
        vx: x 方向流速 (自然单位, c=1)
        vy: y 方向流速

    Returns:
        Lorentz 因子 gamma
    """
    v2 = vx**2 + vy**2
    v2 = min(v2, 1.0 - 1.0e-12)  # 因果性限制: v < c
    return 1.0 / np.sqrt(1.0 - v2)


def rapidity_to_pseudorapidity(pt: float, mass: float, y: float) -> float:
    """
    快度 y 到赝快度 eta_s 的变换

    运动学关系:
        tan(theta) = pt / (mt * sinh(y))
        eta_s = -ln(tan(theta/2))

    其中 mt = sqrt(pt^2 + m^2) 为横向质量

    Args:
        pt: 横动量 (GeV)
        mass: 粒子静止质量 (GeV)
        y: 快度

    Returns:
        赝快度 eta_s
    """
    mt = np.sqrt(pt**2 + mass**2)
    pz = mt * np.sinh(y)
    p_mag = np.sqrt(pt**2 + pz**2)
    if p_mag < 1.0e-15:
        return 0.0
    theta = np.arctan2(pt, pz)
    if theta < 1.0e-15:
        theta = 1.0e-15
    eta = -np.log(np.tan(theta / 2.0))
    return eta


def de_broglie_wavelength(momentum_gev: float) -> float:
    """
    计算德布罗意波长 lambda = hbar / p

    在 QGP 物理中, 热德布罗意波长 lambda_T ~ 1/T 给出
    热德 Broglie 尺度, 当 lambda_T < 核子尺寸 (~1 fm) 时,
    退禁闭相变发生.

    Args:
        momentum_gev: 动量 (GeV/c)

    Returns:
        德布罗意波长 (fm)
    """
    if momentum_gev < 1.0e-15:
        return np.inf
    return HBAR_C / momentum_gev


def entropy_density_ideal(temperature: float) -> float:
    """
    理想 QGP 熵密度

    s = (4*sigma_SB/3) * T^3

    来自热力学关系: s = dP/dT, 对于共形 EoS P = sigma*T^4/3

    Args:
        temperature: 温度 (GeV)

    Returns:
        熵密度 (GeV/fm^3)
    """
    T = max(temperature, 1.0e-10)
    return (4.0 * SIGMA_SB / 3.0) * T**3


def knudsen_number(mean_free_path: float, system_size: float) -> float:
    """
    Knudsen 数 Kn = lambda_mfp / L

    判断流体动力学适用性:
        Kn << 1: 流体动力学有效 (强耦合 QGP)
        Kn ~ 1: 需要粘滞修正
        Kn >> 1: 自由分子流, 流体动力学失效

    Args:
        mean_free_path: 平均自由程 lambda_mfp (fm)
        system_size: 系统特征尺度 L (fm)

    Returns:
        Knudsen 数
    """
    if system_size < 1.0e-15:
        return np.inf
    return mean_free_path / system_size


def reynolds_number(entropy_density: float, flow_velocity: float,
                    system_size: float, shear_viscosity: float) -> float:
    """
    相对论 Reynolds 数

    Re = s * T * L * v / eta

    高 Re 数对应于弱粘滞(近理想)流体,
    LHC 的 QGP 具有 Re ~ 20-30

    Args:
        entropy_density: 熵密度 s (1/fm^3)
        flow_velocity: 特征流速 v
        system_size: 系统尺度 L (fm)
        shear_viscosity: 剪切粘滞 eta (GeV/fm^3 * fm/c)

    Returns:
        Reynolds 数
    """
    if shear_viscosity < 1.0e-15:
        return np.inf
    T_est = (entropy_density / (4.0 * SIGMA_SB / 3.0))**(1.0/3.0)
    return entropy_density * T_est * system_size * flow_velocity / shear_viscosity
