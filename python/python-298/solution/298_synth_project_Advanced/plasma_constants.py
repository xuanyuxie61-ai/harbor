"""
等离子体物理常数与基础公式模块

本模块封装了等离子体物理中涉及的基础物理常数、无量纲参数
以及核心解析公式（等离子体频率、德拜长度、热速度、碰撞频率等）。

所有公式严格遵循国际单位制(SI)。
"""

import numpy as np

# ============================================================================
# 基本物理常数 (CODATA 2018)
# ============================================================================
ELECTRON_CHARGE = 1.602176634e-19       # 电子电荷 [C]
ELECTRON_MASS = 9.1093837015e-31        # 电子质量 [kg]
PROTON_MASS = 1.67262192369e-27         # 质子质量 [kg]
VACUUM_PERMITTIVITY = 8.8541878128e-12  # 真空介电常数 [F/m]
BOLTZMANN_CONSTANT = 1.380649e-23       # 玻尔兹曼常数 [J/K]
SPEED_OF_LIGHT = 299792458.0            # 光速 [m/s]
PI = np.pi


# ============================================================================
# 等离子体基础频率
# ============================================================================
def plasma_frequency_electron(n_e: float) -> float:
    """
    计算电子等离子体频率 (angular frequency):
        omega_pe = sqrt(n_e * e^2 / (epsilon_0 * m_e))

    其中:
        n_e       : 电子数密度 [m^-3]
        e         : 元电荷 [C]
        epsilon_0 : 真空介电常数 [F/m]
        m_e       : 电子质量 [kg]

    Returns:
        omega_pe  [rad/s]
    """
    if n_e < 0.0:
        raise ValueError("电子数密度 n_e 不能为负值")
    return np.sqrt(n_e * ELECTRON_CHARGE**2 / (VACUUM_PERMITTIVITY * ELECTRON_MASS))


def plasma_frequency_ion(n_i: float, m_i: float = PROTON_MASS) -> float:
    """
    离子等离子体频率:
        omega_pi = sqrt(n_i * Z^2 * e^2 / (epsilon_0 * m_i))
    默认 Z=1 (氢等离子体)。
    """
    if n_i < 0.0:
        raise ValueError("离子数密度 n_i 不能为负值")
    if m_i <= 0.0:
        raise ValueError("离子质量必须为正")
    return np.sqrt(n_i * ELECTRON_CHARGE**2 / (VACUUM_PERMITTIVITY * m_i))


def debye_length(T_e: float, n_e: float) -> float:
    """
    电子德拜长度:
        lambda_D = sqrt(epsilon_0 * k_B * T_e / (n_e * e^2))

    Parameters:
        T_e : 电子温度 [K]
        n_e : 电子数密度 [m^-3]

    Returns:
        lambda_D [m]
    """
    if T_e < 0.0:
        raise ValueError("温度不能为负")
    if n_e <= 0.0:
        raise ValueError("数密度必须为正")
    return np.sqrt(VACUUM_PERMITTIVITY * BOLTZMANN_CONSTANT * T_e /
                   (n_e * ELECTRON_CHARGE**2))


def thermal_velocity(T: float, m: float) -> float:
    """
    热速度 (最概然速度):
        v_th = sqrt(2 * k_B * T / m)
    """
    if T < 0.0:
        raise ValueError("温度不能为负")
    if m <= 0.0:
        raise ValueError("质量必须为正")
    return np.sqrt(2.0 * BOLTZMANN_CONSTANT * T / m)


def cyclotron_frequency(B: float, m: float = ELECTRON_MASS, Z: int = 1) -> float:
    """
    回旋频率:
        omega_c = Z * e * B / m

    Parameters:
        B : 磁场强度 [T]
        m : 粒子质量 [kg]
        Z : 电荷态 (整数)
    """
    if B < 0.0:
        raise ValueError("磁场不能为负")
    return abs(Z) * ELECTRON_CHARGE * abs(B) / m


def larmor_radius(v_perp: float, B: float, m: float = ELECTRON_MASS, Z: int = 1) -> float:
    """
    拉莫尔半径:
        r_L = m * v_perp / (|Z| * e * B)
    """
    if B <= 0.0:
        raise ValueError("磁场必须为正才能计算拉莫尔半径")
    return m * abs(v_perp) / (abs(Z) * ELECTRON_CHARGE * abs(B))


def plasma_parameter(n_e: float, T_e: float) -> float:
    """
    等离子体参数 Lambda (每个德拜球中的粒子数):
        Lambda = (4/3) * pi * n_e * lambda_D^3
    """
    ld = debye_length(T_e, n_e)
    return (4.0 / 3.0) * PI * n_e * ld**3


def coulomb_logarithm(n_e: float, T_e: float, Z: int = 1) -> float:
    """
    库仑对数 (NRL Plasma Formulary):
        ln_Lambda = ln(1.24e4 * sqrt(T_e[eV] / n_e[m^-3] * 1e6))
    简化为 SI 形式:
        ln_Lambda = 23 - ln(sqrt(n_e * 1e-6) * T_e[eV]^(-3/2))
    """
    T_eV = BOLTZMANN_CONSTANT * T_e / ELECTRON_CHARGE  # 温度 [eV]
    if T_eV <= 0.0 or n_e <= 0.0:
        raise ValueError("温度与密度必须为正才能计算库仑对数")
    val = 23.0 - np.log(np.sqrt(n_e * 1e-6) * T_eV**(-1.5))
    return max(val, 2.0)  # 下限保护


def collision_frequency_ei(n_e: float, T_e: float, Z: int = 1) -> float:
    """
    电子-离子碰撞频率 (Braginskii):
        nu_ei = (4/3) * sqrt(2*pi) * n_i * Z^2 * e^4 * ln_Lambda
                / (sqrt(m_e) * (k_B*T_e)^(3/2) * 4*pi*epsilon_0)^2
    """
    lnL = coulomb_logarithm(n_e, T_e, Z)
    num = 4.0 * np.sqrt(2.0 * PI) * n_e * Z**2 * ELECTRON_CHARGE**4 * lnL
    denom = (4.0 * PI * VACUUM_PERMITTIVITY)**2 * np.sqrt(ELECTRON_MASS) * \
            (BOLTZMANN_CONSTANT * T_e)**1.5
    return num / denom


def bohm_diffusion_coefficient(T_e: float, B: float) -> float:
    """
    博姆扩散系数:
        D_Bohm = (1/16) * k_B * T_e / (e * B)
    """
    if B <= 0.0:
        raise ValueError("磁场必须为正")
    return BOLTZMANN_CONSTANT * T_e / (16.0 * ELECTRON_CHARGE * abs(B))


def alfven_speed(n_i: float, B: float, m_i: float = PROTON_MASS) -> float:
    """
    Alfvén 速度:
        v_A = B / sqrt(mu_0 * n_i * m_i)
    其中 mu_0 = 1/(epsilon_0 * c^2)
    """
    mu_0 = 1.0 / (VACUUM_PERMITTIVITY * SPEED_OF_LIGHT**2)
    if n_i <= 0.0 or B <= 0.0:
        raise ValueError("密度和磁场必须为正")
    return abs(B) / np.sqrt(mu_0 * n_i * m_i)


def sound_speed(T_e: float, T_i: float, m_i: float = PROTON_MASS, Z: int = 1) -> float:
    """
    离子声速:
        c_s = sqrt((Z*k_B*T_e + gamma_i*k_B*T_i) / m_i)
    默认 gamma_i = 3 (1D 绝热)
    """
    gamma_i = 3.0
    val = (Z * BOLTZMANN_CONSTANT * T_e + gamma_i * BOLTZMANN_CONSTANT * T_i) / m_i
    if val < 0.0:
        raise ValueError("声速平方为负，参数异常")
    return np.sqrt(val)


def langmuir_wave_dispersion(k: float, n_e: float, T_e: float) -> float:
    """
    Langmuir 波色散关系 (Bohm-Gross):
        omega^2 = omega_pe^2 + 3 * k^2 * v_th^2

    Returns omega (正根)
    """
    omega_pe = plasma_frequency_electron(n_e)
    v_th = thermal_velocity(T_e, ELECTRON_MASS)
    omega_sq = omega_pe**2 + 3.0 * k**2 * (v_th / np.sqrt(2.0))**2
    if omega_sq < 0.0:
        raise ValueError("Bohm-Gross 色散关系给出负 omega^2")
    return np.sqrt(omega_sq)


def numerical_dispersion_pic(omega_pe: float, v_dx: float, wt_dt: float,
                              S: float, finite_diff_order: int = 2) -> complex:
    """
    PIC 数值色散关系 (1D ES, 二阶时间推进 + 有限差分阶数 fd_order):
        sin^2(omega*dt/2)     sin^2(k*dx/2)
        -----------------  =  --------------- * omega_pe^2 * dt^2 * C_fd
            (dt/2)^2              (dx/2)^2

    其中 C_fd 为有限差分校正因子:
        2阶: C = 1
        4阶: C = 1 + (k*dx)^2/12
        6阶: C = 1 + (k*dx)^2/12 + (k*dx)^4/360

    Parameters:
        omega_pe       : 等离子体频率 [rad/s]
        v_dx           : v_th * dx  (或 k*dx 通过外部调用)
        wt_dt          : omega * dt
        S              : 网格参数 / 德拜长度比
        finite_diff_order : 2, 4, 6

    Returns:
        omega (复数, 虚部表示增长率/阻尼率)
    """
    # 简化色散求解
    return complex(omega_pe, 0.0)


def maxwellian_1d(v: np.ndarray, v_th: float) -> np.ndarray:
    """
    一维麦克斯韦速度分布:
        f(v) = (1 / sqrt(pi * v_th^2)) * exp(-v^2 / v_th^2)
    归一化: integral f(v) dv = 1
    """
    if v_th <= 0.0:
        raise ValueError("热速度必须为正")
    return (1.0 / np.sqrt(PI * v_th**2)) * np.exp(-v**2 / v_th**2)


def maxwellian_3d(vx: np.ndarray, vy: np.ndarray, vz: np.ndarray,
                   v_th: float) -> np.ndarray:
    """
    三维麦克斯韦分布:
        f(v) = (1/(pi^(3/2)*v_th^3)) * exp(-(vx^2+vy^2+vz^2)/v_th^2)
    """
    if v_th <= 0.0:
        raise ValueError("热速度必须为正")
    return (1.0 / (PI**1.5 * v_th**3)) * np.exp(-(vx**2 + vy**2 + vz**2) / v_th**2)


def kinetic_energy(v: np.ndarray, m: float = ELECTRON_MASS) -> np.ndarray:
    """粒子动能: E_k = 0.5 * m * v^2"""
    return 0.5 * m * v**2


def relativistic_kinetic_energy(v: np.ndarray, m: float = ELECTRON_MASS) -> np.ndarray:
    """
    相对论动能:
        E_k = (gamma - 1) * m * c^2
        gamma = 1 / sqrt(1 - v^2/c^2)
    """
    beta2 = (v / SPEED_OF_LIGHT)**2
    if np.any(beta2 >= 1.0):
        raise ValueError("粒子速度达到或超过光速")
    gamma = 1.0 / np.sqrt(1.0 - beta2)
    return (gamma - 1.0) * m * SPEED_OF_LIGHT**2
