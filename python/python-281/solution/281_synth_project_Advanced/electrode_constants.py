"""
electrode_constants.py
======================
物理常数与 NMC 电极材料参数模块。
融合项目: 053_asa266 (统计/特殊函数常量)

包含:
  - 基本物理常数 (Faraday, R, k_B, e)
  - NMC-622 材料参数 (D_ref, E_a, vm, S_A)
  - 热力学辅助函数 (Arrhenius扩散系数)
"""

import math

# ==================== 基本物理常数 (SI) ====================
FARADAY = 96485.33212          # C/mol, Faraday 常数
R_GAS = 8.314462618            # J/(mol·K), 气体常数
K_BOLTZMANN = 1.380649e-23     # J/K, Boltzmann 常数
E_CHARGE = 1.602176634e-19     # C, 元电荷
AVOGADRO = 6.02214076e23       # 1/mol, Avogadro 常数

# ==================== NMC-622 材料参数 ====================
T_REF = 298.15                 # K, 参考温度 (25°C)
D_REF = 3.9e-14                # m²/s, 参考扩散系数 (c_ref=0.5, T=T_ref)
E_ACT = 42.0e3                 # J/mol, Li+在NMC中的活化能
V_MOLAR = 3.57e-5              # m³/mol, NMC 摩尔体积
C_MAX = 50000.0                # mol/m³, NMC中Li最大嵌入浓度
PARTICLE_RADIUS = 2.0e-6       # m, 正极颗粒半径 (2 μm)

# ==================== 电化学动力学参数 ====================
ALPHA_ANODE = 0.5              # 阳极传递系数
ALPHA_CATHODE = 0.5            # 阴极传递系数
I0_EXCHANGE = 4.0              # A/m², 交换电流密度参考值
K_REACTION = 2.0e-11           # m/(mol^{1/2}·s^{1/2}), 反应速率常数

# ==================== 网格与时间参数 ====================
N_GRID = 64                    # 空间网格点数 (小规模可复现)
N_CHEBYSHEV = 16               # Chebyshev 积分点数
N_TIME_STEPS = 500             # 时间步数
DT_INITIAL = 1.0e-3            # s, 初始时间步长
T_TOTAL = 3600.0               # s, 总模拟时间 (1小时)
C_RATE = 1.0                   # C倍率

# ==================== 数值常数 ====================
COMPACT_FD_ORDER = 4           # 紧致有限差分精度阶数
TOLERANCE_NEWTON = 1.0e-10     # Newton 迭代收敛容差
MAX_NEWTON_ITER = 50           # 最大 Newton 迭代次数
SAFETY_FACTOR = 0.9            # 自适应时间步安全因子


def arrhenius_diffusivity(D_ref, E_a, T):
    """
    Arrhenius 扩散系数温度修正。

    D(T) = D_ref * exp(-E_a/R * (1/T - 1/T_ref))

    Parameters
    ----------
    D_ref : float
        参考温度下的扩散系数 [m²/s]
    E_a : float
        活化能 [J/mol]
    T : float
        当前温度 [K]

    Returns
    -------
    float
        修正后的扩散系数 [m²/s]
    """
    if T <= 0.0:
        raise ValueError("温度必须为正 (T > 0 K)")
    exponent = -E_a / R_GAS * (1.0 / T - 1.0 / T_REF)
    # 数值安全: 防止指数溢出
    exponent = max(min(exponent, 500.0), -500.0)
    return D_ref * math.exp(exponent)


def thermal_voltage(T):
    """
    热电压 V_T = R*T / F

    Parameters
    ----------
    T : float
        温度 [K]

    Returns
    -------
    float
        热电压 [V]
    """
    if T <= 0.0:
        raise ValueError("温度必须为正")
    return R_GAS * T / FARADAY


def dimensionless_groups(T, L_char):
    """
    计算无量纲数群。

    - Thiele modulus: φ = L * sqrt(k/D)
    - Peclet number: Pe = v*L/D (此处 v=0, 纯扩散)
    - Damköhler number: Da = 反应速率/扩散速率

    Parameters
    ----------
    T : float
        温度 [K]
    L_char : float
        特征长度 [m]

    Returns
    -------
    dict
        包含各无量纲数的字典
    """
    D = arrhenius_diffusivity(D_REF, E_ACT, T)
    # Thiele modulus (假设一级反应)
    k_react = K_REACTION * 1e3  # 近似反应速率
    phi_thiele = L_char * math.sqrt(k_react / max(D, 1e-30))
    # Damköhler number
    da_number = k_react * L_char / max(D, 1e-30)
    return {
        'thiele_modulus': phi_thiele,
        'peclet_number': 0.0,  # 纯扩散
        'damkohler_number': da_number,
        'diffusivity': D
    }


def erf_approx(x):
    """
    误差函数的近似计算 (项目053: alnorm 关联)。
    使用 Abramowitz & Stegun 近似 (7.1.26)。

    |ε(x)| ≤ 3.6e-7

    erf(x) = 1 - (a1*t + a2*t² + a3*t³)*exp(-x²)
    其中 t = 1/(1 + 0.47047*x)

    Parameters
    ----------
    x : float
        自变量

    Returns
    -------
    float
        erf(x) 的近似值
    """
    sign = 1.0 if x >= 0 else -1.0
    x = abs(x)
    # Abramowitz & Stegun 常数
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    p = 0.3275911

    t = 1.0 / (1.0 + p * x)
    t2 = t * t
    t3 = t2 * t
    t4 = t3 * t
    t5 = t4 * t

    y = 1.0 - (a1*t + a2*t2 + a3*t3 + a4*t4 + a5*t5) * math.exp(-x * x)
    return sign * y


def normal_cdf(x, mu=0.0, sigma=1.0):
    """
    正态分布累积分布函数 (项目053: alnorm)。

    Φ(x) = 0.5 * (1 + erf((x - μ) / (σ * √2)))

    Parameters
    ----------
    x : float
        评估点
    mu : float
        均值
    sigma : float
        标准差 (必须为正)

    Returns
    -------
    float
        CDF 值 ∈ [0, 1]
    """
    if sigma <= 0:
        raise ValueError("标准差必须为正")
    z = (x - mu) / (sigma * math.sqrt(2.0))
    return 0.5 * (1.0 + erf_approx(z))


def digamma_approx(x):
    """
    Digamma 函数 ψ(x) = d/dx [ln Γ(x)] 的近似计算。
    (项目053: digamma/trigamma)

    使用渐近展开 (x > 6):
    ψ(x) ≈ ln(x) - 1/(2x) - 1/(12x²) + 1/(120x⁴) - 1/(252x⁶)

    对于小x, 使用递推 ψ(x) = ψ(x+1) - 1/x

    Parameters
    ----------
    x : float
        正实数

    Returns
    -------
    float
        ψ(x) 的近似值
    """
    if x <= 0:
        raise ValueError("digamma 函数的参数必须为正")

    result = 0.0
    # 递推至 x > 6
    while x < 6.0:
        result -= 1.0 / x
        x += 1.0

    # 渐近展开
    inv_x = 1.0 / x
    inv_x2 = inv_x * inv_x
    result += math.log(x) - 0.5 * inv_x
    result -= inv_x2 / 12.0
    result += inv_x2 * inv_x2 / 120.0
    result -= inv_x2 * inv_x2 * inv_x2 / 252.0

    return result


def trigamma_approx(x):
    """
    Trigamma 函数 ψ₁(x) = d²/dx² [ln Γ(x)] 的近似。
    (项目053: trigamma)

    渐近展开:
    ψ₁(x) ≈ 1/x + 1/(2x²) + 1/(6x³) - 1/(30x⁵) + 1/(42x⁷)

    Parameters
    ----------
    x : float
        正实数

    Returns
    -------
    float
        ψ₁(x) 的近似值
    """
    if x <= 0:
        raise ValueError("trigamma 函数的参数必须为正")

    result = 0.0
    while x < 6.0:
        result += 1.0 / (x * x)
        x += 1.0

    inv_x = 1.0 / x
    inv_x2 = inv_x * inv_x
    result += inv_x + 0.5 * inv_x2
    result += inv_x2 * inv_x / 6.0
    result -= inv_x2 * inv_x2 * inv_x / 30.0
    result += inv_x2 * inv_x2 * inv_x2 * inv_x / 42.0

    return result
