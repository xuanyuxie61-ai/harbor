"""
thermodynamic_models.py
=======================
热力学模型模块：Redlich-Kister 展开、活度系数、浓度依赖扩散系数。
融合项目: 1081_FranciscoHS_toy-model-cis-code (物理模型参数化),
         1064_ajakef_Earthquake_Infrasound_Paper (信号模型拟合)

核心物理:
  1. 开放电路电压 (OCV) 的 Redlich-Kister 热力学展开
  2. 活度系数 γ(c) 及其对数导数 (热力学因子)
  3. 浓度依赖的扩散系数 D(c,T)
  4. 化学势 μ(c,T) 的完整表达式

数学公式:
  OCV(x) = Σ_k A_k * (2x-1)^k * (2x*(1-x)) + (R*T/F)*ln(x/(1-x))
  γ(c) = exp(Σ_k B_k * P_k(c))  -- Margules 型活度系数
  D(c,T) = D_ref * f_thermo(c) * g_mobility(c) * exp(-E_a/(R*T))
"""

import math
from electrode_constants import (
    R_GAS, FARADAY, T_REF, D_REF, E_ACT, C_MAX,
    digamma_approx, trigamma_approx
)


# ==================== Redlich-Kister 系数 (NMC-622 拟合) ====================
# 单位: V; 来自文献拟合数据
RK_COEFFICIENTS = [
    3.9422,    # A0: 基准电压
    -1.2750,   # A1: 一阶不对称
    -0.9234,   # A2: 二阶
    0.6478,    # A3: 三阶
    -0.3842,   # A4: 四阶
    0.1921,    # A5: 五阶
    -0.0856,   # A6: 六阶
    0.0312,    # A7: 七阶
]

# ==================== Margules 活度系数参数 ====================
MARGULES_ALPHA = 2.5             # 无量纲, 相互作用参数
MARGULES_BETA = -0.8             # 无量纲, 高阶修正
MARGULES_GAMMA_COEFF = 0.15      # 无量纲, 三体交互

# ==================== 迁移率修正参数 ====================
MOBILITY_POW = 1.5               # 迁移率浓度幂指数
MOBILITY_BLOCKING = 500.0        # mol/m³, 阻塞效应浓度标度


def redlich_kister_ocv(x_soc):
    """
    Redlich-Kister 展开计算开放电路电压 (OCV)。

    OCV(x) = A0/2 + (R*T/F)*ln((1-x)/x)
             + Σ_{k=0}^{N} A_k * (2x-1)^k * [x(1-x)]

    其中 x 为 SOC (state of charge), x ∈ (0, 1)。

    该表达式源自正规溶液理论 (regular solution theory):
    - 第一项: 理想混合熵贡献
    - 对数项: Nernst 方程的熵贡献
    - 求和项: excess Gibbs 自由能的 Redlich-Kister 多项式展开

    Parameters
    ----------
    x_soc : float
        荷电状态, x ∈ (0, 1)

    Returns
    -------
    float
        OCV [V]
    """
    # 边界保护: 防止 log(0) 和除零
    x = max(1e-12, min(1.0 - 1e-12, x_soc))

    # 理想 Nernst 贡献
    # V_ideal = (R*T/F) * ln((1-x)/x)
    v_ideal = (R_GAS * T_REF / FARADAY) * math.log((1.0 - x) / x)

    # Redlich-Kister excess 贡献
    # 使用 Clenshaw 递推高效计算多项式
    # P_k(ξ) = (2x-1)^k, ξ = 2x-1
    xi = 2.0 * x - 1.0
    factor = x * (1.0 - x)  # x(1-x) 权重

    # Clenshaw 递推 (数值稳定)
    n = len(RK_COEFFICIENTS)
    b_kp2 = 0.0
    b_kp1 = 0.0
    for k in range(n - 1, -1, -1):
        b_k = RK_COEFFICIENTS[k] + xi * b_kp1
        # 注意: 此处简化为 Horner 格式, 因为基函数为单项式
        b_kp2 = b_kp1
        b_kp1 = b_k

    v_excess = b_kp1 * factor

    return v_ideal + v_excess + RK_COEFFICIENTS[0] / 2.0


def ocv_derivative(x_soc, dx=1e-7):
    """
    OCV 对 SOC 的导数: dU/dx。

    使用中心差分计算, 用于热力学因子。
    dU/dx 在相变区域会变号 (spinodal 不稳定性)。

    Parameters
    ----------
    x_soc : float
        荷电状态
    dx : float
        差分步长

    Returns
    -------
    float
        dU/dx [V]
    """
    x_plus = min(x_soc + dx, 1.0 - 1e-14)
    x_minus = max(x_soc - dx, 1e-14)
    return (redlich_kister_ocv(x_plus) - redlich_kister_ocv(x_minus)) / (x_plus - x_minus)


def ocv_second_derivative(x_soc, dx=1e-5):
    """
    OCV 对 SOC 的二阶导数: d²U/dx²。

    用于判断热力学稳定性 (spinodal 边界)。
    d²U/dx² < 0 表示 spinodal 不稳定区域。

    Parameters
    ----------
    x_soc : float
        荷电状态
    dx : float
        差分步长

    Returns
    -------
    float
        d²U/dx² [V]
    """
    x0 = max(1e-10, min(1.0 - 1e-10, x_soc))
    dx_use = min(dx, x0 * 0.1, (1.0 - x0) * 0.1)
    if dx_use < 1e-14:
        return 0.0
    return (redlich_kister_ocv(x0 + dx_use)
            - 2.0 * redlich_kister_ocv(x0)
            + redlich_kister_ocv(x0 - dx_use)) / (dx_use * dx_use)


def thermodynamic_factor(x_soc):
    """
    热力学因子 Θ(c) = 1 + d(ln γ)/d(ln c)。

    与 OCV 的关系:
    Θ = (F / (R*T)) * x * (1-x) * (-dU/dx)

    物理含义:
    - Θ > 0: 热力学稳定 (Fick 扩散)
    - Θ < 0: spinodal 不稳定 ( uphill diffusion )
    - Θ = 0: spinodal 边界

    Parameters
    ----------
    x_soc : float
        荷电状态 x ∈ (0, 1)

    Returns
    -------
    float
        热力学因子 (无量纲)
    """
    x = max(1e-12, min(1.0 - 1e-12, x_soc))
    dUdx = ocv_derivative(x)
    # Θ = -(F/(R*T)) * x*(1-x) * dU/dx
    theta = -(FARADAY / (R_GAS * T_REF)) * x * (1.0 - x) * dUdx
    return theta


def margules_activity_coefficient(x_soc):
    """
    Margules 活度系数 γ(c)。

    ln(γ) = α*(1-x)² + β*(1-x)*(2x-1) + γ_c*(1-x)²*(2x-1)

    其中 α, β, γ_c 为 Margules 参数。

    该模型描述 Li 在 NMC 晶格中的非理想混合:
    - α 控制对称的非理想性
    - β 控制不对称性
    - γ_c 控制高阶相互作用

    Parameters
    ----------
    x_soc : float
        荷电状态

    Returns
    -------
    float
        活度系数 γ (无量纲, γ ≥ 0)
    """
    x = max(1e-12, min(1.0 - 1e-12, x_soc))
    omx = 1.0 - x
    xi = 2.0 * x - 1.0

    ln_gamma = (MARGULES_ALPHA * omx * omx
                + MARGULES_BETA * omx * xi
                + MARGULES_GAMMA_COEFF * omx * omx * xi)

    # 数值安全: 防止 exp 溢出
    ln_gamma = max(min(ln_gamma, 500.0), -500.0)
    return math.exp(ln_gamma)


def activity_coefficient_derivative(x_soc, dx=1e-7):
    """
    活度系数的对数导数: d(ln γ)/d(ln x)。

    数值中心差分计算。

    Parameters
    ----------
    x_soc : float
        荷电状态
    dx : float
        差分步长

    Returns
    -------
    float
        d(ln γ)/d(ln x)
    """
    x = max(1e-10, min(1.0 - 1e-10, x_soc))
    dx_use = min(dx, x * 0.01, (1.0 - x) * 0.01)
    if dx_use < 1e-15:
        return 0.0

    x_p = x + dx_use
    x_m = x - dx_use
    ln_gamma_p = math.log(max(margules_activity_coefficient(x_p), 1e-300))
    ln_gamma_m = math.log(max(margules_activity_coefficient(x_m), 1e-300))

    # d(ln γ)/d(ln x) = x * d(ln γ)/dx
    dln_gamma_dx = (ln_gamma_p - ln_gamma_m) / (x_p - x_m)
    return x * dln_gamma_dx


def mobility_function(c):
    """
    迁移率修正函数 g_mob(c)。

    考虑位阻效应和高浓度阻塞:
    g_mob(c) = 1 / (1 + (c/c_block)^p)

    当 c → c_max 时, g_mob → 0 (Li 位点饱和)。

    Parameters
    ----------
    c : float
        Li 浓度 [mol/m³]

    Returns
    -------
    float
        迁移率修正因子 ∈ (0, 1]
    """
    c_safe = max(0.0, c)
    ratio = c_safe / MOBILITY_BLOCKING
    # 幂次计算 (数值稳定)
    ratio_pow = ratio ** MOBILITY_POW if ratio > 0 else 0.0
    return 1.0 / (1.0 + ratio_pow)


def concentration_dependent_D(c, T):
    """
    浓度依赖的扩散系数 D(c, T)。

    D(c, T) = D_arrhenius(T) * g_mob(c/c_max) * Θ(c/c_max)

    其中:
    - D_arrhenius(T): Arrhenius 温度修正
    - g_mob: 迁移率修正 (位阻效应)
    - Θ: 热力学因子 (化学势梯度修正)

    完整表达式 (源自化学势驱动的扩散):
    j = -M(c) * c * ∇μ
      = -D(c,T) * ∇c
    其中 M(c) = D₀/(k_B*T) * g_mob 为迁移率

    Parameters
    ----------
    c : float
        Li 浓度 [mol/m³], c ∈ [0, c_max]
    T : float
        温度 [K]

    Returns
    -------
    float
        有效扩散系数 [m²/s]
    """
    # Arrhenius 基准
    exponent = -E_ACT / R_GAS * (1.0 / T - 1.0 / T_REF)
    exponent = max(min(exponent, 500.0), -500.0)
    D_0 = D_REF * math.exp(exponent)

    # 归一化浓度
    x = max(1e-10, min(1.0 - 1e-10, c / C_MAX))

    # 热力学因子
    theta = thermodynamic_factor(x)
    # 保证非负 (防止 uphill diffusion 导致负扩散系数)
    theta_eff = max(abs(theta), 1e-6)

    # 迁移率修正
    g_mob = mobility_function(c)

    return D_0 * theta_eff * g_mob


def diffusivity_derivative(c, T, dc=1e-3):
    """
    扩散系数对浓度的导数: ∂D/∂c。

    用于非线性扩散方程的 Jacobian。
    中心差分计算。

    Parameters
    ----------
    c : float
        Li 浓度 [mol/m³]
    T : float
        温度 [K]
    dc : float
        差分步长

    Returns
    -------
    float
        ∂D/∂c [m²/s per mol/m³]
    """
    c_safe = max(0.0, c)
    dc_use = min(dc, max(c_safe * 0.01, 1e-6))
    c_p = min(c_safe + dc_use, C_MAX * (1.0 - 1e-10))
    c_m = max(c_safe - dc_use, 0.0)
    if abs(c_p - c_m) < 1e-20:
        return 0.0
    return (concentration_dependent_D(c_p, T) -
            concentration_dependent_D(c_m, T)) / (c_p - c_m)


def chemical_potential(c, T):
    """
    Li 在 NMC 中的化学势 μ(c, T)。

    μ = μ₀ + R*T*ln(a) + F*U(x)
      = μ₀ + R*T*ln(γ*x) + F*U(x)

    其中 a = γ*x 为活度, U(x) 为 OCV。

    Parameters
    ----------
    c : float
        Li 浓度 [mol/m³]
    T : float
        温度 [K]

    Returns
    -------
    float
        化学势 (相对 μ₀) [J/mol]
    """
    x = max(1e-12, min(1.0 - 1e-12, c / C_MAX))
    gamma = margules_activity_coefficient(x)
    a_activity = gamma * x  # 活度
    # μ - μ₀ = R*T*ln(a) + F*U(x)
    ln_a = math.log(max(a_activity, 1e-300))
    return R_GAS * T * ln_a + FARADAY * redlich_kister_ocv(x)


def spinodal_boundaries(rk_coeffs=None, n_search=1000):
    """
    寻找 spinodal 边界: d²U/dx² = 0 的点。

    spinodal 边界分隔稳定和不稳定区域。
    在 spinodal 区域内, 会发生 spinodal decomposition (调幅分解)。

    Parameters
    ----------
    rk_coeffs : list, optional
        Redlich-Kister 系数 (默认使用内置值)
    n_search : int
        搜索网格点数

    Returns
    -------
    list of float
        spinodal 边界点的 SOC 值
    """
    if rk_coeffs is None:
        rk_coeffs = RK_COEFFICIENTS

    boundaries = []
    dx = 1.0 / n_search

    for i in range(1, n_search):
        x0 = i * dx
        d2U = ocv_second_derivative(x0)
        d2U_prev = ocv_second_derivative(x0 - dx)

        # 检测符号变化 (二阶导数过零)
        if d2U * d2U_prev < 0:
            # 二分法精确化
            x_lo, x_hi = x0 - dx, x0
            for _ in range(50):
                x_mid = 0.5 * (x_lo + x_hi)
                if ocv_second_derivative(x_mid) * ocv_second_derivative(x_lo) < 0:
                    x_hi = x_mid
                else:
                    x_lo = x_mid
            boundaries.append(0.5 * (x_lo + x_hi))

    return boundaries
