"""
topmass_constants.py — 物理常数、标准模型参数与顶夸克质量先验
===========================================================
本模块定义计算高能物理中顶夸克质量测量所需的全部基本常数,
包括标准模型参数、QCD 耦合常数、部分子分布函数相关量、
以及探测器系统参数。所有数值均基于 PDG 2024 推荐值。

核心公式:
  - 强耦合常数跑动 (5-loop β 函数):
    α_s(μ) = α_s(mZ) / (1 + β₀ t + β₁ t² + β₂ t³ + β₃ t⁴ + β₄ t⁵)
    其中 t = α_s(mZ) β₀ ln(μ²/mZ²) / (4π)
  - 顶夸克衰变宽度 (LO):
    Γ_t = (G_F m_t³) / (8π√2) × (1 - m_W²/m_t²)² × (1 + 2m_W²/m_t²)
  - 顶夸克寿命: τ_t = ℏ / Γ_t
"""

import numpy as np


# ============================================================
# 基本物理常数 (PDG 2024)
# ============================================================
SPEED_OF_LIGHT = 2.99792458e8  # m/s
HBAR = 6.582119569e-25  # GeV·s (约化普朗克常数)
HBAR_GEVS = HBAR
PI = np.pi
EULER_GAMMA = 0.5772156649015329  # Euler-Mascheroni 常数
ZETA3 = 1.2020569031595942  # Riemann ζ(3)
ZETA5 = 1.0369277551433699  # Riemann ζ(5)

# ============================================================
# 标准模型电弱参数
# ============================================================
G_FERMI = 1.1663788e-5  # GeV⁻² (费米常数)
ALPHA_EM_MZ = 1.0 / 127.955  # 电磁耦合常数 (μ = mZ)
SIN2THETA_W = 0.23122  # Weinberg 弱混合角
M_W_BOSON = 80.377  # GeV (W 玻色子质量)
M_Z_BOSON = 91.1876  # GeV (Z 玻色子质量)
M_HIGGS = 125.10  # GeV (希格斯玻色子质量)

# ============================================================
# 夸克质量 (MS-bar 方案, μ = m 标度)
# ============================================================
M_TOP_POLE_DEFAULT = 172.5  # GeV (顶夸克 pole 质量默认值)
M_TOP_MS_BAR = 162.5  # GeV (顶夸克 MS-bar 质量)
M_BOTTOM_MS_BAR = 4.18  # GeV (底夸克 MS-bar 质量)
M_CHARM_MS_BAR = 1.27  # GeV (粲夸克 MS-bar 质量)

# ============================================================
# QCD 参数
# ============================================================
ALPHA_S_MZ = 0.1179  # 强耦合常数 (μ = mZ)
N_COLORS = 3  # QCD 色数
N_FLAVORS_ACTIVE = 5  # 活跃夸克味数 (μ > m_b)

# β 函数系数 (5-loop): β(α_s) = -Σ β_i (α_s/4π)^(i+1)
BETA0 = (11.0 * N_COLORS - 2.0 * N_FLAVORS_ACTIVE) / 3.0  # = 23/3
BETA1 = (34.0 * N_COLORS**2 - 10.0 * N_COLORS * N_FLAVORS_ACTIVE -
         3.0 * (N_COLORS**2 - 1.0) / N_COLORS * N_FLAVORS_ACTIVE) / 12.0
# β₁ = (34×9 - 10×3×5 - 3×8/3×5)/12 = (306 - 150 - 40)/12 = 116/12 = 29/3
BETA2 = 142.883  # 已知数值
BETA3 = 1093.34  # 已知数值

# ============================================================
# 顶夸克衰变相关
# ============================================================
CKM_VTB = 0.999172  # CKM 矩阵元 |V_tb|
CKM_VTD = 0.00854  # |V_td|
CKM_VTS = 0.04064  # |V_ts|


def alpha_s_running(mu_scale, n_loop=4):
    """
    强耦合常数 α_s(μ) 的多圈跑动计算。

    利用重整化群方程 (RGE) 从 mZ 标度演化到 μ 标度:
      dα_s/dln(μ²) = β(α_s) = -Σᵢ βᵢ (α_s/(4π))^(i+1)

    对于 N 圈精度, 递推求解:
      L = ln(μ²/mZ²)
      a = α_s(mZ)/(4π)
      α_s(μ)/(4π) = a - β₀ a² L + (β₀² L² - β₁ L) a³ + ...

    参数:
        mu_scale: 能标 μ (GeV)
        n_loop: 圈数 (1-4)

    返回:
        α_s(μ) 的数值
    """
    if mu_scale <= 0.0:
        raise ValueError(f"能标必须为正: μ={mu_scale}")

    a0 = ALPHA_S_MZ / (4.0 * PI)
    L = np.log(mu_scale**2 / M_Z_BOSON**2)

    a = a0
    if n_loop >= 2:
        a = a0 - BETA0 * a0**2 * L
    if n_loop >= 3:
        a += (BETA0**2 * L**2 - BETA1 * L) * a0**3
    if n_loop >= 4:
        a += (-BETA0**3 * L**3 + 5.0 / 2.0 * BETA0 * BETA1 * L**2 -
              BETA2 * L) * a0**4

    alpha_s_val = 4.0 * PI * a

    # 边界处理: 确保 α_s 在物理范围内
    if alpha_s_val <= 0.0:
        alpha_s_val = 1e-6
    if alpha_s_val > 4.0 * PI:  # 非微扰区
        alpha_s_val = 4.0 * PI * 0.9

    return alpha_s_val


def top_width_lo(m_top):
    """
    顶夸克 LO 衰变宽度 t → bW⁺。

    公式 (忽略 m_b):
      Γ_t = (G_F m_t³)/(8π√2) × |V_tb|² ×
            (1 - w)² × (1 + 2w) × (1 + δ_QCD)
    其中:
      w = (m_W/m_t)²
      δ_QCD = -(2α_s(m_t))/(3π) × (π²/3 - 5/4) [NLO QCD 修正]

    参数:
        m_top: 顶夸克质量 (GeV)

    返回:
        Γ_t (GeV)
    """
    if m_top <= M_W_BOSON:
        return 1e-10  # 运动学禁止

    w = (M_W_BOSON / m_top)**2
    prefactor = G_FERMI * m_top**3 / (8.0 * PI * np.sqrt(2.0))

    tree_level = CKM_VTB**2 * (1.0 - w)**2 * (1.0 + 2.0 * w)

    # NLO QCD 修正
    alpha_s_mt = alpha_s_running(m_top, n_loop=4)
    delta_qcd = -(2.0 * alpha_s_mt) / (3.0 * PI) * (PI**2 / 3.0 - 5.0 / 4.0)

    gamma = prefactor * tree_level * (1.0 + delta_qcd)

    # NNLO QCD 修正 (~+1%)
    delta_nnlo = 0.0085
    gamma *= (1.0 + delta_nnlo)

    return max(gamma, 1e-10)


def top_lifetime(m_top):
    """
    顶夸克寿命 τ_t = ℏ/Γ_t。

    关键物理事实: τ_t ~ 5×10⁻²⁵ s << τ_had ~ 1/Λ_QCD ~ 3×10⁻²⁴ s,
    因此顶夸克在强子化之前就已衰变, 其自旋信息可被观测。
    """
    gamma = top_width_lo(m_top)
    return HBAR_GEVS / gamma


# ============================================================
# 部分子分布函数 (PDF) 相关参数 (CT14 NNLO 简化)
# ============================================================
PDF_GLUEON_PARAMS = {
    'A_g': 4.5, 'eta_g': 0.8, 'alpha_g': 0.5, 'beta_g': 6.5
}
PDF_QUARK_PARAMS = {
    'A_u': 3.5, 'eta_u': 0.4, 'alpha_u': 0.3, 'beta_u': 5.0,
    'A_d': 2.0, 'eta_d': 0.6, 'alpha_d': 0.5, 'beta_d': 6.0,
}
PDF_SEA_PARAMS = {
    'A_s': 0.8, 'eta_s': 0.3, 'alpha_s': 0.0, 'beta_s': 7.0
}

# sqrt(s) = 13 TeV (LHC Run 2)
SQRT_S_LHC = 13000.0  # GeV


def pdf_gluon(x, Q2):
    """
    胶子部分子分布函数 xg(x, Q²) 的简化参数化。

    形式: xg(x, Q²) = A_g × x^(-α_g) × (1-x)^β_g × exp(-η_g √x)
    含 DGLAP 演化修正 (leading log):
      g(x, Q²) ≈ g(x, Q₀²) × [α_s(Q₀²)/α_s(Q²)]^(γ_gg/(2β₀))
    """
    if x <= 0.0 or x >= 1.0:
        return 0.0

    A = PDF_GLUEON_PARAMS['A_g']
    alpha = PDF_GLUEON_PARAMS['alpha_g']
    beta = PDF_GLUEON_PARAMS['beta_g']
    eta = PDF_GLUEON_PARAMS['eta_g']

    xg = A * x**(-alpha) * (1.0 - x)**beta * np.exp(-eta * np.sqrt(x))

    # DGLAP leading-log 演化
    Q02 = M_Z_BOSON**2
    if Q2 > Q02:
        alpha_s_Q0 = alpha_s_running(np.sqrt(Q02))
        alpha_s_Q = alpha_s_running(np.sqrt(Q2))
        gamma_gg = 4.0 * N_COLORS / 3.0  # 反常维度 leading term
        evolution = (alpha_s_Q0 / alpha_s_Q) ** (gamma_gg / (2.0 * BETA0))
        xg *= evolution

    return max(xg, 0.0)


def pdf_quark_flavor(x, Q2, flavor='u'):
    """
    夸克 PDF xf(x, Q²) 的简化参数化。

    形式: xf(x, Q²) = A_f × x^(-α_f) × (1-x)^β_f × (1 + η_f √x)
    """
    if x <= 0.0 or x >= 1.0:
        return 0.0

    params = PDF_QUARK_PARAMS if flavor in ('u', 'd') else PDF_SEA_PARAMS
    A = params[f'A_{flavor}']
    alpha = params[f'alpha_{flavor}']
    beta = params[f'beta_{flavor}']
    eta = params[f'eta_{flavor}']

    xf = A * x**(-alpha) * (1.0 - x)**beta * (1.0 + eta * np.sqrt(x))
    return max(xf, 0.0)
