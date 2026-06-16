"""
spectral_function.py — 顶夸克谱函数与阈值产生截面的格林函数方法
=================================================================
本模块实现顶夸克对产生截面在阈值区域的理论计算,
基于非相对论 QCD (NRQCD) 的格林函数方法。
这是顶夸克质量精度测量的理论基础。

核心物理:
  在 tt̄ 阈值区域 (√s ≈ 2m_t), 顶夸克对的速度 v ~ α_s ≪ 1,
  需要重求库仑胶子交换 (α_s/v)ⁿ 项到所有阶。

  关键量 — 矢量流关联函数 (格林函数) 的虚部:
    R(s) = 12π Im[C₁(s)]
  其中 C₁(s) 是 S 波矢量流关联函数:
    C₁(s) = (N_c/(4π)) × G(E + iΓ_t; 0, 0)
  G(E; r, r') 是 Schrödinger 方程的格林函数:
    [-∇²/m_t + V(r) - E] G(E; r, r') = δ³(r-r')
  V(r) 是 QCD 静态势 (包含高阶修正):
    V(r) = -C_F α_s(1/r)/r × [1 + Σ_k c_k (α_s/π)^k]
  其中 C_F = (N_c²-1)/(2N_c) = 4/3

  阈值截面:
    σ_tt̄(s) = (4πα²_em)/(3s) × N_c × Σ_q e_q² ×
              [β(3-β²)/2 × (1 + δ_hard) + R_threshold(s)]
  其中 β = √(1 - 4m_t²/s) 是速度。

  束缚态共振 (t̄t toponium):
    由于 Γ_t >> Λ_QCD, 真正的束缚态不会形成,
    但阈值增强效应仍然显著。

  Sommerfeld 增强因子 (LO):
    S(v) = z/(1 - e^(-z)), z = 2πα_s/(βC_F⁻¹) = πC_F α_s/v

映射种子项目:
  - 1028_nereagurru: 原行星盘粘滞扩散方程 → 阈值格林函数传播
  - 505_hankel_inverse: 逆汉克尔变换 → 动量空间 ↔ 坐标空间变换
"""

import numpy as np
from topmass_constants import (
    PI, ALPHA_EM_MZ, G_FERMI, N_COLORS, M_W_BOSON,
    M_TOP_POLE_DEFAULT, ALPHA_S_MZ, ZETA3,
    alpha_s_running, top_width_lo, pdf_gluon, SQRT_S_LHC
)

# SU(3) 群 Casimir 因子
C_F = (N_COLORS**2 - 1.0) / (2.0 * N_COLORS)  # = 4/3
C_A = N_COLORS  # = 3
T_F = 0.5  # fundamental representation trace normalization


def qcd_coulomb_potential(r, m_top, n_loop=3):
    """
    QCD 静态势 V(r) 的高阶计算 (动量空间 → 坐标空间)。

    在动量空间中:
      Ṽ(q) = -4πC_F α_V(q)/q²
    其中 α_V(q) 是 V-scheme 耦合常数:
      α_V(q) = α_s(q) × [1 + a₁ α_s/(4π) + a₂ (α_s/(4π))² + ...]

    系数:
      a₁ = C_A(31/9 - 2ln2/3) - 4T_F n_f/9
         + (C_A²(453/36 - 11ln2/18 + ln²2/2) + ...)

    坐标空间势:
      V(r) = ∫d³q/(2π)³ e^(iq·r) Ṽ(q)
           = -C_F α_V(1/r)/r

    参数:
        r: 夸克间距 (GeV⁻¹)
        m_top: 顶夸克质量 (GeV)
        n_loop: 微扰阶数

    返回:
        V(r) (GeV)
    """
    if r <= 1e-10:
        return -1e10  # 奇点

    mu = 1.0 / r  # 自然标度选择
    if mu > 1000.0:
        mu = 1000.0

    alpha_s = alpha_s_running(mu, n_loop=min(n_loop, 4))

    # V-scheme 系数
    a1 = C_A * (31.0 / 9.0) - 4.0 * T_F * 5 / 9.0  # n_f=5
    a2 = (C_A**2 * (453.0 / 36.0 - 0.5 * ZETA3) -
          2.0 * C_A * T_F * 5 * (71.0 / 36.0) +
          4.0 * T_F**2 * 25 / 81.0)

    alpha_V = alpha_s
    if n_loop >= 2:
        alpha_V += a1 * alpha_s**2 / (4.0 * PI)
    if n_loop >= 3:
        alpha_V += a2 * (alpha_s / (4.0 * PI))**2 * alpha_s

    V = -C_F * alpha_V / r

    return V


def sommerfeld_enhancement(beta, m_top, alpha_s):
    """
    Sommerfeld 库仑重求和增强因子。

    对于 S 波 tt̄ 产生, 库仑相互作用重求和给出:
      S(β) = z / (1 - e^(-z))
    其中:
      z = π C_F α_s / β
      β = √(1 - 4m_t²/s) 是相对速度

    当 β → 0 (阈值), S → πC_F α_s/β → ∞ (库仑奇点)
    顶夸克宽度 Γ_t 正则化此奇点: β → √(β² + Γ_t²/m_t²)

    参数:
        beta: 相对速度 (或正则化后的有效速度)
        m_top: 顶夸克质量
        alpha_s: 强耦合常数

    返回:
        S(β) 增强因子
    """
    if beta <= 1e-12:
        # 使用顶夸克宽度正则化
        return PI * C_F * alpha_s * m_top / 1e-10

    z = PI * C_F * alpha_s / beta
    if z > 50.0:
        return z  # 渐近极限
    if z < 1e-6:
        return 1.0 + z / 2.0  # Taylor 展开

    return z / (1.0 - np.exp(-z))


def greens_function_threshold(E, m_top, gamma_t, n_points=200):
    """
    阈值区域 Schrödinger 方程格林函数 G(E+iΓ; 0, 0)。

    使用数值求解径向 Schrödinger 方程:
      [-d²/dr² + l(l+1)/r² + m_t V(r) - m_t(E+iΓ)] u_l(r) = 0

    对于 S 波 (l=0):
      [-d²/dr² + m_t V(r)] u(r) = m_t(E+iΓ) u(r)

    格林函数在原点的值通过 Numerov 方法计算:
      u_{n+1} = 2u_n - u_{n-1} + h² f_n u_n (标准 Numerov 格式)

    更精确的 6 阶 Numerov 修正:
      u_{n+1} = (2(1-5h²f/12)u_n - (1+h²f/12)u_{n-1}) /
                (1 + h²f_{n+1}/12)

    映射种子项目:
      - 392_fem1d_heat_steady: FEM 弱形式 → 有限差分离散化
      - 1028_nereagurru: 原行星盘扩散 → Schrödinger 方程传播子

    参数:
        E: 质心系能量偏移 E = √s - 2m_t (GeV)
        m_top: 顶夸克质量 (GeV)
        gamma_t: 顶夸克宽度 (GeV)
        n_points: 径向网格点数

    返回:
        G(E+iΓ; 0, 0) 的虚部 (与截面成正比)
    """
    # 正则化能量 (包含宽度效应)
    E_complex = E + 1j * gamma_t

    # 径向网格: r ∈ [r_min, r_max], 对数均匀
    r_min = 0.01 / m_top  # ~ 0.003 GeV⁻¹
    r_max = 20.0 / (m_top * 0.1)  # ~ 2 GeV⁻¹
    r = np.logspace(np.log10(r_min), np.log10(r_max), n_points)
    h = np.diff(np.log(r))  # 对数步长

    # 计算有效势 (包含角动量屏障)
    V_eff = np.zeros(n_points, dtype=complex)
    for i in range(n_points):
        V_eff[i] = m_top * qcd_coulomb_potential(r[i], m_top)

    # 定义 k² = m_t(E + iΓ)
    k_squared = m_top * E_complex

    # Numerov 方法求解 Schrödinger 方程
    # 方程: u''(r) = [V_eff(r) - k²] u(r) ≡ f(r) u(r)
    # 边界条件: u(0) = 0, u(r_max) = outgoing wave

    f = np.zeros(n_points, dtype=complex)
    for i in range(n_points):
        f[i] = V_eff[i] - k_squared

    # Numerov 递推 (6 阶精度)
    # (1 + h²f_{n+1}/12) u_{n+1} =
    #   2(1 - 5h²f_n/12) u_n - (1 + h²f_{n-1}/12) u_{n-1}

    u = np.zeros(n_points, dtype=complex)
    u[0] = 0.0 + 0j
    u[1] = r[1] + 0j  # 起始条件: u ~ r 当 r→0

    # 归一化因子累积 (防止指数增长溢出)
    renorm_factor = 1.0

    for n in range(1, n_points - 1):
        hn = h[min(n, len(h) - 1)]
        hn2 = hn**2

        coeff_next = 1.0 + hn2 * f[n + 1] / 12.0
        coeff_curr = 2.0 * (1.0 - 5.0 * hn2 * f[n] / 12.0)
        coeff_prev = 1.0 + hn2 * f[n - 1] / 12.0

        if abs(coeff_next) < 1e-15:
            coeff_next = 1e-15 + 0j

        u_next = (coeff_curr * u[n] - coeff_prev * u[n - 1]) / coeff_next

        # 周期性重归一化防止溢出
        abs_unext = abs(u_next)
        if abs_unext > 1e100:
            scale = 1e100 / abs_unext
            u[:n + 1] *= scale
            u_next *= scale
            renorm_factor *= scale
        elif abs_unext < 1e-100 and abs_unext > 0:
            scale = 1e-100 / abs_unext
            u[:n + 1] *= scale
            u_next *= scale
            renorm_factor *= scale

        # NaN/Inf 保护
        if not np.isfinite(u_next.real) or not np.isfinite(u_next.imag):
            u[n + 1] = u[n]  # 保持最后有效值
        else:
            u[n + 1] = u_next

    # 格林函数 G(0,0;E) ~ |u(r_min)|² / (m_t × u'(r_min) × u(r_min))
    # 近似: Im[G] ~ Im[-m_t / (u'(r_min)/u(r_min))]

    if abs(u[1]) < 1e-30:
        return 0.0

    # 对数网格上的导数
    dr = r[1] - r[0]
    u_prime = (u[1] - u[0]) / dr if dr > 0 else 1.0

    if abs(u[0]) < 1e-30:
        u_at_origin = u[1]
    else:
        u_at_origin = u[0]

    # 对数导数
    if abs(u_at_origin) < 1e-30:
        log_deriv = u_prime / u[1] if abs(u[1]) > 1e-30 else 1.0
    else:
        log_deriv = u_prime / u_at_origin

    # Im[G(0,0)] 与截面的关系
    # G(0,0;E) = -m_t/(4π) × (u'/u)|_{r→0}
    G_imag = -m_top / (4.0 * PI) * np.imag(1.0 / log_deriv)

    return max(G_imag, 0.0)


def ttbar_threshold_cross_section(sqrt_s, m_top, n_points=200):
    """
    tt̄ 阈值产生总截面 σ(e⁺e⁻ → tt̄) 在阈值区域。

    虽然 LHC 是 pp 碰撞, 但阈值行为由以下截面描述:

    σ_tt̄(s) = σ_Born × [1 + δ_QCD(β) + δ_EW] + σ_threshold

    Born 截面 (e⁺e⁻):
      σ_Born = (4πα²)/(3s) × N_c × Σ_q e_q² × β(3-β²)/2

    阈值贡献 (来自格林函数):
      σ_threshold = (4πα²)/(3s) × 12π × Im[C₁(E+iΓ)]
      C₁ = (N_c/(4π)) × G(E+iΓ; 0, 0)

    对于 pp 碰撞, 需要与部分子光度卷积:
      σ_pp→tt̄(s_pp) = ∫dx₁dx₂ Σ_{ij} f_i(x₁)f_j(x₂) σ̂_ij(x₁x₂s_pp)

    映射种子项目:
      - 505_hankel_inverse: 动量空间变换技术

    参数:
        sqrt_s: 质心系能量 √s (GeV)
        m_top: 顶夸克质量 (GeV)
        n_points: 数值积分点数

    返回:
        σ_tt̄ (pb, 1 pb = 10⁻³⁶ cm²)
    """
    if sqrt_s <= 2.0 * m_top * 0.9:
        return 0.0

    threshold = 2.0 * m_top
    E = sqrt_s - threshold  # 阈值能量偏移
    beta = np.sqrt(max(1.0 - 4.0 * m_top**2 / sqrt_s**2, 0.0))

    alpha_s = alpha_s_running(sqrt_s / 2.0, n_loop=4)
    gamma_t = top_width_lo(m_top)

    # Born 截面 (QED 部分, 用于归一化)
    e_top = 2.0 / 3.0  # 顶夸克电荷
    sigma_born = (4.0 * PI * ALPHA_EM_MZ**2) / (3.0 * sqrt_s**2)
    sigma_born *= N_COLORS * e_top**2 * beta * (3.0 - beta**2) / 2.0

    # QCD 硬修正 δ_hard (NNLO)
    delta_hard = alpha_s / PI * (PI**2 / 3.0 - 1.0)
    delta_hard += (alpha_s / PI)**2 * 12.8

    # 阈值格林函数贡献
    G_imag = greens_function_threshold(E, m_top, gamma_t, n_points)

    # C₁ 系数
    C1_imag = N_COLORS / (4.0 * PI) * G_imag

    # 总截面 = Born × (1 + δ) + 阈值贡献
    sigma_total = sigma_born * (1.0 + delta_hard)
    sigma_total += (4.0 * PI * ALPHA_EM_MZ**2) / (3.0 * sqrt_s**2) * 12.0 * PI * C1_imag

    # 单位转换: GeV⁻² → pb (1 GeV⁻² = 0.3894 × 10⁶ pb)
    GEV2_TO_PB = 0.3894e6
    sigma_total *= GEV2_TO_PB

    return max(sigma_total, 0.0)


def pp_ttbar_luminosity(tau, m_top):
    """
    pp → tt̄ 的部分子光度函数 dL/dτ。

    对于 gg 初态 (主导贡献 ~85%):
      dL_gg/dτ = ∫_τ^1 dx/x g(x, μ_F²) g(τ/x, μ_F²)

    其中 τ = M_tt̄²/s, μ_F 是因子化标度 (取 μ_F = M_tt̄/2)。

    映射种子项目:
      - 300_disk01_integrands: 二维积分技术
      - 945_quad_trapezoid: 梯形积分规则

    参数:
        tau: τ = M²/s (无量纲)
        m_top: 顶夸克质量 (GeV)

    返回:
        dL_gg/dτ (无量纲)
    """
    if tau <= 0.0 or tau >= 1.0:
        return 0.0

    Q2 = (2.0 * m_top)**2  # 因子化标度平方
    n_quad = 50

    # Gauss-Legendre 积分
    x_nodes, w_nodes = np.polynomial.legendre.leggauss(n_quad)

    # 映射到 [τ, 1]
    x_phys = 0.5 * ((1.0 - tau) * x_nodes + (1.0 + tau))
    w_phys = 0.5 * (1.0 - tau) * w_nodes

    luminosity = 0.0
    for i in range(n_quad):
        x1 = x_phys[i]
        x2 = tau / x1
        if x2 <= 0.0 or x2 >= 1.0:
            continue
        luminosity += w_phys[i] * pdf_gluon(x1, Q2) * pdf_gluon(x2, Q2) / x1

    return max(luminosity, 0.0)


def ttbar_invariant_mass_spectrum(m_inj, m_top, sqrt_s=None):
    """
    tt̄  ne变质量谱 dσ/dM_tt̄ (微分截面)。

    这是顶夸克质量测量的核心可观测量的:
      dσ/dM = (2M/s) × σ̂_gg(M²) × dL_gg/dτ(M²/s)

    其中:
      σ̂_gg(M²) = gg → tt̄ 部分子截面 (包含阈增强)
      dL_gg/dτ = 胶子部分子光度

    映射种子项目:
      - 300_disk01_integrands: 积分区间处理
      - 945_quad_trapezoid: 数值积分精度

    参数:
        m_inj: tt̄ ne变质量 M (GeV)
        m_top: 顶夸克质量假设 (GeV)
        sqrt_s: pp 质心系能量 (GeV)

    返回:
        dσ/dM (pb/GeV)
    """
    if sqrt_s is None:
        sqrt_s = SQRT_S_LHC

    if m_inj <= 2.0 * m_top * 0.8:
        return 0.0

    # 部分子截面 (使用阈值公式)
    sigma_hat = ttbar_threshold_cross_section(m_inj, m_top)

    # 转换为 GeV²
    PB_TO_GEV2 = 1.0 / 0.3894e6
    sigma_hat_gev2 = sigma_hat * PB_TO_GEV2

    # 部分子光度
    tau = m_inj**2 / sqrt_s**2
    luminosity = pp_ttbar_luminosity(tau, m_top)

    # 微分截面
    dsigma_dM = (2.0 * m_inj / sqrt_s**2) * sigma_hat_gev2 * luminosity

    # 转回 pb/GeV
    dsigma_dM_pb = dsigma_dM * 0.3894e6

    return max(dsigma_dM_pb, 0.0)
