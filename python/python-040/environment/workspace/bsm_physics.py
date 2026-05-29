#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bsm_physics.py
超越标准模型（BSM）新物理核心理论模块

包含：
- Z' 玻色子 Breit-Wigner 共振传播子
- 有效场论（EFT）四费米子接触相互作用
- 暗物质-重媒介子-标准模型耦合的散射振幅
- 复杂幺正性约束与耦合常数边界
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional

# ---------------------------------------------------------------------------
# 物理常数（自然单位制 ℏ = c = 1）
# ---------------------------------------------------------------------------
ALPHA_EM = 1.0 / 137.035999084  # 精细结构常数
MZ_POLE = 91.1876  # GeV, Z^0 玻色子极点质量
GF_FERMI = 1.1663787e-5  # GeV^{-2}, 费米常数
VEV_HIGGS = 246.0  # GeV, 希格斯场真空期望值


@dataclass
class ZPrimeModel:
    """
    Z' 玻色子模型参数

    在 U(1)' 规范扩展模型中，Z' 是额外的有质量规范玻色子，
    通过规范混合与标准模型 Z^0 耦合。
    """
    mass: float          # Z' 质量 M_{Z'} [GeV]
    total_width: float   # 总衰变宽度 \Gamma_{Z'} [GeV]
    gq_coupling: float   # 与夸克的矢量耦合 g_q^V
    gl_coupling: float   # 与轻子的矢量耦合 g_\ell^V
    gq_axial: float = 0.0  # 与夸克的轴矢量耦合 g_q^A
    gl_axial: float = 0.0  # 与轻子的轴矢量耦合 g_\ell^A
    chi: float = 0.0     # 动能混合参数 \chi

    def __post_init__(self):
        assert self.mass > 0.0, "Z' 质量必须为正"
        assert self.total_width >= 0.0, "衰变宽度必须非负"
        # 幺正性边界: 树级别幺正性要求 g \lesssim 4\pi
        assert abs(self.gq_coupling) < 4.0 * np.pi, "夸克耦合超出微扰幺正性边界"
        assert abs(self.gl_coupling) < 4.0 * np.pi, "轻子耦合超出微扰幺正性边界"


def breit_wigner_propagator(s: float, model: ZPrimeModel) -> complex:
    """
    计算 Z' 传播子在曼德尔斯坦变量 s 下的值。

    传播子张量结构（在幺正规范下）：
        \hat{\Delta}_{\mu\nu}(q) = \frac{-g_{\mu\nu} + q_\mu q_\nu / M_{Z'}^2}
                                         {q^2 - M_{Z'}^2 + i M_{Z'} \Gamma_{Z'}}

    对于无质量外腿（e^+ e^- → ℓ^+ ℓ^-），q_μ q_ν 项被流守恒抑制，
    故仅保留标量部分：
        D(s) = 1 / (s - M_{Z'}^2 + i M_{Z'} \Gamma_{Z'})

    Parameters
    ----------
    s : float
        曼德尔斯坦变量 s = (p_1 + p_2)^2 [GeV^2]
    model : ZPrimeModel
        Z' 模型参数

    Returns
    -------
    complex
        复数传播子振幅
    """
    if s <= 0.0:
        return 0.0 + 0.0j
    m2 = model.mass ** 2
    # === HOLE 1: 补全 Breit-Wigner 传播子分母 ===
    # 提示: D(s) = 1 / (s - M^2 + i M Γ)
    denom = None  # TODO: 补全分母公式
    # =============================================
    # 避免在共振极点附近数值溢出
    if abs(denom) < 1e-12:
        denom = 1e-12 * (denom / abs(denom)) if denom != 0 else 1e-12
    return 1.0 / denom


def dilepton_cross_section(
    s: np.ndarray,
    cos_theta: np.ndarray,
    model: ZPrimeModel,
    include_sm: bool = True
) -> np.ndarray:
    """
    计算 e^+ e^- → ℓ^+ ℓ^- 过程的微分截面（Z' 共振区）。

    总振幅为 SM γ/Z^0 交换与 Z' 交换的相干叠加：
        \mathcal{M} = \mathcal{M}_\gamma + \mathcal{M}_{Z^0} + \mathcal{M}_{Z'}

    Z' 贡献的微分截面（极化单态求和后）：
        \frac{d\sigma_{Z'}}{d\Omega} = \frac{\alpha_{em}^2}{4s}
            \left[ A_{Z'}(1+\cos^2\theta) + B_{Z'}\cos\theta \right]
            |D(s)|^2 s^2

    其中系数 A_{Z'} 和 B_{Z'} 由矢量/轴矢量耦合决定：
        A_{Z'} = (g_e^{V2} + g_e^{A2})(g_\ell^{V2} + g_\ell^{A2})
        B_{Z'} = 4 g_e^V g_e^A g_\ell^V g_\ell^A

    为简化数值实现，使用有效耦合 g_eff = g_l_coupling * g_q_coupling。

    Parameters
    ----------
    s : np.ndarray
        质心能量平方 [GeV^2]，形状 (N,)
    cos_theta : np.ndarray
        散射角余弦，形状 (M,)
    model : ZPrimeModel
        Z' 模型参数
    include_sm : bool
        是否包含 SM 背景

    Returns
    -------
    np.ndarray
        微分截面 dσ/dΩ [pb]，形状 (N, M)
    """
    s = np.atleast_1d(s)
    cos_theta = np.atleast_1d(cos_theta)
    N, M = s.size, cos_theta.size

    # 转换到 (N, 1) 和 (1, M) 以便广播
    s_grid = s.reshape(N, 1)
    c_grid = cos_theta.reshape(1, M)

    # 传播子模方 |D(s)|^2
    d_vals = np.array([breit_wigner_propagator(si, model) for si in s.ravel()])
    d2 = np.abs(d_vals) ** 2
    d2 = d2.reshape(N, 1)

    # 有效耦合因子（矢量耦合主导近似）
    gv_e = model.gl_coupling
    gv_q = model.gq_coupling
    ga_e = model.gl_axial
    ga_q = model.gq_axial

    # === HOLE 2: 补全 Z' 微分截面耦合系数与截面公式 ===
    # 提示:
    #   A_{Z'} = (g_e^{V2} + g_e^{A2})(g_q^{V2} + g_q^{A2})
    #   B_{Z'} = 4 g_e^V g_e^A g_q^V g_q^A
    #   dσ/dΩ = α_em^2/(4s) * [A(1+cos²θ) + B cosθ] * |D(s)|² * s² * conv
    A_zp = None  # TODO: 补全 A 系数
    B_zp = None  # TODO: 补全 B 系数

    # 数值转换因子: GeV^{-2} → pb (1 pb = 2.56819e-9 GeV^{-2})
    GEV2_TO_PB = 0.389379e9  # 精确值: 1 GeV^{-2} = 0.389379 mb = 389.379 pb

    # Z' 贡献截面
    prefactor = ALPHA_EM ** 2 / (4.0 * s_grid)
    angular = None  # TODO: 补全角向依赖
    dsigma_zp = None  # TODO: 补全截面公式
    # ====================================================

    # 边界处理: 非物理区域置零
    dsigma_zp = np.where(s_grid > 0.0, dsigma_zp, 0.0)

    if include_sm:
        # Drell-Yan SM 背景（简化光子交换主导）
        # dσ_SM/dΩ ≈ α_em^2/(4s) * (1 + cos^2θ) * Q_e^2 Q_ℓ^2
        qe2 = (1.0 / 3.0) ** 2  # 下夸克电荷平方近似
        dsigma_sm = prefactor * (1.0 + c_grid ** 2) * qe2 * GEV2_TO_PB
        dsigma_sm = np.where(s_grid > 0.0, dsigma_sm, 0.0)
        return dsigma_zp + dsigma_sm

    return dsigma_zp


def eft_contact_interaction(
    s: float,
    eta_ll: float,
    eta_rr: float,
    eta_lr: float,
    Lambda: float
) -> float:
    """
    有效场论（EFT）四费米子接触相互作用对截面的修正。

    在质量远大于 √s 的重媒介子积分掉后，产生有效算符：
        \mathcal{L}_{\text{CI}} = \frac{2\pi}{\Lambda^2}
            \sum_{i,j=L,R} \eta_{ij}
            (\bar{\ell}_i \gamma^\mu \ell_i)(\bar{q}_j \gamma_\mu q_j)

    其中 Λ 为新物理能标，η_{ij} ∈ {-1, 0, +1} 为手征结构系数。

    对 e^+ e^- → ℓ^+ ℓ^- 截面的修正（干涉项 + 纯接触项）：
        \delta\sigma_{\text{CI}} = \frac{\pi \alpha_{em}^2}{2 \Lambda^4}
            \left[ (\eta_{LL}^2 + \eta_{RR}^2)(1 + \cos^2\theta)
                   + 2 \eta_{LR}^2 (1 - \cos^2\theta) \right] s

    Parameters
    ----------
    s : float
        质心能量平方 [GeV^2]
    eta_ll, eta_rr, eta_lr : float
        左手-左手、右手-右手、左手-右手结构系数
    Lambda : float
        有效 cutoff 能标 [GeV]

    Returns
    -------
    float
        接触相互作用对总截面的修正量 [pb]
    """
    if Lambda <= 0.0 or s <= 0.0:
        return 0.0

    # 幺正性约束: Λ > √(s) 时 EFT 有效
    if Lambda < np.sqrt(s):
        # 超出 EFT 有效范围，引入阻尼因子
        damping = (Lambda ** 2 / s) ** 2
    else:
        damping = 1.0

    GEV2_TO_PB = 0.389379e9
    prefactor = np.pi * ALPHA_EM ** 2 / (2.0 * Lambda ** 4) * s * GEV2_TO_PB

    # 对全立体角积分后的角向因子
    # ∫(1+cos²θ)dΩ = 16π/3, ∫(1-cos²θ)dΩ = 8π/3
    angular_integral = (16.0 * np.pi / 3.0) * (eta_ll ** 2 + eta_rr ** 2) \
                     + (8.0 * np.pi / 3.0) * (2.0 * eta_lr ** 2)

    return prefactor * angular_integral * damping


def decay_width_dilepton(model: ZPrimeModel) -> float:
    """
    计算 Z' → ℓ^+ ℓ^- 的部分衰变宽度。

    树级别衰变宽度公式（无质量轻子近似）：
        \Gamma(Z' → ℓ^+ ℓ^-) = \frac{M_{Z'}}{12\pi}
            \left[ (g_\ell^V)^2 + (g_\ell^A)^2 \right]

    包含 N_c = 3 色因子的夸克道：
        \Gamma(Z' → q\bar{q}) = \frac{N_c M_{Z'}}{12\pi}
            \left[ (g_q^V)^2 + (g_q^A)^2 \right]

    Parameters
    ----------
    model : ZPrimeModel
        Z' 模型参数

    Returns
    -------
    float
        单轻子道衰变宽度 [GeV]
    """
    mzp = model.mass
    if mzp <= 0.0:
        return 0.0
    glv2 = model.gl_coupling ** 2 + model.gl_axial ** 2
    return mzp * glv2 / (12.0 * np.pi)


def decay_width_hadronic(model: ZPrimeModel) -> float:
    """
    计算 Z' → q q̄ 的部分衰变宽度（对所有活跃夸克味求和）。
    """
    mzp = model.mass
    if mzp <= 0.0:
        return 0.0
    # 简化: 假设对所有6种夸克味（3代×2同位旋）开放
    n_f = 6  # 活跃夸克味数
    nc = 3   # 色数
    gqv2 = model.gq_coupling ** 2 + model.gq_axial ** 2
    return n_f * nc * mzp * gqv2 / (12.0 * np.pi)


def width_consistency_check(model: ZPrimeModel) -> bool:
    """
    检查输入的总宽度是否与微扰计算自洽。

    理论总宽度（忽略稀有衰变道）：
        \Gamma_{Z'}^{\text{th}} = \sum_f \Gamma(Z' → f\bar{f})

    若偏差 > 10%，发出数值警告但继续执行（允许暗区衰变通道）。
    """
    gamma_ll = 3.0 * decay_width_dilepton(model)   # 3 代轻子
    gamma_qq = decay_width_hadronic(model)
    gamma_nu = gamma_ll  # 中微子道近似等于轻子道（假设普适耦合）
    gamma_theory = gamma_ll + gamma_qq + gamma_nu

    if model.total_width <= 0.0:
        return True

    ratio = gamma_theory / model.total_width
    # 允许 10% 偏差以及暗区衰变修正
    return 0.5 <= ratio <= 2.0


def scattering_amplitude_matrix(
    s_vals: np.ndarray,
    model: ZPrimeModel
) -> np.ndarray:
    """
    构建复数散射振幅矩阵 A_{ij} = M(s_i, cosθ_j)。

    利用 c8lib 思想：复数矩阵运算处理传播子虚部。
    振幅矩阵用于后续的 SVD 分析。

    Parameters
    ----------
    s_vals : np.ndarray
        质心能量平方数组，形状 (N,)
    model : ZPrimeModel
        Z' 模型参数

    Returns
    -------
    np.ndarray
        复数振幅矩阵，形状 (N, M_cos)
    """
    s_vals = np.atleast_1d(s_vals)
    n_s = s_vals.size
    # 取9个cosθ点用于角分布分析
    cos_theta = np.linspace(-1.0, 1.0, 9)
    m_cos = cos_theta.size

    amp = np.zeros((n_s, m_cos), dtype=complex)

    gv_e = model.gl_coupling
    gv_q = model.gq_coupling
    ga_e = model.gl_axial
    ga_q = model.gq_axial

    for i, s in enumerate(s_vals):
        if s <= 0.0:
            continue
        # 传播子虚部
        d = breit_wigner_propagator(s, model)
        for j, ct in enumerate(cos_theta):
            # 矢量-矢量振幅
            amp_vv = gv_e * gv_q * (1.0 + ct ** 2) * s * d
            # 轴矢量-轴矢量振幅
            amp_aa = ga_e * ga_q * 2.0 * ct * s * d
            # 相干叠加
            amp[i, j] = ALPHA_EM * (amp_vv + amp_aa)

    return amp


def chi_square_signal(
    observed: np.ndarray,
    expected_bkg: np.ndarray,
    expected_sig: np.ndarray,
    uncertainties: np.ndarray
) -> float:
    """
    计算信号假设下的 χ² 统计量。

        χ² = \sum_i \frac{(n_i^{obs} - n_i^{bkg} - n_i^{sig})^2}
                         {\sigma_i^2 + n_i^{sig}}

    分母中额外加入 n_i^{sig} 以近似泊松统计。

    Parameters
    ----------
    observed : np.ndarray
        观测计数
    expected_bkg : np.ndarray
        背景预期
    expected_sig : np.ndarray
        信号预期
    uncertainties : np.ndarray
        系统误差

    Returns
    -------
    float
        χ² 值
    """
    residuals = observed - expected_bkg - expected_sig
    denom = uncertainties ** 2 + np.abs(expected_sig)
    denom = np.where(denom > 1e-12, denom, 1e-12)
    return float(np.sum(residuals ** 2 / denom))


def exclusion_limit_at_95cl(
    signal_yield: float,
    background_yield: float,
    luminosity: float,
    systematic_unc: float = 0.1
) -> float:
    """
    基于 CL_s 方法计算 95% CL 排除限。

    简化的 Asimov 近似：
        q_\mu = \frac{(s + b - b)^2}{b + (\epsilon b)^2}
              = \frac{s^2}{b + (\epsilon b)^2}

    95% CL 对应 q_μ = 1.96² ≈ 3.84（单边检验）。
    解得最大可容忍信号产额 s_max：
        s_{95} = 1.96 \sqrt{b + (\epsilon b)^2}

    然后转换为截面限：
        σ_{95} = s_{95} / (\mathcal{L} \times \epsilon_{sel})

    这里简化为 σ_{95} = s_{95} / L（假设 ε_sel = 1）。

    Parameters
    ----------
    signal_yield : float
        信号产额预期
    background_yield : float
        背景产额预期
    luminosity : float
        积分光度 [fb^{-1}]
    systematic_unc : float
        相对系统误差

    Returns
    -------
    float
        95% CL 截面上限 [pb]
    """
    if luminosity <= 0.0:
        return np.inf
    b = max(background_yield, 0.0)
    sigma_b = systematic_unc * b
    # 使用 CL_s 修正的近似公式
    s_95 = 1.96 * np.sqrt(b + sigma_b ** 2)
    # 若信号已超 excluded，返回当前截面
    if signal_yield > s_95 * 1.5:
        return signal_yield / luminosity
    return s_95 / luminosity
