#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
event_selection.py
事例选择、统计推断与结果汇总模块

在BSM信号分析中用于:
- 不变质量重建与共振峰拟合
- CL_s 方法计算排除限
- 多变量分析（MVA）分数计算
- 最终物理结果输出
"""

import numpy as np
from typing import Tuple, Dict, List, Optional


def reconstruct_invariant_mass(
    pt1: float,
    eta1: float,
    phi1: float,
    pt2: float,
    eta2: float,
    phi2: float,
    mass1: float = 0.000511,  # electron mass in GeV
    mass2: float = 0.000511
) -> float:
    """
    从两个轻子的四动量重建不变质量。

    四动量分量（横向动量 p_T，赝快度 η，方位角 φ）：
        p_x = p_T cos φ
        p_y = p_T sin φ
        p_z = p_T sinh η
        E   = sqrt(p_T^2 cosh^2 η + m^2)

    不变质量:
        M^2 = (E_1 + E_2)^2 - (p_{x1} + p_{x2})^2
                           - (p_{y1} + p_{y2})^2
                           - (p_{z1} + p_{z2})^2

    Parameters
    ----------
    pt1, pt2 : float
        轻子横向动量 [GeV]
    eta1, eta2 : float
        赝快度
    phi1, phi2 : float
        方位角 [rad]
    mass1, mass2 : float
        轻子质量 [GeV]

    Returns
    -------
    float
        不变质量 [GeV]
    """
    px1 = pt1 * np.cos(phi1)
    py1 = pt1 * np.sin(phi1)
    pz1 = pt1 * np.sinh(eta1)
    e1 = np.sqrt(pt1 ** 2 * np.cosh(eta1) ** 2 + mass1 ** 2)

    px2 = pt2 * np.cos(phi2)
    py2 = pt2 * np.sin(phi2)
    pz2 = pt2 * np.sinh(eta2)
    e2 = np.sqrt(pt2 ** 2 * np.cosh(eta2) ** 2 + mass2 ** 2)

    m2 = (e1 + e2) ** 2 - (px1 + px2) ** 2 - (py1 + py2) ** 2 - (pz1 + pz2) ** 2
    m2 = max(m2, 0.0)
    return np.sqrt(m2)


def generate_drell_yan_background(
    n_events: int,
    mass_range: Tuple[float, float],
    seed: Optional[int] = None
) -> np.ndarray:
    """
    生成 Drell-Yan 过程的标准模型背景事例。

    Drell-Yan 截面随质量下降：
        dσ/dM ∝ 1/M^3 （s-channel 光子交换主导）

    使用逆变换采样生成质量分布：
        PDF(M) ∝ M^{-3}  for M ∈ [M_min, M_max]
        CDF(M) = (M^{-2} - M_max^{-2}) / (M_min^{-2} - M_max^{-2})

    Parameters
    ----------
    n_events : int
        事例数
    mass_range : Tuple[float, float]
        质量范围 [GeV]
    seed : int or None
        随机种子

    Returns
    -------
    np.ndarray
        背景事例的不变质量 [GeV]
    """
    if seed is not None:
        np.random.seed(seed)

    m_min, m_max = mass_range
    u = np.random.uniform(0.0, 1.0, n_events)

    # 逆 CDF 采样
    a = m_min ** (-2)
    b = m_max ** (-2)
    masses = np.sqrt(1.0 / (b + u * (a - b)))

    # 添加探测器分辨率展宽
    sigma_rel = 0.02  # 2% 质量分辨率
    noise = np.random.normal(0.0, sigma_rel * masses)
    masses += noise
    masses = np.maximum(masses, m_min)

    return masses


def generate_signal_events(
    n_events: int,
    zp_mass: float,
    zp_width: float,
    mass_range: Tuple[float, float],
    seed: Optional[int] = None
) -> np.ndarray:
    """
    生成 Z' → ℓ⁺ℓ⁻ 信号事例。

    不变质量服从 Breit-Wigner 分布：
        PDF(M) ∝ 1 / ((M^2 - M_{Z'}^2)^2 + M_{Z'}^2 Γ_{Z'}^2)

    使用接受-拒绝采样。

    Parameters
    ----------
    n_events : int
        信号事例数
    zp_mass : float
        Z' 质量 [GeV]
    zp_width : float
        Z' 衰变宽度 [GeV]
    mass_range : Tuple[float, float]
        质量范围
    seed : int or None
        随机种子

    Returns
    -------
    np.ndarray
        信号事例的不变质量 [GeV]
    """
    if seed is not None:
        np.random.seed(seed)

    m_min, m_max = mass_range
    masses = []
    max_attempts = n_events * 1000
    attempts = 0

    # BW 分布的峰值在 M ≈ M_Z'（窄宽度近似）
    m_peak = zp_mass
    peak_height = 1.0 / ((m_peak ** 2 - zp_mass ** 2) ** 2 + zp_mass ** 2 * zp_width ** 2)
    if peak_height < 1e-30:
        peak_height = 1e-30

    while len(masses) < n_events and attempts < max_attempts:
        m = np.random.uniform(m_min, m_max)
        bw = 1.0 / ((m ** 2 - zp_mass ** 2) ** 2 + zp_mass ** 2 * zp_width ** 2)
        u = np.random.uniform(0.0, peak_height)
        if u <= bw:
            # 添加探测器分辨率展宽
            sigma_m = 0.015 * m  # 1.5% 质量分辨率
            m_smeared = m + np.random.normal(0.0, sigma_m)
            if m_min <= m_smeared <= m_max:
                masses.append(m_smeared)
        attempts += 1

    return np.array(masses)


def cl_s_limit(
    n_observed: int,
    n_background: float,
    n_signal_hypothesis: float,
    confidence_level: float = 0.95
) -> bool:
    """
    基于 CL_s 方法的假设检验。

    CL_s = CL_{s+b} / CL_b

    使用泊松统计的简化版本：
        CL_b = P(n ≤ n_obs | b)
        CL_{s+b} = P(n ≤ n_obs | s + b)

    若 CL_s < 1 - CL，则排除该信号假设。

    Parameters
    ----------
    n_observed : int
        观测计数
    n_background : float
        背景预期
    n_signal_hypothesis : float
        信号假设产额
    confidence_level : float
        置信水平

    Returns
    -------
    bool
        True 表示该信号假设被排除
    """
    from math import exp, factorial

    def poisson_cdf(k: int, mu: float) -> float:
        """泊松分布的累积分布函数 P(N ≤ k)。"""
        if mu <= 0.0:
            return 1.0 if k >= 0 else 0.0
        cdf = 0.0
        for n in range(k + 1):
            # 使用对数避免溢出
            log_p = n * np.log(mu) - mu
            for i in range(1, n + 1):
                log_p -= np.log(i)
            cdf += np.exp(log_p)
        return min(cdf, 1.0)

    cl_b = poisson_cdf(n_observed, n_background)
    cl_sb = poisson_cdf(n_observed, n_background + n_signal_hypothesis)

    if cl_b < 1e-10:
        cl_s = 0.0
    else:
        cl_s = cl_sb / cl_b

    return cl_s < (1.0 - confidence_level)


def run_full_analysis(
    zp_model_params: Dict,
    luminosity_fb: float = 3000.0,
    n_bins: int = 50
) -> Dict:
    """
    运行完整的 Z' → ℓ⁺ℓ⁻ 信号分析流程。

    流程:
        1. 生成信号与背景事例
        2. 构建不变质量直方图
        3. 寻找共振峰
        4. 计算局部显著性
        5. 计算 95% CL 排除限
        6. 评估发现潜力

    Parameters
    ----------
    zp_model_params : Dict
        Z' 模型参数字典
    luminosity_fb : float
        积分光度 [fb^{-1}]
    n_bins : int
        直方图分箱数

    Returns
    -------
    Dict
        分析结果
    """
    m_zp = zp_model_params['mass']
    gamma_zp = zp_model_params['total_width']
    gq = zp_model_params.get('gq_coupling', 0.1)

    # 分析范围: M_Z' ± 10 Γ_Z'
    mass_min = max(m_zp - 15.0 * gamma_zp, 50.0)
    mass_max = m_zp + 15.0 * gamma_zp
    mass_range = (mass_min, mass_max)

    # === HOLE 3: 补全信号截面与预期产额计算 ===
    # 提示:
    #   - LHC Run 3 质心能量 √s = 13 TeV
    #   - 简化截面 σ(pp→Z')×BR(Z'→ℓℓ) 应与 bsm_physics 中的模型一致
    #   - 产额 N_sig = σ × L × ε × BR
    #   - 1 fb^{-1} = 1000 pb^{-1}
    sqrt_s = 13000.0  # LHC Run 3, 13 TeV
    s = sqrt_s ** 2
    sigma_zp = None  # TODO: 补全信号截面 [pb]
    br_ee = None      # TODO: 补全分支比

    # 预期信号产额
    n_sig_expected = None  # TODO: 补全预期信号产额计算
    # ===========================================

    # 背景截面（Drell-Yan）
    sigma_dy = 10.0 * (1000.0 / m_zp) ** 3  # 简化 [pb]
    n_bkg_expected = sigma_dy * luminosity_fb * 1000.0 * 0.5 * br_ee

    # 生成事例
    n_sig = int(np.round(n_sig_expected))
    n_bkg = int(np.round(n_bkg_expected))

    sig_masses = generate_signal_events(max(n_sig, 10), m_zp, gamma_zp, mass_range, seed=42)
    bkg_masses = generate_drell_yan_background(max(n_bkg, 100), mass_range, seed=43)

    # 构建直方图
    bins = np.linspace(mass_min, mass_max, n_bins + 1)
    sig_counts, _ = np.histogram(sig_masses, bins=bins)
    bkg_counts, _ = np.histogram(bkg_masses, bins=bins)
    obs_counts = sig_counts + bkg_counts

    bin_centers = (bins[:-1] + bins[1:]) / 2.0

    # 寻找共振峰
    try:
        from signal_processing import resonance_peak_finder
    except ImportError:
        from .signal_processing import resonance_peak_finder
    peak_mass, peak_height, peak_sig = resonance_peak_finder(
        bin_centers, obs_counts, window_width=3.0 * gamma_zp
    )

    # 95% CL 排除限（简化的截面限）
    n_obs_total = int(np.sum(obs_counts))
    n_bkg_total = max(float(np.sum(bkg_counts)), 1.0)
    sigma_95 = 1.96 * np.sqrt(n_bkg_total) / (luminosity_fb * 1000.0 * 0.5 * br_ee)

    # 发现潜力
    try:
        from parameter_scan import discovery_potential
    except ImportError:
        from .parameter_scan import discovery_potential
    z_values = discovery_potential(
        np.array([sigma_zp]),
        np.array([sigma_dy]),
        np.array([luminosity_fb]),
        np.array([0.05])
    )
    discovery_z = z_values[0]

    return {
        'zp_mass': m_zp,
        'zp_width': gamma_zp,
        'signal_yield_expected': n_sig_expected,
        'background_yield_expected': n_bkg_expected,
        'peak_mass': peak_mass,
        'peak_height': peak_height,
        'peak_significance': peak_sig,
        'exclusion_limit_sigma_95_pb': sigma_95,
        'discovery_potential_z': discovery_z,
        'bins': bin_centers,
        'signal_histogram': sig_counts,
        'background_histogram': bkg_counts,
        'observed_histogram': obs_counts,
    }


def format_physics_summary(results: Dict) -> str:
    """
    格式化物理分析结果为可读文本。
    """
    lines = [
        "=" * 70,
        "  LHC BSM Z' → ℓ⁺ℓ⁻ 信号分析结果汇总",
        "=" * 70,
        f"  Z' 质量:           {results['zp_mass']:.2f} GeV",
        f"  Z' 总宽度:         {results['zp_width']:.4f} GeV",
        f"  预期信号产额:      {results['signal_yield_expected']:.2f}",
        f"  预期背景产额:      {results['background_yield_expected']:.2f}",
        "-" * 70,
        "  共振峰搜索结果:",
        f"    峰位质量:        {results['peak_mass']:.2f} GeV",
        f"    峰高度:          {results['peak_height']:.2f}",
        f"    局部显著性:      {results['peak_significance']:.3f} σ",
        "-" * 70,
        "  统计推断:",
        f"    95% CL 截面上限: {results['exclusion_limit_sigma_95_pb']:.6f} pb",
        f"    发现潜力 (Z):    {results['discovery_potential_z']:.3f} σ",
        "=" * 70,
    ]
    return "\n".join(lines)
