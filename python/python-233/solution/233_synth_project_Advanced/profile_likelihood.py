"""
profile_likelihood.py — 轮廓似然与统计推断
==========================================
本模块实现顶夸克质量测量中的轮廓似然比 (PLR) 方法,
这是高能物理实验中最常用的统计推断框架。

统计模型:
  扩展似然函数:
    L(μ, θ) = Π_{i=1}^{N_bins} Poisson(n_i | μ s_i(θ) + b_i(θ)) ×
              Π_{k=1}^{N_syst} π_k(θ_k)

  其中:
    n_i: 第 i 个 bin 的观测事件数
    s_i(θ): 信号预测 (依赖于 m_top 和 nuisance θ)
    b_i(θ): 背景预测
    π_k(θ_k): nuisance 参数的约束 (通常取高斯)

  轮廓似然比:
    λ(m_t) = L(m_t, θ̂̂(m_t)) / L(m̂_t, θ̂)
    其中 θ̂̂(m_t) = argmax_θ L(m_t, θ) (profiled)
    (m̂_t, θ̂) = argmax_{m_t,θ} L (全局最优)

  检验统计量:
    q(m_t) = -2 ln λ(m_t)

  置信区间 (Neyman 构造):
    68% CL: q(m_t) < 1.0 (Δχ² = 1 for 1 DOF)
    95% CL: q(m_t) < 3.84

  参数Profile方法:
    对每个 m_t 假设, 通过迭代求解:
      ∂lnL/∂θ_k = 0 (对所有 k)
    得到 θ̂̂(m_t), 然后计算 ln L_profile(m_t)

  映射种子项目:
    - 300_disk01_integrands: 多维积分技术
    - 945_quad_trapezoid: 梯形积分规则
"""

import numpy as np


def poisson_log_likelihood(n_obs, expected):
    """
    计算 Poisson 对数似然。

    ln L = Σ_i [n_i ln(μ_i) - μ_i - ln(n_i!)]

    对于大 n, 使用 Stirling 近似:
      ln(n!) ≈ n ln(n) - n + 0.5 ln(2πn)

    参数:
        n_obs: 观测数数组
        expected: 期望值数组

    返回:
        ln L (标量)
    """
    n_obs = np.asarray(n_obs, dtype=float)
    expected = np.asarray(expected, dtype=float)

    # 边界保护
    expected = np.maximum(expected, 1e-10)

    from scipy.special import gammaln
    ll = np.sum(n_obs * np.log(expected) - expected - gammaln(n_obs + 1.0))

    return ll


def gaussian_constraint(theta, theta0, sigma):
    """
    高斯约束项 (nuisance parameter penalty)。

    ln π(θ) = -0.5 × ((θ - θ₀)/σ)²

    参数:
        theta: nuisance 参数值
        theta0: 先验中心值 (通常 = 0)
        sigma: 约束宽度 (系统误差)

    返回:
        ln π (标量)
    """
    if sigma <= 0:
        return 0.0
    return -0.5 * ((theta - theta0) / sigma)**2


def template_signal(m_inj_grid, m_top, luminosity=139.0):
    """
    生成信号模板 s_i(m_t)。

    在 LHC pp 碰撞中, tt̄ ne变质量谱的峰值位于 ~2m_t 附近,
    形状由以下因素决定:
      1. 部分子光度: 随 M 增加而快速下降
      2. 相空间: β = √(1-4m_t²/M²) 阈值行为
      3. 探测器分辨率: 高斯展宽

    简化参数化 (NNLO K-factor × PDF 光度):
      dσ/dM ∝ (1/M²) × β × L_gg(M²/s) × K_NNLO
    其中:
      β = √(1 - 4m_t²/M²) (相空间因子)
      L_gg(τ) = τ × g(√τ) × g(√τ) (胶子光度)
      K_NNLO ≈ 1.5 (高阶修正因子)

    参数:
        m_inj_grid: 质量 bin 边界 (GeV)
        m_top: 顶夸克质量假设
        luminosity: 积分光度 (fb⁻¹)

    返回:
        signal_template: 每个 bin 的信号事件数
    """
    from topmass_constants import SQRT_S_LHC
    from detector_response import detector_resolution_function
    from scipy.special import erf

    n_bins = len(m_inj_grid) - 1
    signal = np.zeros(n_bins)
    K_NNLO = 1.5  # NNLO K-factor
    sigma_total_ttbar = 832.0  # pb (NNLO tt̄ cross section at 13 TeV)

    # 归一化常数: 使得总信号 ≈ σ × L × ε
    total_efficiency = 0.15  # 典型半轻道效率

    for i in range(n_bins):
        M_low = m_inj_grid[i]
        M_high = m_inj_grid[i + 1]
        M_center = 0.5 * (M_low + M_high)

        if M_center <= 2.0 * m_top:
            signal[i] = 0.0
            continue

        # 相空间因子
        beta = np.sqrt(1.0 - 4.0 * m_top**2 / M_center**2)

        # 部分子光度 (简化: 指数下降)
        tau = M_center**2 / SQRT_S_LHC**2
        luminosity_pdf = np.exp(-8.0 * tau) * (1.0 - tau)**3

        # Breit-Wigner 型共振 (宽度由 Γ_t 和分辨率决定)
        sigma_res = detector_resolution_function(M_center)
        gamma_effective = np.sqrt(max(1.3**2 + sigma_res**2, 1.0))

        # 信号形状 (归一化的 Breit-Wigner × 相空间 × 光度)
        dsigma_dM = (K_NNLO * beta * luminosity_pdf /
                     (M_center**2 * gamma_effective * np.pi / 2.0))
        dsigma_dM *= sigma_total_ttbar / 100.0  # 归一化

        # 探测器效率
        epsilon = total_efficiency * np.exp(-((M_center - 2 * m_top) / (3.0 * m_top))**2)
        epsilon = max(epsilon, 0.01)

        bin_width = M_high - M_low
        signal[i] = luminosity * 1000 * dsigma_dM * bin_width * epsilon

    # 归一化: 总信号 ≈ σ_tt̄ × L × ε
    total_signal_raw = np.sum(signal)
    target_total = sigma_total_ttbar * luminosity * total_efficiency
    if total_signal_raw > 0:
        signal *= target_total / total_signal_raw

    return np.maximum(signal, 0.0)


def template_background(m_inj_grid, bg_normalization=50.0, bg_slope=-0.01):
    """
    生成背景模板 b_i。

    参数化: b(M) = N_bg × exp(slope × (M - M₀))
    其中主要背景:
      - W+jets (峰值在 m_W)
      - QCD 多喷注 (指数下降)

    参数:
        m_inj_grid: 质量 bin 边界
        bg_normalization: 背景归一化
        bg_slope: 指数斜率

    返回:
        background_template: 每个 bin 的背景事件数
    """
    n_bins = len(m_inj_grid) - 1
    background = np.zeros(n_bins)

    M0 = 350.0  # 参考质量

    for i in range(n_bins):
        M_center = 0.5 * (m_inj_grid[i] + m_inj_grid[i + 1])
        bin_width = m_inj_grid[i + 1] - m_inj_grid[i]
        background[i] = bg_normalization * np.exp(bg_slope * (M_center - M0)) * bin_width

    return np.maximum(background, 0.0)


def profile_likelihood(m_top_hypothesis, m_inj_grid, n_observed,
                       bg_template, nuisance_params=None,
                       systematic_uncertainties=None):
    """
    计算给定 m_t 假设下的轮廓似然值。

    ln L_profile(m_t) = max_θ ln L(m_t, θ)

    使用迭代 profile:
    对每个 nuisance θ_k:
      θ_k ← θ_k + η × ∂lnL/∂θ_k
    重复直到收敛。

    参数:
        m_top_hypothesis: m_t 假设值
        m_inj_grid: 质量网格
        n_observed: 观测事件数
        bg_template: 背景模板
        nuisance_params: nuisance 参数 (JES, b-scale, etc.)
        systematic_uncertainties: 系统误差列表

    返回:
        (lnL_profile, nuisance_best_fit)
    """
    if nuisance_params is None:
        nuisance_params = np.zeros(3)  # [JES, b_energy_scale, luminosity_unc]
    if systematic_uncertainties is None:
        systematic_uncertainties = [0.01, 0.005, 0.02]  # 1%, 0.5%, 2%

    # 信号模板
    sig_template = template_signal(m_inj_grid, m_top_hypothesis)

    # 迭代 profile
    theta = nuisance_params.copy()
    n_nuisance = len(theta)
    learning_rate = 0.5

    for iteration in range(50):
        # 计算期望值 (含 nuisance 调制)
        expected = bg_template.copy()

        # JES 效应: 移动质量轴
        jes_shift = theta[0]  # Jet Energy Scale
        expected *= (1.0 + jes_shift * systematic_uncertainties[0])

        # b-energy scale
        b_scale = theta[1]
        expected *= (1.0 + b_scale * systematic_uncertainties[1])

        # 光度不确定性
        lumi_unc = theta[2]
        expected += sig_template * (1.0 + lumi_unc * systematic_uncertainties[2])

        expected = np.maximum(expected, 1e-10)

        # 对数似然
        lnL = poisson_log_likelihood(n_observed, expected)
        for k in range(n_nuisance):
            lnL += gaussian_constraint(theta[k], 0.0, 1.0)

        # 数值梯度
        grad = np.zeros(n_nuisance)
        h = 1e-4
        for k in range(n_nuisance):
            theta_p = theta.copy()
            theta_m = theta.copy()
            theta_p[k] += h
            theta_m[k] -= h

            exp_p = bg_template * (1.0 + theta_p[0] * systematic_uncertainties[0]) * \
                    (1.0 + theta_p[1] * systematic_uncertainties[1]) + \
                    sig_template * (1.0 + theta_p[2] * systematic_uncertainties[2])
            exp_m = bg_template * (1.0 + theta_m[0] * systematic_uncertainties[0]) * \
                    (1.0 + theta_m[1] * systematic_uncertainties[1]) + \
                    sig_template * (1.0 + theta_m[2] * systematic_uncertainties[2])

            exp_p = np.maximum(exp_p, 1e-10)
            exp_m = np.maximum(exp_m, 1e-10)

            lnL_p = poisson_log_likelihood(n_observed, exp_p) + \
                    gaussian_constraint(theta_p[k], 0.0, 1.0)
            lnL_m = poisson_log_likelihood(n_observed, exp_m) + \
                    gaussian_constraint(theta_m[k], 0.0, 1.0)

            grad[k] = (lnL_p - lnL_m) / (2.0 * h)

        # 梯度上升
        theta += learning_rate * grad

        # 收敛检查
        if np.max(np.abs(learning_rate * grad)) < 1e-6:
            break

    return lnL, theta


def scan_profile_likelihood(m_top_scan, m_inj_grid, n_observed, bg_template,
                            systematic_uncertainties=None):
    """
    扫描 m_t 假设, 计算轮廓似然曲线。

    参数:
        m_top_scan: m_t 假设值数组
        m_inj_grid: 质量网格
        n_observed: 观测数据
        bg_template: 背景模板
        systematic_uncertainties: 系统误差

    返回:
        (lnL_scan, m_best_fit, delta_lnL)
    """
    lnL_values = np.zeros(len(m_top_scan))

    for i, m_t in enumerate(m_top_scan):
        lnL, _ = profile_likelihood(
            m_t, m_inj_grid, n_observed, bg_template,
            systematic_uncertainties=systematic_uncertainties
        )
        lnL_values[i] = lnL

    # 最佳拟合
    best_idx = np.argmax(lnL_values)
    m_best = m_top_scan[best_idx]
    lnL_max = lnL_values[best_idx]

    # ΔlnL = lnL - lnL_max
    delta_lnL = lnL_values - lnL_max

    return lnL_values, m_best, delta_lnL


def confidence_interval(m_top_scan, delta_lnL, cl=0.68):
    """
    从 ΔlnL 曲线提取置信区间。

    对于 1 DOF:
      68% CL: ΔlnL > -0.5 (即 -2ΔlnL < 1)
      95% CL: ΔlnL > -1.92 (即 -2ΔlnL < 3.84)

    参数:
        m_top_scan: m_t 扫描值
        delta_lnL: ΔlnL 数组
        cl: 置信水平

    返回:
        (m_lower, m_upper, m_best)
    """
    if cl == 0.68:
        threshold = -0.5
    elif cl == 0.95:
        threshold = -1.92
    else:
        from scipy.stats import chi2
        threshold = -chi2.ppf(cl, 1) / 2.0

    best_idx = np.argmax(delta_lnL)
    m_best = m_top_scan[best_idx]

    # 左边界
    m_lower = m_top_scan[0]
    for i in range(best_idx, -1, -1):
        if delta_lnL[i] < threshold:
            # 线性插值
            if i < best_idx:
                frac = (threshold - delta_lnL[i]) / (delta_lnL[i + 1] - delta_lnL[i])
                m_lower = m_top_scan[i] + frac * (m_top_scan[i + 1] - m_top_scan[i])
            else:
                m_lower = m_top_scan[i]
            break

    # 右边界
    m_upper = m_top_scan[-1]
    for i in range(best_idx, len(m_top_scan)):
        if delta_lnL[i] < threshold:
            if i > 0:
                frac = (threshold - delta_lnL[i - 1]) / (delta_lnL[i] - delta_lnL[i - 1])
                m_upper = m_top_scan[i - 1] + frac * (m_top_scan[i] - m_top_scan[i - 1])
            else:
                m_upper = m_top_scan[i]
            break

    return m_lower, m_upper, m_best


def asymptotic_pvalue(q_observed, n_dof=1):
    """
    计算渐近 p-value。

    p = 1 - F_χ²(q; n_dof)

    对于发现显著性:
      Z = Φ⁻¹(1 - p) (标准差)

    参数:
        q_observed: 检验统计量
        n_dof: 自由度

    返回:
        (p_value, significance_sigma)
    """
    from scipy.stats import chi2
    from scipy.stats import norm

    p_value = 1.0 - chi2.cdf(q_observed, n_dof)
    significance = norm.ppf(1.0 - p_value) if p_value > 0 else 10.0

    return p_value, significance
