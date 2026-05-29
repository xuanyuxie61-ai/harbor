"""
statistical_analysis.py
希格斯信号发现的统计分析与显著性计算

基于 209_conte_deboor 项目中的数值方法重构:
  - bisect, muller: 根查找 (用于求解似然方程)
  - rgfls: 优化搜索

物理内容:
  信号强度参数 mu 的极大似然估计:
    L(mu, theta) = prod_i Poisson(n_i | mu*s_i(theta) + b_i(theta))
    
  检验统计量 (profile likelihood):
    q_mu = -2*ln(L(mu, theta_hat(mu)) / L(mu_hat, theta_hat))
    
  显著性 (渐近公式, Wilks 定理):
    Z = sqrt(q_0)  (对 mu=0 的零假设)
    
  这里实现简化的单分箱高斯近似:
    Z = N_signal / sqrt(N_background)
    
  以及更精确的基于似然比的方法。
"""
import numpy as np
from constants import TINY, M_HIGGS, GAMMA_H
from utils import bisection, muller_method, lu_factor_scaled, lu_solve, safe_divide

# ============================================================
# 1. 泊松似然函数
# ============================================================
def poisson_likelihood(n_obs, mu, s, b):
    """
    泊松似然 (对数)
    
    L(n | mu*s + b) = (mu*s+b)^n * exp(-(mu*s+b)) / n!
    ln L = n * ln(mu*s+b) - (mu*s+b) - ln(n!)
    
    参数:
        n_obs: 观测计数
        mu: 信号强度参数
        s: 预期信号计数
        b: 预期背景计数
    返回:
        对数似然值
    """
    lam = mu * s + b
    if lam < TINY:
        lam = TINY
    # 使用 Stirling 近似: ln(n!) ~ n*ln(n) - n + 0.5*ln(2*pi*n)
    log_fact = 0.0
    if n_obs > 0:
        log_fact = n_obs * np.log(n_obs) - n_obs + 0.5 * np.log(2.0 * np.pi * max(n_obs, 1.0))
    return n_obs * np.log(lam) - lam - log_fact


def profile_log_likelihood(n_obs_list, s_list, b_list, mu):
    """
    轮廓对数似然 (对多个分箱)
    
    ln L_profile(mu) = sum_i ln Poisson(n_i | mu*s_i + b_i)
    """
    total = 0.0
    for n, s, b in zip(n_obs_list, s_list, b_list):
        total += poisson_likelihood(n, mu, s, b)
    return total


# ============================================================
# 2. 信号强度 MLE (映射 bisect / muller)
# ============================================================
def solve_mu_mle(n_obs_list, s_list, b_list, mu_min=0.0, mu_max=10.0):
    """
    求解信号强度 mu 的极大似然估计
    
    似然方程:
      d/dmu ln L = sum_i (n_i - mu*s_i - b_i) * s_i / (mu*s_i + b_i) = 0
    
    使用二分法求解 (要求导数在 [mu_min, mu_max] 上变号)
    
    参数:
        n_obs_list, s_list, b_list: 各分箱的观测、信号、背景
        mu_min, mu_max: 搜索区间
    返回:
        mu_hat: MLE 估计
        info: 求解信息
    """
    def dlnL_dmu(mu):
        total = 0.0
        for n, s, b in zip(n_obs_list, s_list, b_list):
            lam = mu * s + b
            if lam < TINY:
                lam = TINY
            total += (n - lam) * s / lam
        return total
    
    # 检查符号
    f_min = dlnL_dmu(mu_min)
    f_max = dlnL_dmu(mu_max)
    
    if f_min * f_max > 0:
        # 导数同号，边界解
        if abs(f_min) < abs(f_max):
            return mu_min, {"status": "boundary", "reason": "derivative_same_sign"}
        return mu_max, {"status": "boundary", "reason": "derivative_same_sign"}
    
    mu_hat, info = bisection(dlnL_dmu, mu_min, mu_max, tol=1.0e-10, max_iter=200)
    if mu_hat is None:
        # 尝试 Muller 方法
        mu_hat, info = muller_method(dlnL_dmu, mu_min, (mu_min + mu_max) / 2.0, mu_max, tol=1.0e-10)
    
    if mu_hat is None or mu_hat < 0:
        mu_hat = 0.0
    
    return mu_hat, info


# ============================================================
# 3. 显著性计算
# ============================================================
def significance_simple(signal, background):
    """
    简单显著性 (高斯近似):
      Z = S / sqrt(B)
    
    边界处理:
      - B <= 0 时返回 0
      - S < 0 时返回负显著性
    """
    if background <= 0:
        return 0.0
    return signal / np.sqrt(background)


def significance_likelihood_ratio(n_obs_list, s_list, b_list, mu_test=0.0):
    """
    基于似然比的显著性
    
    检验统计量:
      q_mu = -2 * [ln L(mu_test) - ln L(mu_hat)]
    
    对零假设 mu=0:
      Z = sqrt(q_0)  (若 mu_hat > 0)
    
    参数:
        n_obs_list, s_list, b_list: 各分箱数据
        mu_test: 待检验的 mu 值 (通常为 0)
    返回:
        Z: 显著性 (标准差数)
        q_mu: 检验统计量
        mu_hat: MLE
    """
    mu_hat, _ = solve_mu_mle(n_obs_list, s_list, b_list)
    mu_hat = max(mu_hat, 0.0)
    
    lnL_test = profile_log_likelihood(n_obs_list, s_list, b_list, mu_test)
    lnL_hat = profile_log_likelihood(n_obs_list, s_list, b_list, mu_hat)
    
    q = -2.0 * (lnL_test - lnL_hat)
    q = max(q, 0.0)
    
    Z = np.sqrt(q)
    return Z, q, mu_hat


# ============================================================
# 4. 置信区间 (映射 bisection)
# ============================================================
def confidence_interval_mu(n_obs_list, s_list, b_list, cl=0.95):
    """
    使用似然比方法计算 mu 的置信区间
    
    对 95% CL:
      ln L(mu) = ln L(mu_hat) - 1.92  (大样本近似)
      
    求解方程: ln L(mu) - ln L(mu_hat) + 1.92 = 0
    
    参数:
        n_obs_list, s_list, b_list: 数据
        cl: 置信水平
    返回:
        mu_lower, mu_upper: 置信区间
    """
    mu_hat, _ = solve_mu_mle(n_obs_list, s_list, b_list)
    lnL_max = profile_log_likelihood(n_obs_list, s_list, b_list, mu_hat)
    
    # 对泊松近似，delta ln L = 1.92 对应 95% CL (1D)
    delta = 0.5 * 3.841  # chi2(1, 0.95) / 2
    
    def target(mu):
        return profile_log_likelihood(n_obs_list, s_list, b_list, mu) - lnL_max + delta
    
    # 下界
    mu_lower = 0.0
    if target(0.0) > 0:
        # 搜索下界
        try:
            mu_lower, _ = bisection(target, 0.0, mu_hat, tol=1.0e-8)
            if mu_lower is None:
                mu_lower = 0.0
        except Exception:
            mu_lower = 0.0
    
    # 上界
    mu_upper = mu_hat
    # 扩大搜索范围直到 target 变号
    scale = 2.0
    while scale < 1.0e6:
        mu_test = mu_hat + scale * max(mu_hat, 0.1)
        if target(mu_test) < 0:
            try:
                mu_upper, _ = bisection(target, mu_hat, mu_test, tol=1.0e-8)
                if mu_upper is None:
                    mu_upper = mu_test
            except Exception:
                mu_upper = mu_test
            break
        scale *= 2.0
    else:
        mu_upper = mu_hat + 1.0e6
    
    return mu_lower, mu_upper


# ============================================================
# 5. 系统误差传播 (矩阵方法)
# ============================================================
def covariance_matrix_from_systematics(syst_errors):
    """
    从系统误差构造协方差矩阵
    
    简单模型 (完全相关系统误差):
      V_{ij} = delta_i * delta_j + delta_stat_i^2 * delta_{ij}
    
    参数:
        syst_errors: 每个分箱的系统误差列表
    返回:
        cov: 协方差矩阵
    """
    n = len(syst_errors)
    cov = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            cov[i, j] = syst_errors[i] * syst_errors[j]
        cov[i, i] += syst_errors[i] ** 2  # 加上统计部分占位
    return cov


def significance_with_systematics(signal, background, syst_bkg_frac=0.1):
    """
    考虑背景系统误差的显著性
    
    修正公式:
      Z = S / sqrt(B + (f_syst * B)^2)
    
    参数:
        signal: 信号计数
        background: 背景计数
        syst_bkg_frac: 背景系统误差比例
    """
    if background <= 0:
        return 0.0
    denom = np.sqrt(background + (syst_bkg_frac * background) ** 2)
    return signal / denom


# ============================================================
# 6. 完整统计报告
# ============================================================
def full_statistical_report(mass_bins, n_obs, n_bkg, n_sig_expected):
    """
    生成完整的统计分析报告
    
    参数:
        mass_bins: 质量分箱中心
        n_obs: 观测计数数组
        n_bkg: 背景预期数组
        n_sig_expected: 信号预期数组 (mu=1)
    返回:
        dict: 统计结果
    """
    n_obs = np.asarray(n_obs, dtype=float)
    n_bkg = np.asarray(n_bkg, dtype=float)
    n_sig = np.asarray(n_sig_expected, dtype=float)
    
    # 总信号和背景
    total_obs = np.sum(n_obs)
    total_bkg = np.sum(n_bkg)
    total_sig = np.sum(n_sig)
    
    # 简单显著性
    simple_Z = significance_simple(total_obs - total_bkg, total_bkg)
    
    # 似然比显著性
    lr_Z, q_mu, mu_hat = significance_likelihood_ratio(n_obs.tolist(), n_sig.tolist(), n_bkg.tolist())
    
    # 置信区间
    mu_lo, mu_hi = confidence_interval_mu(n_obs.tolist(), n_sig.tolist(), n_bkg.tolist())
    
    # 含系统误差
    Z_syst = significance_with_systematics(total_obs - total_bkg, total_bkg, syst_bkg_frac=0.15)
    
    return {
        "total_observed": total_obs,
        "total_background": total_bkg,
        "total_signal_expected": total_sig,
        "mu_hat": mu_hat,
        "mu_lower_95cl": mu_lo,
        "mu_upper_95cl": mu_hi,
        "significance_simple": simple_Z,
        "significance_likelihood": lr_Z,
        "significance_with_syst": Z_syst,
        "test_statistic_q0": q_mu,
        "mass_bins": mass_bins,
        "n_observed": n_obs,
        "n_background": n_bkg,
        "n_signal": n_sig,
    }
