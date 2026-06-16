"""
clinical_statistics_calphad.py
===============================
临床统计学方法在 CALPHAD 参数敏感性分析中的应用。

种子项目 1100_ling112211_Exercise-Prescription-System:
  - Welch t 检验
  - Clopper-Pearson 置信区间
  - Friedman/Wilcoxon 检验
  - 多重插补 (MICE)
  - Tipping-point 敏感性分析

种子项目 450_gamblers_ruin_simulation:
  - Monte Carlo 随机游走模拟
  - 经验统计量收集

映射到 CALPHAD:
  将不同 CALPHAD 参数集视为不同 "处理组",
  相平衡预测视为 "结果指标"。

  1. Welch t 检验: 比较两组参数集的预测是否有显著差异
  2. 置信区间: 估计相平衡成分的精确区间
  3. Wilcoxon 符号秩检验: 非参数检验参数扰动的影响
  4. Tipping-point 分析: 找参数变化的临界阈值

  随机游走模型 (gamblers_ruin 映射):
    参数在不确定域内随机游走,
    当预测偏离实验值超过阈值时 "破产"。
    统计 "存活时间" 的分布。
"""

import numpy as np
import math
from calphad_fec_constants import MC_RANDOM_SEED


def welch_t_test(sample1, sample2):
    """
    Welch t 检验 (不等方差 t 检验):

    H0: mu_1 = mu_2
    H1: mu_1 ≠ mu_2

    t = (mean_1 - mean_2) / sqrt(var_1/n_1 + var_2/n_2)
    df = (var_1/n_1 + var_2/n_2)^2 / [(var_1/n_1)^2/(n_1-1) + (var_2/n_2)^2/(n_2-1)]

    Parameters
    ----------
    sample1, sample2 : np.ndarray
        两组样本

    Returns
    -------
    dict
        {
            't_statistic': float,
            'df': float,
            'p_value_approx': float,
            'significant_005': bool,
        }
    """
    s1 = np.asarray(sample1, dtype=float)
    s2 = np.asarray(sample2, dtype=float)

    # 移除 NaN
    s1 = s1[~np.isnan(s1)]
    s2 = s2[~np.isnan(s2)]

    n1, n2 = len(s1), len(s2)
    if n1 < 2 or n2 < 2:
        return {
            't_statistic': 0.0, 'df': 0.0,
            'p_value_approx': 1.0, 'significant_005': False,
        }

    mean1, mean2 = np.mean(s1), np.mean(s2)
    var1, var2 = np.var(s1, ddof=1), np.var(s2, ddof=1)

    se = math.sqrt(var1 / n1 + var2 / n2) if (var1 / n1 + var2 / n2) > 0 else 1e-30
    t_stat = (mean1 - mean2) / se

    # Welch-Satterthwaite 自由度
    num = (var1 / n1 + var2 / n2) ** 2
    denom = (var1 / n1) ** 2 / (n1 - 1) + (var2 / n2) ** 2 / (n2 - 1)
    df = num / denom if denom > 0 else 1.0

    # 近似 p-value (使用正态近似, 大样本)
    # 对于 df > 30, t 分布接近正态
    z = abs(t_stat)
    p_value = 2.0 * _normal_sf(z)

    return {
        't_statistic': float(t_stat),
        'df': float(df),
        'p_value_approx': float(p_value),
        'significant_005': p_value < 0.05,
    }


def _normal_sf(z):
    """标准正态分布的生存函数 (上尾概率) 近似。"""
    # Abramowitz & Stegun 近似
    if z > 8.0:
        return 0.0
    p = 0.2316419
    b1, b2, b3, b4, b5 = 0.319381530, -0.356563782, 1.781477937, -1.821255978, 1.330274429
    t = 1.0 / (1.0 + p * abs(z))
    phi = math.exp(-z * z / 2.0) / math.sqrt(2.0 * math.pi)
    return phi * t * (b1 + t * (b2 + t * (b3 + t * (b4 + t * b5))))


def clopper_pearson_ci(k, n, alpha=0.05):
    """
    Clopper-Pearson 精确二项置信区间:

    对于 k 次成功 / n 次试验,
    成功概率 p 的 (1-alpha) 精确置信区间:

    p_lower = Beta(alpha/2; k, n-k+1)
    p_upper = Beta(1-alpha/2; k+1, n-k)

    Parameters
    ----------
    k : int
        成功次数
    n : int
        总试验次数
    alpha : float
        显著性水平

    Returns
    -------
    tuple
        (p_lower, p_upper)
    """
    if n == 0:
        return (0.0, 1.0)
    if k == 0:
        p_lower = 0.0
    else:
        p_lower = _beta_inv(alpha / 2, k, n - k + 1)
    if k == n:
        p_upper = 1.0
    else:
        p_upper = _beta_inv(1 - alpha / 2, k + 1, n - k)
    return (float(p_lower), float(p_upper))


def _beta_inv(p, a, b):
    """
    Beta 分布分位数的近似 (Newton 法)。
    """
    if p <= 0:
        return 0.0
    if p >= 1:
        return 1.0

    # 初始猜测: 使用均值
    x = a / (a + b)
    x = np.clip(x, 0.01, 0.99)

    for _ in range(50):
        cdf = _beta_cdf(x, a, b)
        pdf = _beta_pdf(x, a, b)
        if pdf < 1e-30:
            break
        x_new = x - (cdf - p) / pdf
        x = np.clip(x_new, 1e-10, 1.0 - 1e-10)
        if abs(cdf - p) < 1e-10:
            break

    return x


def _beta_cdf(x, a, b):
    """Beta CDF 的数值积分近似。"""
    n_quad = 100
    t = np.linspace(0, x, n_quad)
    dt = t[1] - t[0] if n_quad > 1 else x
    # 归一化常数
    from math import lgamma
    ln_B = lgamma(a) + lgamma(b) - lgamma(a + b)
    integrand = np.where(
        (t > 0) & (t < 1),
        np.exp((a - 1) * np.log(np.maximum(t, 1e-30))
               + (b - 1) * np.log(np.maximum(1 - t, 1e-30))
               - ln_B),
        0.0
    )
    return float(np.sum(integrand) * dt)


def _beta_pdf(x, a, b):
    """Beta PDF。"""
    if x <= 0 or x >= 1:
        return 0.0
    from math import lgamma
    ln_B = lgamma(a) + lgamma(b) - lgamma(a + b)
    return math.exp((a - 1) * math.log(x) + (b - 1) * math.log(1 - x) - ln_B)


def wilcoxon_signed_rank_test(differences):
    """
    Wilcoxon 符号秩检验:

    检验差值的中位数是否为零 (非参数检验)。

    1. 计算 |d_i|
    2. 排序, 分配秩
    3. W+ = sum of ranks where d_i > 0
    4. W- = sum of ranks where d_i < 0
    5. W = min(W+, W-)

    Parameters
    ----------
    differences : np.ndarray
        成对差值

    Returns
    -------
    dict
        {
            'W_plus': float,
            'W_minus': float,
            'W_statistic': float,
            'n_nonzero': int,
            'z_approx': float,
            'p_value_approx': float,
        }
    """
    d = np.asarray(differences, dtype=float)
    d = d[~np.isnan(d)]
    d_nz = d[np.abs(d) > 1e-15]
    n = len(d_nz)

    if n < 5:
        return {
            'W_plus': 0.0, 'W_minus': 0.0, 'W_statistic': 0.0,
            'n_nonzero': n, 'z_approx': 0.0, 'p_value_approx': 1.0,
        }

    abs_d = np.abs(d_nz)
    ranks = np.argsort(np.argsort(abs_d)) + 1  # 1-based ranks

    W_plus = float(np.sum(ranks[d_nz > 0]))
    W_minus = float(np.sum(ranks[d_nz < 0]))
    W = min(W_plus, W_minus)

    # 正态近似
    mu_W = n * (n + 1) / 4.0
    sigma_W = math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    z = (W - mu_W) / sigma_W if sigma_W > 0 else 0.0
    p_value = 2.0 * _normal_sf(abs(z))

    return {
        'W_plus': W_plus,
        'W_minus': W_minus,
        'W_statistic': W,
        'n_nonzero': n,
        'z_approx': float(z),
        'p_value_approx': float(p_value),
    }


def tipping_point_analysis(T, phase_a, phase_b,
                           param_index=0, perturbation_range=0.2,
                           n_steps=20):
    """
    Tipping-point 敏感性分析:

    逐步增大第 param_index 个参数的扰动幅度,
    直到相平衡预测发生质变 (不收敛或成分跳变)。

    找临界扰动幅度 delta_critical。

    Parameters
    ----------
    T : float
        温度 (K)
    phase_a : str
        alpha 相
    phase_b : str
        beta 相
    param_index : int
        参数索引 (0-5)
    perturbation_range : float
        最大相对扰动
    n_steps : int
        扫描步数

    Returns
    -------
    dict
        {
            'perturbations': np.ndarray,
            'x_alpha_values': np.ndarray,
            'convergence_status': list,
            'tipping_point': float or None,
        }
    """
    from newton_maehly_equilibrium import newton_maehly_solve
    import calphad_fec_constants as const
    from calphad_fec_constants import (
        L_FCC_FE_C, L_BCC_FE_C, L_LIQUID_FE_C,
    )

    L_all = list(L_FCC_FE_C) + list(L_BCC_FE_C) + list(L_LIQUID_FE_C)
    n_params = len(L_all)

    if param_index >= n_params:
        param_index = 0

    L_orig = L_all[:]
    perturbations = np.linspace(0, perturbation_range, n_steps)
    x_alpha_vals = np.zeros(n_steps)
    conv_status = []

    baseline_result = newton_maehly_solve(T, phase_a, phase_b)
    if not baseline_result['converged']:
        return {
            'perturbations': perturbations,
            'x_alpha_values': x_alpha_vals,
            'convergence_status': conv_status,
            'tipping_point': None,
        }

    baseline_xa = baseline_result['x_C_alpha']
    tipping_point = None

    for i, delta in enumerate(perturbations):
        L_pert = L_orig[:]
        L_pert[param_index] *= (1.0 + delta)

        # 设置参数
        const.L_FCC_FE_C = L_pert[:3]
        const.L_BCC_FE_C = L_pert[3:5]
        const.L_LIQUID_FE_C = L_pert[5:6] if len(L_pert) > 5 else list(L_LIQUID_FE_C)

        try:
            res = newton_maehly_solve(T, phase_a, phase_b)
            if res['converged']:
                x_alpha_vals[i] = res['x_C_alpha']
                conv_status.append('converged')
                # 检查是否发生跳变
                if tipping_point is None and abs(res['x_C_alpha'] - baseline_xa) > 0.01:
                    tipping_point = float(delta)
            else:
                x_alpha_vals[i] = np.nan
                conv_status.append('not_converged')
                if tipping_point is None:
                    tipping_point = float(delta)
        except Exception:
            x_alpha_vals[i] = np.nan
            conv_status.append('error')
            if tipping_point is None:
                tipping_point = float(delta)

    # 恢复参数
    const.L_FCC_FE_C = L_orig[:3]
    const.L_BCC_FE_C = L_orig[3:5]
    const.L_LIQUID_FE_C = L_orig[5:6] if len(L_orig) > 5 else list(L_LIQUID_FE_C)

    return {
        'perturbations': perturbations,
        'x_alpha_values': x_alpha_vals,
        'convergence_status': conv_status,
        'tipping_point': tipping_point,
    }


def gamblers_ruin_parameter_survival(T, phase_a, phase_b,
                                     initial_stake=0.05,
                                     n_games=200, max_steps=100):
    """
    赌徒破产模型映射: CALPHAD 参数 "存活" 分析。

    类比:
    - 赌徒的 "赌资" = 参数预测的置信度
    - 每局赌注 = 一次参数扰动
    - 赢/输 = 预测是否在实验范围内
    - 破产 = 预测偏离超过阈值

    统计:
    - 平均存活步数 (参数稳健性指标)
    - 破产概率

    Parameters
    ----------
    T : float
        温度 (K)
    phase_a : str
        alpha 相
    phase_b : str
        beta 相
    initial_stake : float
        初始 "赌资" (置信度)
    n_games : int
        模拟局数
    max_steps : int
        每局最大步数

    Returns
    -------
    dict
        {
            'survival_steps': np.ndarray,
            'mean_survival': float,
            'ruin_probability': float,
            'final_stakes': np.ndarray,
        }
    """
    from newton_maehly_equilibrium import newton_maehly_solve
    import calphad_fec_constants as const
    from calphad_fec_constants import L_FCC_FE_C, L_BCC_FE_C

    rng = np.random.RandomState(MC_RANDOM_SEED)

    survival_steps = np.zeros(n_games)
    final_stakes = np.zeros(n_games)

    L_orig_fcc = const.L_FCC_FE_C[:]
    L_orig_bcc = const.L_BCC_FE_C[:]

    baseline = newton_maehly_solve(T, phase_a, phase_b)
    if not baseline['converged']:
        return {
            'survival_steps': survival_steps,
            'mean_survival': 0.0,
            'ruin_probability': 1.0,
            'final_stakes': final_stakes,
        }

    baseline_xa = baseline['x_C_alpha']

    for game in range(n_games):
        stake = initial_stake

        for step in range(max_steps):
            if stake <= 0.001:
                # 破产
                survival_steps[game] = step
                final_stakes[game] = stake
                break

            # 随机扰动参数
            perturbation = rng.randn() * 0.01
            L_pert_fcc = [L_orig_fcc[j] * (1.0 + perturbation) for j in range(3)]
            const.L_FCC_FE_C = L_pert_fcc

            try:
                res = newton_maehly_solve(T, phase_a, phase_b)
                if res['converged']:
                    deviation = abs(res['x_C_alpha'] - baseline_xa)
                    if deviation < 0.01:
                        stake += 0.005  # 赢
                    else:
                        stake -= 0.01  # 输
                else:
                    stake -= 0.02  # 大输
            except Exception:
                stake -= 0.02

            stake = max(stake, 0.0)

        else:
            # 达到最大步数
            survival_steps[game] = max_steps
            final_stakes[game] = stake

        # 恢复
        const.L_FCC_FE_C = L_orig_fcc

    ruin_count = np.sum(final_stakes < 0.001)

    return {
        'survival_steps': survival_steps,
        'mean_survival': float(np.mean(survival_steps)),
        'ruin_probability': float(ruin_count / n_games),
        'final_stakes': final_stakes,
    }
