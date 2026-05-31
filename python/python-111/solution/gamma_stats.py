"""
不完全 Gamma 函数统计分析与亚稳态停留时间模块
基于 asa147 核心算法：不完全 Gamma 函数的级数展开计算。

在蛋白质折叠中的应用：
- 亚稳态停留时间分布的统计建模（Gamma 分布）
- 聚类分析、RMSD 分布的卡方拟合优度检验
- Boltzmann 因子截断积分的解析计算
- 晶体学 B 因子分布分析
- Metropolis-Hastings 接受概率的广义计算

数学基础:
    不完全 Gamma 函数 (下不完全):
        γ(x, p) = ∫_0^x t^{p-1} e^{-t} dt
    
    归一化形式 (本模块采用):
        P(x, p) = γ(x, p) / Γ(p)
                 = (x^p e^{-x} / Γ(p+1)) * Σ_{n=0}^∞ x^n / [(p+1)(p+2)...(p+n)]
    
    级数初始化: c_0 = 1, value = 1
    迭代: c_n = c_{n-1} * x / (p + n)
          value = value + c_n
    终止条件: |c_n / value| < ε (如 1e-9)
    
    对数形式防溢出:
        arg = p*ln(x) - ln(Γ(p+1)) - x
        value = exp(arg) * series_sum
"""

import numpy as np
from scipy.special import gammaln
from typing import Tuple


def gammds(x: float, p: float, eps: float = 1e-9) -> Tuple[float, int]:
    """
    计算归一化下不完全 Gamma 函数 P(x, p) = γ(x, p) / Γ(p)。
    
    算法来源: Algorithm AS 147 (Chi Leung Lau, Applied Statistics, 1980)。
    
    Parameters
    ----------
    x : float
        积分上限，要求 x > 0。
    p : float
        形状参数，要求 p > 0。
    eps : float
        收敛容差。
    
    Returns
    -------
    value : float
        归一化不完全 Gamma 值，范围 [0, 1]。
    ifault : int
        错误标志 (0=正常, 1=非法输入, 2=下溢)。
    """
    if x <= 0.0 or p <= 0.0:
        return 0.0, 1
    
    # 使用对数防止溢出
    arg = p * np.log(x) - gammaln(p + 1.0) - x
    if arg < -100.0:
        # 下溢保护
        return 0.0, 2
    
    e = np.exp(arg)
    if e < 1e-37:
        return 0.0, 2
    
    c = 1.0
    series_sum = 1.0
    a = p
    
    while True:
        a += 1.0
        c *= x / a
        series_sum += c
        if c / series_sum < eps:
            break
        # 防止无限循环
        if a > p + 1e6:
            break
    
    value = e * series_sum
    # 数值保护
    value = min(max(value, 0.0), 1.0)
    return value, 0


def gamma_cdf(x: np.ndarray, shape: float, scale: float) -> np.ndarray:
    """
    Gamma 分布的累积分布函数 (CDF)。
    
    Gamma 分布 PDF:
        f(t; shape, scale) = t^{shape-1} * exp(-t/scale) / (scale^shape * Γ(shape))
    
    CDF:
        F(x) = P(x/scale, shape) = γ(x/scale, shape) / Γ(shape)
    
    Parameters
    ----------
    x : np.ndarray
        自变量值，要求 x >= 0。
    shape : float
        形状参数 α > 0。
    scale : float
        尺度参数 β > 0。
    
    Returns
    -------
    cdf : np.ndarray
        CDF 值，范围 [0, 1]。
    """
    if shape <= 0 or scale <= 0:
        raise ValueError("shape and scale must be positive")
    
    x_flat = np.atleast_1d(x)
    cdf = np.zeros_like(x_flat, dtype=float)
    for i, xi in enumerate(x_flat):
        if xi <= 0:
            cdf[i] = 0.0
        else:
            val, _ = gammds(xi / scale, shape)
            cdf[i] = val
    return cdf


def estimate_gamma_parameters(data: np.ndarray) -> Tuple[float, float]:
    """
    用最大似然估计 (MLE) 拟合 Gamma 分布参数。
    
    对于 Gamma(shape=α, scale=β) 分布:
        MLE 方程:
            ln(α) - ψ(α) = ln( mean(data) ) - mean( ln(data) )
            β = mean(data) / α
    
    其中 ψ 为 digamma 函数。第一个方程用 Newton-Raphson 求解。
    
    Parameters
    ----------
    data : np.ndarray
        正样本数据。
    
    Returns
    -------
    shape : float
        估计的形状参数 α。
    scale : float
        估计的尺度参数 β。
    """
    from scipy.special import digamma, polygamma
    
    data = np.array(data)
    if np.any(data <= 0):
        raise ValueError("All data points must be positive")
    
    mean_log = np.mean(np.log(data))
    log_mean = np.log(np.mean(data))
    s = log_mean - mean_log
    
    # 初始猜测
    alpha = 0.5 / s if s > 0 else 1.0
    
    # Newton-Raphson
    for _ in range(100):
        f = np.log(alpha) - digamma(alpha) - s
        fp = 1.0 / alpha - polygamma(1, alpha)
        if abs(fp) < 1e-14:
            break
        alpha_new = alpha - f / fp
        if alpha_new <= 0:
            alpha_new = alpha * 0.5
        if abs(alpha_new - alpha) < 1e-10:
            break
        alpha = alpha_new
    
    beta = np.mean(data) / alpha
    return float(alpha), float(beta)


def metastable_state_residence_time_distribution(residence_times: np.ndarray) -> dict:
    """
    分析亚稳态停留时间分布。
    
    在分子动力学模拟中，系统在不同亚稳态（如折叠态、未折叠态）之间跳转。
    停留时间（dwell time）常被建模为 Gamma 分布或指数分布。
    
    Parameters
    ----------
    residence_times : np.ndarray
        观测到的停留时间序列。
    
    Returns
    -------
    result : dict
        包含 Gamma 参数估计、均值、方差、半衰期。
    """
    data = np.array(residence_times)
    if len(data) == 0:
        return {}
    
    mean_t = float(np.mean(data))
    var_t = float(np.var(data))
    
    try:
        shape, scale = estimate_gamma_parameters(data)
    except Exception:
        shape = mean_t ** 2 / max(var_t, 1e-12)
        scale = mean_t / max(shape, 1e-12)
    
    half_life = float(scale * shape * (2 ** (1.0 / shape) - 1.0))
    
    return {
        "mean": mean_t,
        "variance": var_t,
        "gamma_shape": shape,
        "gamma_scale": scale,
        "half_life": half_life,
        "n_samples": len(data),
    }


def chi_square_pvalue(chi2_stat: float, dof: int) -> float:
    """
    计算卡方统计量的 p 值。
    
    p = 1 - P(chi2_stat/2, dof/2)
    
    其中 P 为归一化下不完全 Gamma 函数。
    
    Parameters
    ----------
    chi2_stat : float
        卡方统计量。
    dof : int
        自由度。
    
    Returns
    -------
    pvalue : float
        p 值，范围 [0, 1]。
    """
    if chi2_stat < 0 or dof <= 0:
        return 0.0
    val, _ = gammds(chi2_stat / 2.0, dof / 2.0)
    pvalue = 1.0 - val
    return float(max(0.0, min(1.0, pvalue)))
