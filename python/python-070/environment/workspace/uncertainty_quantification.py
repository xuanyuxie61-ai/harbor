"""
uncertainty_quantification.py
渔业资源评估不确定性量化模块

整合算法：
1. Hammersley 拟蒙特卡洛序列（低差异序列）
2. 经典蒙特卡洛积分

核心科学应用：
在渔业资源评估中，种群参数（内禀增长率 r、承载力 K、可捕系数 q）
往往存在观测不确定性。本模块通过 QMC 和 MC 方法量化这些不确定性
对最优捕捞策略和总生物量的影响。

数学基础：
1. Hammersley 序列：在 M 维空间生成低差异点集
   第 i 个点：
   r_1 = i / N
   r_{j+1} = van_der_corput(i, p_j), j=1,...,M-1
   其中 p_j 为第 j 个素数

2. 多维积分估计：
   I = \int_{[0,1]^M} f(x) dx ≈ (1/N) \sum_{i=1}^N f(x_i)
   QMC 的收敛速率 O((log N)^M / N) 优于 MC 的 O(1/\sqrt{N})

3. 渔业风险评估指标：
   - 生物量低于 BLIM 的概率 P(B < B_{lim})
   - 期望经济收益 E[\Pi]
   - 收益方差 Var[\Pi]
"""

import numpy as np
from utils import NumericalConfig


# 前 100 个素数表（用于 Hammersley 序列）
_PRIMES = np.array([
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29,
    31, 37, 41, 43, 47, 53, 59, 61, 67, 71,
    73, 79, 83, 89, 97, 101, 103, 107, 109, 113,
    127, 131, 137, 139, 149, 151, 157, 163, 167, 173,
    179, 181, 191, 193, 197, 199, 211, 223, 227, 229,
    233, 239, 241, 251, 257, 263, 269, 271, 277, 281,
    283, 293, 307, 311, 313, 317, 331, 337, 347, 349,
    353, 359, 367, 373, 379, 383, 389, 397, 401, 409,
    419, 421, 431, 433, 439, 443, 449, 457, 461, 463,
    467, 479, 487, 491, 499, 503, 509, 521, 523, 541
], dtype=int)


def van_der_corput(i, base):
    """
    计算 van der Corput 序列的第 i 项（基于 base）

    算法：将 i 表示为 base 进制数，然后反向读取小数位
    i = d_k d_{k-1} ... d_0 (base)
    r = d_0 / base + d_1 / base^2 + ... + d_k / base^{k+1}

    Parameters
    ----------
    i : int
        序列索引，i >= 0
    base : int
        基数，base >= 2

    Returns
    -------
    r : float
        van der Corput 序列值，范围 [0, 1)
    """
    if i < 0:
        raise ValueError("i must be non-negative")
    if base < 2:
        raise ValueError("base must be >= 2")

    r = 0.0
    inv_base = 1.0 / base
    factor = inv_base
    while i > 0:
        digit = i % base
        r += digit * factor
        factor *= inv_base
        i //= base
    return r


def hammersley_value(i, m, n):
    """
    计算 Hammersley 序列的第 i 个 M 维点

    Parameters
    ----------
    i : int
        序列索引
    m : int
        空间维数，1 <= m <= 100
    n : int
        第一分量使用的基数（通常为总样本数）

    Returns
    -------
    r : ndarray, shape (m,)
        M 维 Hammersley 点
    """
    if m < 1 or m > 100:
        raise ValueError("m must be between 1 and 100")
    if n < 1:
        raise ValueError("n must be >= 1")

    r = np.zeros(m, dtype=float)
    r[0] = (i % (n + 1)) / n if n > 0 else 0.0

    for j in range(1, m):
        base = _PRIMES[j - 1]
        r[j] = van_der_corput(i, base)

    return r


def hammersley_sequence(i1, i2, m, n):
    """
    生成 Hammersley 序列中从第 i1 到第 i2 个点的批量

    Parameters
    ----------
    i1, i2 : int
        起始和结束索引（包含）
    m : int
        空间维数
    n : int
        第一分量基数

    Returns
    -------
    points : ndarray, shape (l, m)
        点集矩阵，l = abs(i2 - i1) + 1
    """
    if i1 <= i2:
        step = 1
    else:
        step = -1

    l = abs(i2 - i1) + 1
    points = np.zeros((l, m), dtype=float)
    idx = 0
    for i in range(i1, i2 + step, step):
        points[idx, :] = hammersley_value(i, m, n)
        idx += 1

    return points


def monte_carlo_integral_1d(func, a, b, n_samples, method='mc'):
    """
    一维数值积分：支持经典 MC 和 Hammersley QMC

    估计公式：
        I = \int_a^b f(x) dx ≈ (b-a) * (1/N) \sum_{i=1}^N f(x_i)

    Parameters
    ----------
    func : callable
        被积函数 f(x)
    a, b : float
        积分区间
    n_samples : int
        采样点数
    method : str
        'mc' 为经典蒙特卡洛，'qmc' 为 Hammersley 拟蒙特卡洛

    Returns
    -------
    estimate : float
        积分估计值
    """
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")

    if method == 'mc':
        x = np.random.uniform(0.0, 1.0, size=n_samples)
    elif method == 'qmc':
        pts = hammersley_sequence(0, n_samples - 1, m=1, n=n_samples)
        x = pts[:, 0]
    else:
        raise ValueError("method must be 'mc' or 'qmc'")

    # 将 [0,1] 映射到 [a,b]
    x_mapped = a + (b - a) * x
    fx = func(x_mapped)
    return (b - a) * np.mean(fx)


def fishery_risk_assessment(r_dist, K_dist, q_dist, E_fixed, p, c, delta,
                            n_samples=5000, method='qmc'):
    """
    渔业资源风险评估：量化参数不确定性下的经济收益分布

    将参数不确定性空间视为三维 (r, K, q)，通过 QMC/MC 采样
    计算：
    - 期望利润 E[\Pi]
    - 利润标准差 Std[\Pi]
    - 生物量低于安全阈值 B_{lim} = 0.3 K_{mean} 的概率
    - 利润为负（经济亏损）的概率

    Parameters
    ----------
    r_dist : tuple (r_mean, r_std)
        r 的对数正态分布参数（先取对数再采样）
    K_dist : tuple (K_mean, K_std)
        K 的对数正态分布参数
    q_dist : tuple (q_mean, q_std)
        q 的对数正态分布参数
    E_fixed : float
        固定捕捞努力量
    p, c, delta : float
        经济参数
    n_samples : int
        采样数
    method : str
        'mc' 或 'qmc'

    Returns
    -------
    results : dict
        包含期望利润、标准差、风险概率等指标
    """
    r_mean, r_std = r_dist
    K_mean, K_std = K_dist
    q_mean, q_std = q_dist

    # 生成 [0,1]^3 的采样点
    if method == 'qmc':
        samples = hammersley_sequence(0, n_samples - 1, m=3, n=n_samples)
    else:
        samples = np.random.uniform(0.0, 1.0, size=(n_samples, 3))

    # 通过逆变换采样转换为对数正态分布
    # 对数正态参数转换
    def lognormal_params(mu, sigma):
        sigma_ln = np.sqrt(np.log(1.0 + (sigma / mu) ** 2))
        mu_ln = np.log(mu) - 0.5 * sigma_ln ** 2
        return mu_ln, sigma_ln

    mu_r, sig_r = lognormal_params(r_mean, r_std)
    mu_K, sig_K = lognormal_params(K_mean, K_std)
    mu_q, sig_q = lognormal_params(q_mean, q_std)

    # 逆正态 CDF（近似：Box-Muller 变换的逆不适用，这里用直接采样+变换）
    # 对数正态采样：若 Z ~ N(0,1)，则 X = exp(mu + sigma*Z)
    # 通过标准正态逆 CDF 将均匀变量映射
    def approx_norm_ppf(u):
        """Beasley-Springer-Moro 近似逆正态 CDF"""
        a1 = -3.969683028665376e+01
        a2 = 2.209460984245205e+02
        a3 = -2.759285104469687e+02
        a4 = 1.383577518672690e+02
        a5 = -3.066479806614716e+01
        a6 = 2.506628277459239e+00
        b1 = -5.447609879822406e+01
        b2 = 1.615858368580409e+02
        b3 = -1.556989798598866e+02
        b4 = 6.680131188771972e+01
        b5 = -1.328068155288572e+01
        c1 = -7.784894002430293e-03
        c2 = -3.223964580411365e-01
        c3 = -2.400758277161838e+00
        c4 = -2.549732539343734e+00
        c5 = 4.374664141464968e+00
        c6 = 2.938163982698783e+00
        d1 = 7.784695709041460e-03
        d2 = 3.224671290700398e-01
        d3 = 2.445134137142996e+00
        d4 = 3.754408661907416e+00
        p_low = 0.02425
        p_high = 1.0 - p_low

        u = np.asarray(u, dtype=float)
        # 严格限制在开区间 (0,1) 内，防止 log(0) 导致数值溢出
        u = np.clip(u, NumericalConfig.EPS, 1.0 - NumericalConfig.EPS)
        z = np.zeros_like(u)

        mask1 = u < p_low
        q_val = np.sqrt(-2.0 * np.log(u[mask1]))
        z[mask1] = (((((c1 * q_val + c2) * q_val + c3) * q_val + c4) * q_val + c5) * q_val + c6) / \
                   ((((d1 * q_val + d2) * q_val + d3) * q_val + d4) * q_val + 1.0)

        mask2 = (u >= p_low) & (u <= p_high)
        q_val = u[mask2] - 0.5
        r_val = q_val * q_val
        z[mask2] = (((((a1 * r_val + a2) * r_val + a3) * r_val + a4) * r_val + a5) * r_val + a6) * q_val / \
                   (((((b1 * r_val + b2) * r_val + b3) * r_val + b4) * r_val + b5) * r_val + 1.0)

        mask3 = u > p_high
        q_val = np.sqrt(-2.0 * np.log(1.0 - u[mask3]))
        z[mask3] = -(((((c1 * q_val + c2) * q_val + c3) * q_val + c4) * q_val + c5) * q_val + c6) / \
                    ((((d1 * q_val + d2) * q_val + d3) * q_val + d4) * q_val + 1.0)

        return z

    z_r = approx_norm_ppf(samples[:, 0])
    z_K = approx_norm_ppf(samples[:, 1])
    z_q = approx_norm_ppf(samples[:, 2])

    r_samples = np.exp(mu_r + sig_r * z_r)
    K_samples = np.exp(mu_K + sig_K * z_K)
    q_samples = np.exp(mu_q + sig_q * z_q)

    # 计算稳态生物量和利润
    profits = np.zeros(n_samples, dtype=float)
    biomasses = np.zeros(n_samples, dtype=float)
    B_lim = 0.3 * K_mean

    # HOLE 3: Implement the biomass and profit calculation loop
    # 对每一样本参数 (r, K, q) 计算：
    # 1. 稳态生物量 B = K * max(0, 1 - q * E_fixed / r)
    # 2. 单位时间利润 profit_rate = p * q * E_fixed * B - c * E_fixed
    # 3. 贴现因子 discount_factor = (1 - exp(-delta * 50)) / delta
    # 4. 总利润 profits[i] = profit_rate * discount_factor
    for i in range(n_samples):
        pass

    results = {
        'expected_profit': np.mean(profits),
        'std_profit': np.std(profits),
        'profit_cv': np.std(profits) / abs(np.mean(profits)) if abs(np.mean(profits)) > NumericalConfig.EPS else np.inf,
        'prob_biomass_below_limit': np.mean(biomasses < B_lim),
        'prob_negative_profit': np.mean(profits < 0),
        'expected_biomass': np.mean(biomasses),
        'biomass_percentile_5': np.percentile(biomasses, 5),
        'biomass_percentile_95': np.percentile(biomasses, 95),
    }

    return results
