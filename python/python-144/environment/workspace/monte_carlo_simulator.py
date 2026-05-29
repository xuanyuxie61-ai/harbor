"""
monte_carlo_simulator.py
蒙特卡洛模拟与Bootstrap重采样模块。

融入的原项目核心算法：
- 609_jai_alai_simulation: 蒙特卡洛赛事模拟与统计计数
- 181_circle_monte_carlo: 单位圆上的随机采样

科学背景：
投资组合的风险评估依赖于对资产收益率分布尾部行为的准确刻画。
蒙特卡洛方法通过大量随机路径模拟，能够估计极端损失概率；
Bootstrap方法通过对历史数据重采样，提供不依赖分布假设的统计推断。
"""

import numpy as np
from scipy.linalg import cholesky


def simulate_returns_mc(mu: np.ndarray, sigma: np.ndarray, corr: np.ndarray,
                        T: int = 252, n_paths: int = 5000,
                        rng: np.random.Generator = None) -> np.ndarray:
    """
    基于几何布朗运动（GBM）与相关性结构模拟资产收益率路径。

    随机微分方程：
        dS_t / S_t = μ dt + Σ^{1/2} dW_t,
    其中 W_t 为标准布朗运动，Σ 为协方差矩阵。

    离散化（Euler-Maruyama）：
        r_{t+Δt} = μ Δt + Σ^{1/2} √Δt Z_t,
    其中 Z_t ~ N(0, I)。

    参数
    ----------
    mu : np.ndarray, shape (n,)
        年化预期收益率。
    sigma : np.ndarray, shape (n,)
        年化波动率。
    corr : np.ndarray, shape (n, n)
        相关性矩阵。
    T : int
        模拟时间步数（交易日）。
    n_paths : int
        模拟路径数。
    rng : np.random.Generator
        随机数生成器。

    返回
    -------
    np.ndarray, shape (n_paths, T, n)
        模拟的收益率路径。
    """
    if rng is None:
        rng = np.random.default_rng()
    n = len(mu)
    if sigma.shape != (n,) or corr.shape != (n, n):
        raise ValueError("simulate_returns_mc: 参数维度不匹配。")
    # 构建协方差矩阵
    cov = np.outer(sigma, sigma) * corr
    # Cholesky 分解
    try:
        L = cholesky(cov, lower=True)
    except np.linalg.LinAlgError:
        # 若相关矩阵不正定，进行正则化
        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvals = np.maximum(eigvals, 1e-8)
        cov = eigvecs @ np.diag(eigvals) @ eigvecs.T
        L = cholesky(cov, lower=True)

    dt = 1.0 / 252.0  # 日度时间步长
    Z = rng.standard_normal((n_paths, T, n))
    returns = np.zeros((n_paths, T, n))
    for t in range(T):
        shocks = Z[:, t, :] @ L.T
        returns[:, t, :] = mu * dt + shocks * np.sqrt(dt)
    return returns


def bootstrap_risk_analysis(returns: np.ndarray, n_bootstrap: int = 2000,
                            alpha: float = 0.05,
                            rng: np.random.Generator = None) -> dict:
    """
    对历史收益率进行Bootstrap重采样，估计风险统计量的置信区间。

    Bootstrap 原理（Efron, 1979）：
    设原始样本为 {r_i}_{i=1}^N，从中独立有放回地抽取 N 个样本构成
    Bootstrap 样本 r^{*b}。重复 B 次，得到统计量 θ 的经验分布，
    从而构造置信区间。

    参数
    ----------
    returns : np.ndarray, shape (T, n)
        历史收益率矩阵。
    n_bootstrap : int
        Bootstrap 次数。
    alpha : float
        显著性水平。
    rng : np.random.Generator
        随机数生成器。

    返回
    -------
    dict
        包含 Bootstrap 样本的均值、VaR、CVaR 的均值与置信区间。
    """
    if rng is None:
        rng = np.random.default_rng()
    T, n = returns.shape
    if T < 30:
        raise ValueError("bootstrap_risk_analysis: 样本量至少为30。")

    mean_boot = np.zeros((n_bootstrap, n))
    var_boot = np.zeros(n_bootstrap)
    cvar_boot = np.zeros(n_bootstrap)

    for b in range(n_bootstrap):
        idx = rng.integers(0, T, size=T)
        sample = returns[idx, :]
        mean_boot[b, :] = np.mean(sample, axis=0)
        # 等权重组合的收益率
        port_ret = np.mean(sample, axis=1)
        var_boot[b] = np.percentile(port_ret, alpha * 100)
        cvar_boot[b] = np.mean(port_ret[port_ret <= var_boot[b]])

    def ci(arr):
        return (float(np.percentile(arr, alpha / 2 * 100)),
                float(np.percentile(arr, (1 - alpha / 2) * 100)))

    return {
        "mean_estimate": np.mean(mean_boot, axis=0).tolist(),
        "mean_ci": [ci(mean_boot[:, i]) for i in range(n)],
        "VaR_mean": float(np.mean(var_boot)),
        "VaR_ci": ci(var_boot),
        "CVaR_mean": float(np.mean(cvar_boot)),
        "CVaR_ci": ci(cvar_boot),
        "n_bootstrap": n_bootstrap,
    }


def tournament_risk_simulation(strengths: np.ndarray, n_games: int = 10000,
                                rng: np.random.Generator = None) -> np.ndarray:
    """
    基于 jai_alai 锦标赛模拟思想，模拟资产间的"竞争"与"淘汰"过程。

    模型解释：
    将 n 个资产视为 n 个竞争者，每个资产具有相对"强度" strength_i。
    在每一轮中，资产按强度排序，强度最高的资产以概率
        p_i = strength_i / Σ_j strength_j
    获得胜利（表现最好）。
    经过 n_games 轮模拟，统计每个资产的获胜次数，归一化后作为
    相对表现概率估计。

    该模型可用于评估不同资产在市场竞争环境中的生存概率，
    进而为投资组合的尾部风险提供情景分析。

    参数
    ----------
    strengths : np.ndarray
        各资产的相对强度（正数）。
    n_games : int
        模拟轮数。
    rng : np.random.Generator
        随机数生成器。

    返回
    -------
    np.ndarray
        各资产的获胜频率（和为1）。
    """
    if rng is None:
        rng = np.random.default_rng()
    strengths = np.asarray(strengths, dtype=float)
    if np.any(strengths <= 0):
        raise ValueError("tournament_risk_simulation: 强度必须为正数。")
    n = len(strengths)
    stats = np.zeros(n, dtype=int)
    probs = strengths / np.sum(strengths)
    for _ in range(n_games):
        winner = rng.choice(n, p=probs)
        stats[winner] += 1
    return stats / n_games


def high_dim_sphere_sampling(n_samples: int, dim: int,
                              rng: np.random.Generator = None) -> np.ndarray:
    """
    在高维单位球面 S^{dim-1} 上均匀采样。

    算法：
    生成标准正态向量 Z ~ N(0, I_dim)，单位化：
        U = Z / ||Z||。
    由正态分布的旋转不变性，U 在球面上均匀分布。
    """
    if rng is None:
        rng = np.random.default_rng()
    Z = rng.standard_normal((n_samples, dim))
    norms = np.linalg.norm(Z, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return Z / norms
