"""
statistical_aggregator.py — Monte Carlo 统计量聚合

科学背景
========
从 N_mc 个 MC 实现中, 逐点计算:
1. 样本均值:  μ̂(x) = (1/N) Σ_m u_m(x)
2. 样本方差:  σ̂²(x) = (1/(N-1)) Σ_m (u_m(x) - μ̂(x))²
3. 样本标准差:  σ̂(x) = √σ̂²(x)
4. 标准误差:  SE(x) = σ̂(x) / √N

以及 Bootstrap 统计量:
5. 极大统计量:  M_n = max_x |Z_n(x)|
6. Bootstrap CDF:  F̂_M(c) = (1/B) Σ_b 1{M_n^{(b)} ≤ c}

核心公式
========
1. t 分布临界值:  t_{α/2, ν}  其中 ν = N-1
2. 点态 CI:  μ̂ ± t_{α/2,ν} · SE
3. 点态 PI:  μ̂ ± t_{α/2,ν} · σ̂·√(1+1/N)
4. Bootstrap 方差:  Var_boot = (1/B) Σ_b (θ^{(b)} - θ̄)^2
"""

import numpy as np


def compute_mc_statistics(mc_solutions):
    """计算 MC 样本的基本统计量.

    参数
    ----
    mc_solutions : ndarray, shape (n_mc, nt, nx)

    返回
    ----
    stats : dict
        'mean'      : ndarray, shape (nt, nx)
        'variance'  : ndarray, shape (nt, nx)
        'std'       : ndarray, shape (nt, nx)
        'stderr'    : ndarray, shape (nt, nx)
        'median'    : ndarray, shape (nt, nx)
        'q05'       : ndarray, shape (nt, nx)  5% 分位数
        'q95'       : ndarray, shape (nt, nx)  95% 分位数
        'min'       : ndarray
        'max'       : ndarray
    """
    n_mc = mc_solutions.shape[0]

    stats = {
        'mean': np.mean(mc_solutions, axis=0),
        'variance': np.var(mc_solutions, axis=0, ddof=1) if n_mc > 1 else np.zeros_like(mc_solutions[0]),
        'std': np.std(mc_solutions, axis=0, ddof=1) if n_mc > 1 else np.zeros_like(mc_solutions[0]),
        'stderr': np.std(mc_solutions, axis=0, ddof=1) / np.sqrt(n_mc) if n_mc > 1 else np.zeros_like(mc_solutions[0]),
        'median': np.median(mc_solutions, axis=0),
        'q05': np.percentile(mc_solutions, 5, axis=0),
        'q95': np.percentile(mc_solutions, 95, axis=0),
        'min': np.min(mc_solutions, axis=0),
        'max': np.max(mc_solutions, axis=0),
    }
    return stats


def compute_standardized_residuals(mc_solutions, mean_field, std_field):
    """计算标准化残差场.

    Z_m(x) = (u_m(x) - μ̂(x)) / σ̂(x)

    参数
    ----
    mc_solutions : ndarray, shape (n_mc, nt, nx)
    mean_field : ndarray, shape (nt, nx)
    std_field : ndarray, shape (nt, nx)

    返回
    ----
    residuals : ndarray, shape (n_mc, nt, nx)
    """
    # 设定下限, 避免边界节点处除零
    positive_std = std_field[std_field > 0]
    floor_val = 1.0e-6
    if len(positive_std) > 0:
        floor_val = max(1.0e-6, np.mean(positive_std) * 0.01)
    std_safe = np.maximum(std_field, floor_val)
    residuals = (mc_solutions - mean_field[None, :, :]) / std_safe[None, :, :]
    return residuals


def compute_max_statistics(residuals, axis=(-1,)):
    """计算每个 MC 实现的极大统计量.

    M_m = max_x |Z_m(x)|

    参数
    ----
    residuals : ndarray, shape (n_mc, ...)
    axis : tuple of int

    返回
    ----
    max_stats : ndarray, shape (n_mc,)
    """
    return np.max(np.abs(residuals), axis=axis)


def bootstrap_max_statistics(residuals, n_bootstrap=500, seed=42):
    """Bootstrap 极大统计量分布.

    参数
    ----
    residuals : ndarray, shape (n_mc, nt, nx)
    n_bootstrap : int
    seed : int

    返回
    ----
    boot_max_stats : ndarray, shape (n_bootstrap,)
        各 Bootstrap 样本的极大统计量
    """
    rng = np.random.default_rng(seed)
    n_mc = residuals.shape[0]

    boot_max = np.zeros(n_bootstrap)
    for b in range(n_bootstrap):
        # Bootstrap 重采样
        idx = rng.choice(n_mc, size=n_mc, replace=True)
        resampled = residuals[idx]
        boot_max[b] = np.max(np.abs(resampled))

    return boot_max


def empirical_cdf(values, eval_points):
    """计算经验 CDF.

    F̂(x) = (1/N) Σ_i 1{x_i ≤ x}

    参数
    ----
    values : ndarray
    eval_points : ndarray

    返回
    ----
    cdf_values : ndarray
    """
    sorted_vals = np.sort(values)
    n = len(sorted_vals)
    cdf = np.searchsorted(sorted_vals, eval_points, side='right') / n
    return cdf


def compute_sobol_indices_approx(mc_solutions, input_params):
    """近似计算一阶 Sobol 灵敏度指数.

    S_i ≈ Var[E[Y|X_i]] / Var[Y]

    使用相关系数近似.

    参数
    ----
    mc_solutions : ndarray, shape (n_mc, n_points)
    input_params : ndarray, shape (n_mc, n_params)

    返回
    ----
    sobol : ndarray, shape (n_params,)
    """
    n_params = input_params.shape[1]
    output_var = np.var(np.mean(mc_solutions, axis=1))
    if output_var < 1e-30:
        return np.zeros(n_params)

    sobol = np.zeros(n_params)
    mean_y = np.mean(mc_solutions, axis=1)
    for k in range(n_params):
        corr = np.corrcoef(input_params[:, k], mean_y)[0, 1]
        sobol[k] = corr ** 2 if not np.isnan(corr) else 0.0

    return sobol
