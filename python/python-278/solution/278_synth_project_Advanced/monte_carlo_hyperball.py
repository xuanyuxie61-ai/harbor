"""
monte_carlo_hyperball.py
========================
蒙特卡罗不确定性传播: 基于超球面采样分析 CALPHAD 参数的
不确定性对相平衡预测的影响。

种子项目 555_hyperball_positive_distance: 估计超球体内
两点间距离的均值和方差, 使用 Monte Carlo 方法。

本模块将此思想推广:
  1. 在参数空间的正超球体内采样
  2. 计算参数扰动后相平衡成分的统计分布
  3. 报告均值、方差、置信区间

采样方法:
  在 R^d 的单位正超球体内均匀采样:
    x_i = |z_i| / ||z|| * r^{1/d}
  其中 z_i ~ N(0,1), r ~ U(0,1)

参数扰动模型:
  L^(n)_perturbed = L^(n)_nominal * (1 + sigma_n * xi_n)
  其中 xi ∈ B^+_d (正超球体), sigma_n 为相对不确定度
"""

import numpy as np
from calphad_fec_constants import (
    MC_N_SAMPLES, MC_N_DIM, MC_RANDOM_SEED,
    L_FCC_FE_C, L_BCC_FE_C, L_LIQUID_FE_C,
    R_GAS,
)


def sample_positive_hyperball(n_samples, n_dim, seed=None):
    """
    在 R^d 的单位正超球体内均匀采样。

    方法 (Muller 1959):
      1. 生成 n_dim 维标准正态向量 z
      2. 取绝对值 |z| (映射到正卦限)
      3. 归一化到单位球面: u = |z| / ||z||
      4. 径向缩放: x = u * r^{1/d}, r ~ U(0,1)

    Parameters
    ----------
    n_samples : int
        采样数
    n_dim : int
        维度
    seed : int
        随机种子

    Returns
    -------
    np.ndarray
        形状 (n_samples, n_dim) 的采样点矩阵
    """
    if seed is None:
        seed = MC_RANDOM_SEED
    rng = np.random.RandomState(seed)

    # 正态采样 + 绝对值 (正卦限)
    Z = np.abs(rng.randn(n_samples, n_dim))

    # 归一化到球面
    norms = np.linalg.norm(Z, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-30)  # 避免零向量
    U = Z / norms

    # 径向缩放: r^{1/d}
    r = rng.rand(n_samples, 1)
    radial = r ** (1.0 / n_dim)

    samples = U * radial
    return samples


def pairwise_distance_statistics(samples):
    """
    计算采样点集合的成对距离统计量。

    对于 N 个 d 维采样点, 计算所有 C(N,2) 对距离的
    均值和方差。借鉴 seed 555 的核心计算。

    E[D] = 积分 integral integral ||x - y|| rho(x) rho(y) dx dy
    Var[D] = E[D²] - (E[D])²

    Parameters
    ----------
    samples : np.ndarray
        (N, d) 采样矩阵

    Returns
    -------
    dict
        {
            'mean_distance': float,
            'var_distance': float,
            'std_distance': float,
            'min_distance': float,
            'max_distance': float,
            'n_pairs': int,
        }
    """
    N = samples.shape[0]
    if N < 2:
        return {
            'mean_distance': 0.0, 'var_distance': 0.0,
            'std_distance': 0.0, 'min_distance': 0.0,
            'max_distance': 0.0, 'n_pairs': 0,
        }

    # 成对距离 (上三角)
    distances = []
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(samples[i] - samples[j])
            distances.append(d)

    distances = np.array(distances)
    n_pairs = len(distances)

    mean_d = np.mean(distances) if n_pairs > 0 else 0.0
    var_d = np.var(distances) if n_pairs > 0 else 0.0

    return {
        'mean_distance': float(mean_d),
        'var_distance': float(var_d),
        'std_distance': float(np.sqrt(max(var_d, 0.0))),
        'min_distance': float(np.min(distances)) if n_pairs > 0 else 0.0,
        'max_distance': float(np.max(distances)) if n_pairs > 0 else 0.0,
        'n_pairs': n_pairs,
    }


def calphad_uncertainty_propagation(T, phase_a, phase_b,
                                    sigma_params=None,
                                    n_samples=None):
    """
    CALPHAD 参数不确定性传播到相平衡预测。

    对每个 MC 样本:
      1. 扰动 Redlich-Kister 参数: L^(n) → L^(n)*(1 + sigma*xi_n)
      2. 重新计算相平衡
      3. 记录平衡成分

    Parameters
    ----------
    T : float
        温度 (K)
    phase_a : str
        alpha 相
    phase_b : str
        beta 相
    sigma_params : list of float, optional
        各参数的相对不确定度 (默认 [0.05, 0.05, ..., 0.05])
    n_samples : int, optional
        MC 样本数

    Returns
    -------
    dict
        {
            'x_alpha_mean': float,
            'x_alpha_std': float,
            'x_beta_mean': float,
            'x_beta_std': float,
            'x_alpha_samples': np.ndarray,
            'x_beta_samples': np.ndarray,
            'param_samples': np.ndarray,
            'distance_stats': dict,
            'confidence_interval_95': tuple,
        }
    """
    if n_samples is None:
        n_samples = MC_N_SAMPLES
    if sigma_params is None:
        sigma_params = [0.05] * MC_N_DIM  # 5% 相对不确定度

    # 在超球体内采样
    xi = sample_positive_hyperball(n_samples, MC_N_DIM, seed=MC_RANDOM_SEED)

    x_alpha_samples = np.zeros(n_samples)
    x_beta_samples = np.zeros(n_samples)

    # 原始参数
    L_orig = L_FCC_FE_C[:3] + L_BCC_FE_C[:2] + L_LIQUID_FE_C[:1]
    n_params = min(len(L_orig), MC_N_DIM)

    from newton_maehly_equilibrium import newton_maehly_solve

    for s in range(n_samples):
        # 扰动参数
        L_perturbed = list(L_orig)
        for p in range(n_params):
            if p < len(L_perturbed) and p < len(sigma_params):
                L_perturbed[p] = L_orig[p] * (1.0 + sigma_params[p] * xi[s, p])

        # 临时修改全局参数 (使用上下文管理器模式)
        import calphad_fec_constants as const
        import gibbs_energy_calphad as ge

        # 保存原始参数
        orig_fcc = const.L_FCC_FE_C[:]
        orig_bcc = const.L_BCC_FE_C[:]
        orig_liq = const.L_LIQUID_FE_C[:]

        # 设置扰动参数
        const.L_FCC_FE_C = L_perturbed[:3]
        const.L_BCC_FE_C = L_perturbed[3:5]
        const.L_LIQUID_FE_C = L_perturbed[5:6] if len(L_perturbed) > 5 else orig_liq

        # 求解相平衡
        try:
            result = newton_maehly_solve(T, phase_a, phase_b)
            if result['converged']:
                x_alpha_samples[s] = result['x_C_alpha']
                x_beta_samples[s] = result['x_C_beta']
            else:
                x_alpha_samples[s] = np.nan
                x_beta_samples[s] = np.nan
        except Exception:
            x_alpha_samples[s] = np.nan
            x_beta_samples[s] = np.nan

        # 恢复原始参数
        const.L_FCC_FE_C = orig_fcc
        const.L_BCC_FE_C = orig_bcc
        const.L_LIQUID_FE_C = orig_liq

    # 统计量
    valid_alpha = x_alpha_samples[~np.isnan(x_alpha_samples)]
    valid_beta = x_beta_samples[~np.isnan(x_beta_samples)]

    # 成对距离统计
    param_dist = pairwise_distance_statistics(xi[:min(200, n_samples)])

    # 95% 置信区间
    if len(valid_alpha) > 2:
        alpha_mean = np.mean(valid_alpha)
        alpha_std = np.std(valid_alpha, ddof=1)
        se = alpha_std / np.sqrt(len(valid_alpha))
        ci_95 = (alpha_mean - 1.96 * se, alpha_mean + 1.96 * se)
    else:
        alpha_mean = 0.0
        alpha_std = 0.0
        ci_95 = (0.0, 0.0)

    return {
        'x_alpha_mean': float(alpha_mean),
        'x_alpha_std': float(alpha_std),
        'x_beta_mean': float(np.mean(valid_beta)) if len(valid_beta) > 0 else 0.0,
        'x_beta_std': float(np.std(valid_beta, ddof=1)) if len(valid_beta) > 1 else 0.0,
        'x_alpha_samples': x_alpha_samples,
        'x_beta_samples': x_beta_samples,
        'param_samples': xi,
        'distance_stats': param_dist,
        'confidence_interval_95': ci_95,
        'n_valid': len(valid_alpha),
    }


def sobol_sensitivity_indices(T, phase_a, phase_b, n_samples=100):
    """
    简化的一阶 Sobol 灵敏度指数计算。

    S_i = V[E[Y|X_i]] / V[Y]

    其中 Y = 相平衡成分, X_i = 第 i 个 CALPHAD 参数。

    使用 Saltelli 采样方案的高效估计:
        S_i ≈ (1/N) * sum_j Y_j * Y_j^(i) / V[Y] - (E[Y])²/V[Y]

    Parameters
    ----------
    T : float
        温度 (K)
    phase_a, phase_b : str
        相名称
    n_samples : int
        样本数

    Returns
    -------
    dict
        各参数的灵敏度指数
    """
    n_params = MC_N_DIM
    rng = np.random.RandomState(MC_RANDOM_SEED + 1)

    # 生成两组独立样本 A, B
    A = sample_positive_hyperball(n_samples, n_params, seed=MC_RANDOM_SEED)
    B = sample_positive_hyperball(n_samples, n_params, seed=MC_RANDOM_SEED + 42)

    sigma_params = [0.05] * n_params
    L_orig = L_FCC_FE_C[:3] + L_BCC_FE_C[:2] + L_LIQUID_FE_C[:1]
    n_p = min(len(L_orig), n_params)

    from newton_maehly_equilibrium import newton_maehly_solve
    import calphad_fec_constants as const

    def evaluate_with_perturbation(xi_row):
        L_pert = list(L_orig)
        for p in range(n_p):
            if p < len(L_pert):
                L_pert[p] = L_orig[p] * (1.0 + sigma_params[p] * xi_row[p])
        const.L_FCC_FE_C = L_pert[:3]
        const.L_BCC_FE_C = L_pert[3:5] if len(L_pert) > 3 else const.L_BCC_FE_C
        const.L_LIQUID_FE_C = L_pert[5:6] if len(L_pert) > 5 else const.L_LIQUID_FE_C
        try:
            res = newton_maehly_solve(T, phase_a, phase_b)
            return res['x_C_alpha'] if res['converged'] else np.nan
        except Exception:
            return np.nan

    # 评估 A, B
    Y_A = np.array([evaluate_with_perturbation(A[i]) for i in range(n_samples)])
    Y_B = np.array([evaluate_with_perturbation(B[i]) for i in range(n_samples)])

    # 恢复参数
    const.L_FCC_FE_C = L_orig[:3]
    const.L_BCC_FE_C = L_orig[3:5]
    const.L_LIQUID_FE_C = L_orig[5:6] if len(L_orig) > 5 else const.L_LIQUID_FE_C

    # 计算 Sobol 指数 (Jansen 估计器)
    var_Y = np.nanvar(np.concatenate([Y_A, Y_B]))
    S = np.zeros(n_params)

    if var_Y > 1e-30:
        for i in range(n_params):
            # 构造 AB_i: A 的第 i 列替换为 B 的第 i 列
            AB_i = A.copy()
            AB_i[:, i] = B[:, i]
            Y_AB_i = np.array([evaluate_with_perturbation(AB_i[j])
                               for j in range(n_samples)])
            # Jansen 估计: S_i = (1/(2N)) * sum (Y_A - Y_AB_i)² / V[Y]
            valid = ~(np.isnan(Y_A) | np.isnan(Y_AB_i))
            if np.sum(valid) > 1:
                S[i] = np.mean((Y_A[valid] - Y_AB_i[valid]) ** 2) / (2.0 * var_Y)

    # 恢复参数
    const.L_FCC_FE_C = L_orig[:3]
    const.L_BCC_FE_C = L_orig[3:5]
    const.L_LIQUID_FE_C = L_orig[5:6] if len(L_orig) > 5 else const.L_LIQUID_FE_C

    param_names = ['L0_FCC', 'L1_FCC', 'L2_FCC',
                   'L0_BCC', 'L1_BCC', 'L0_LIQ']
    return {
        'S_first_order': S,
        'param_names': param_names[:n_params],
        'total_variance': float(var_Y),
    }
