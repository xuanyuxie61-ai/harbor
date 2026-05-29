# -*- coding: utf-8 -*-
"""
parameter_sampling.py

基于 latin_edge (Latin hypercube edge sampling) 与 opt_sample (random sampling optimization)
的激光-等离子体参数空间采样与优化模块。

原项目 651_latin_edge 提供了在 [0,1]^d 上生成边缘拉丁超立方样本的算法；
原项目 837_opt_sample 提供了基于随机采样的函数极值估计。
二者融合后用于:
    1. 在激光参数空间 (强度、波长、焦斑尺寸、脉冲宽度) 和
       等离子体参数空间 (密度、温度、标长) 上进行结构化采样。
    2. 通过随机优化寻找使能量耦合效率最大化的最优参数组合。

核心公式:
    Latin edge 采样:
        对每个维度 i，生成随机排列 perm，样本点为:
            x_{i,j} = (perm(j) - 1) / (point_num - 1)
        保证每行/列恰有一个样本位于边缘网格上。

    参数空间变换:
        p = p_min + x * (p_max - p_min)

    优化目标（能量耦合效率）:
        η_couple = E_dep / E_incident
        其中 E_dep 为逆轫致吸收沉积能量，E_incident 为入射激光能量。
"""

import numpy as np


def latin_edge_sample(dim_num, point_num, seed=None):
    """
    生成 d 维边缘拉丁超立方样本。

    原 latin_edge 核心算法:
        对每个维度 i，生成 1..point_num 的随机排列 perm，
        样本坐标 x_{i,j} = (perm(j) - 1) / (point_num - 1)。

    Parameters
    ----------
    dim_num : int
        维度数。
    point_num : int
        样本点数，必须 >= 2。
    seed : int, optional
        随机种子。

    Returns
    -------
    samples : ndarray, shape (dim_num, point_num)
        样本矩阵，每列为一个样本点。
    """
    if point_num < 2:
        raise ValueError("point_num 必须 >= 2。")
    if dim_num < 1:
        raise ValueError("dim_num 必须 >= 1。")

    rng = np.random.default_rng(seed)
    samples = np.zeros((dim_num, point_num), dtype=float)

    for i in range(dim_num):
        perm = rng.permutation(point_num)
        for j in range(point_num):
            samples[i, j] = perm[j] / (point_num - 1.0)

    return samples


def transform_samples_to_parameter_space(samples, param_bounds):
    """
    将 [0,1]^d 样本映射到实际参数空间。

    Parameters
    ----------
    samples : ndarray, shape (dim_num, point_num)
        拉丁超立方样本。
    param_bounds : list of tuple
        每个维度的 (p_min, p_max)。

    Returns
    -------
    params : ndarray, shape (dim_num, point_num)
        参数空间样本。
    """
    dim_num = samples.shape[0]
    if len(param_bounds) != dim_num:
        raise ValueError("param_bounds 长度必须与样本维度数一致。")

    params = np.zeros_like(samples)
    for i in range(dim_num):
        p_min, p_max = param_bounds[i]
        params[i, :] = p_min + samples[i, :] * (p_max - p_min)
    return params


def random_sampling_optimizer(objective_func, param_bounds, n_samples=5000, seed=None):
    """
    基于随机采样的全局优化器（源自 opt_sample 思想）。

    在参数空间中均匀随机采样 n_samples 个点，计算目标函数值，
    返回最优解。

    Parameters
    ----------
    objective_func : callable
        目标函数 f(p) -> float，p 为形状 (d,) 的 ndarray。
    param_bounds : list of tuple
        每个维度的 (p_min, p_max)。
    n_samples : int, optional
        采样点数，默认 5000。
    seed : int, optional
        随机种子。

    Returns
    -------
    best_param : ndarray
        最优参数组合。
    best_value : float
        最优目标函数值。
    all_values : ndarray
        所有采样点的目标函数值。
    all_params : ndarray
        所有采样点。
    """
    dim_num = len(param_bounds)
    rng = np.random.default_rng(seed)

    all_params = rng.random((n_samples, dim_num))
    # 映射到参数空间
    for i in range(dim_num):
        p_min, p_max = param_bounds[i]
        all_params[:, i] = p_min + all_params[:, i] * (p_max - p_min)

    all_values = np.zeros(n_samples, dtype=float)
    for k in range(n_samples):
        try:
            val = objective_func(all_params[k, :])
            if not np.isfinite(val):
                val = -np.inf
        except Exception:
            val = -np.inf
        all_values[k] = val

    best_idx = np.argmax(all_values)
    best_param = all_params[best_idx, :]
    best_value = all_values[best_idx]

    return best_param, best_value, all_values, all_params


def latin_hypercube_optimizer(objective_func, param_bounds, point_num=50, seed=None):
    """
    基于拉丁超立方采样的优化器。

    Parameters
    ----------
    objective_func : callable
        目标函数。
    param_bounds : list of tuple
        参数边界。
    point_num : int, optional
        每维采样点数，默认 50。
    seed : int, optional
        随机种子。

    Returns
    -------
    best_param : ndarray
        最优参数。
    best_value : float
        最优值。
    all_values : ndarray
        所有点的目标函数值。
    all_params : ndarray
        所有参数点。
    """
    dim_num = len(param_bounds)
    samples = latin_edge_sample(dim_num, point_num, seed=seed)
    params = transform_samples_to_parameter_space(samples, param_bounds)

    all_values = np.zeros(point_num, dtype=float)
    for k in range(point_num):
        try:
            val = objective_func(params[:, k])
            if not np.isfinite(val):
                val = -np.inf
        except Exception:
            val = -np.inf
        all_values[k] = val

    best_idx = np.argmax(all_values)
    best_param = params[:, best_idx]
    best_value = all_values[best_idx]

    return best_param, best_value, all_values, params


def sample_laser_plasma_parameters(n_samples, seed=None):
    """
    生成激光-等离子体参数的标准化样本。

    参数空间 (6 维):
        0: 激光强度 I_0 [W/m^2], 范围 [1e16, 1e20]
        1: 激光波长 λ [m], 范围 [0.3e-6, 1.064e-6]
        2: 焦斑半径 w_0 [m], 范围 [1e-6, 50e-6]
        3: 峰值密度 n_0 [m^{-3}], 范围 [1e24, 1e27]
        4: 电子温度 T_e [eV], 范围 [100, 5000]
        5: 密度标长 L_s [m], 范围 [1e-6, 100e-6]

    Parameters
    ----------
    n_samples : int
        样本数。
    seed : int, optional
        随机种子。

    Returns
    -------
    params : ndarray, shape (n_samples, 6)
        参数样本。
    param_names : list of str
        参数名称。
    param_bounds : list of tuple
        参数边界。
    """
    param_names = [
        'laser_intensity_W_m2',
        'laser_wavelength_m',
        'focal_spot_radius_m',
        'peak_density_m3',
        'electron_temperature_eV',
        'density_scale_length_m'
    ]
    param_bounds = [
        (1e16, 1e20),
        (0.3e-6, 1.064e-6),
        (1e-6, 50e-6),
        (1e24, 1e27),
        (100.0, 5000.0),
        (1e-6, 100e-6)
    ]

    dim_num = len(param_bounds)
    samples = latin_edge_sample(dim_num, n_samples, seed=seed)
    params = transform_samples_to_parameter_space(samples, param_bounds)
    return params.T, param_names, param_bounds


def sample_quality_metrics(params):
    """
    评估参数样本的质量指标。

    计算最小间距比、覆盖度等量，用于判断采样的数值鲁棒性。

    Parameters
    ----------
    params : ndarray, shape (n_samples, dim_num)
        参数样本（已归一化到 [0,1] 或实际空间）。

    Returns
    -------
    metrics : dict
        包含 'min_pairwise_dist', 'max_pairwise_dist', 'covering_radius' 等指标。
    """
    n_samples = params.shape[0]
    if n_samples < 2:
        return {'min_pairwise_dist': 0.0, 'max_pairwise_dist': 0.0, 'covering_radius': 0.0}

    # 计算所有两两欧氏距离
    dists = []
    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            d = np.linalg.norm(params[i, :] - params[j, :])
            dists.append(d)
    dists = np.array(dists)

    metrics = {
        'min_pairwise_dist': float(np.min(dists)),
        'max_pairwise_dist': float(np.max(dists)),
        'mean_pairwise_dist': float(np.mean(dists)),
        'covering_radius': float(np.max(dists) / 2.0)
    }
    return metrics
