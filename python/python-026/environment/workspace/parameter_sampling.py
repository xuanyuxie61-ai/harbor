# -*- coding: utf-8 -*-

import numpy as np


def latin_edge_sample(dim_num, point_num, seed=None):
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
    dim_num = samples.shape[0]
    if len(param_bounds) != dim_num:
        raise ValueError("param_bounds 长度必须与样本维度数一致。")

    params = np.zeros_like(samples)
    for i in range(dim_num):
        p_min, p_max = param_bounds[i]
        params[i, :] = p_min + samples[i, :] * (p_max - p_min)
    return params


def random_sampling_optimizer(objective_func, param_bounds, n_samples=5000, seed=None):
    dim_num = len(param_bounds)
    rng = np.random.default_rng(seed)

    all_params = rng.random((n_samples, dim_num))

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
    n_samples = params.shape[0]
    if n_samples < 2:
        return {'min_pairwise_dist': 0.0, 'max_pairwise_dist': 0.0, 'covering_radius': 0.0}


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
