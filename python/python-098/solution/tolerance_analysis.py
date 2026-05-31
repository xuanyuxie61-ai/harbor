# -*- coding: utf-8 -*-
"""
tolerance_analysis.py
基于 craps_simulation（概率统计模拟），对超表面制造公差进行蒙特卡洛分析。

核心科学问题：
  纳米加工误差（如电子束光刻的线宽粗糙度、刻蚀深度不均匀性）
  会导致 meta-atom 的谐振频率偏移，进而产生相位误差 δφ。
  通过蒙特卡洛模拟大量随机误差样本，评估全息重建的良率（yield）。

关键公式：
  1. 相位误差统计模型:
       δφ_i ~ N(0, σ_φ²)
  2. 远场重构误差（均方根）:
       ε = √( (1/N) Σ |E_target - E_reconstructed|² )
  3. 良率（Yield）:
       Yield = P(ε < ε_threshold) ≈ (n_pass / n_total)
  4. 误差传播（一阶近似）:
       δφ_total² = Σ_i (∂E/∂φ_i)² δφ_i²
  5. 参考 craps_probability 的精确概率思想：
       对于独立高斯误差，可通过解析累积分布函数快速估计。
"""

import numpy as np


def craps_exact_probability():
    """
    参考 craps_probability 的精确概率思想。
    这里返回一个占位常数，用于标定蒙特卡洛的基准。
    在 craps 中 P(win) = 244/495 ≈ 0.4929。
    """
    return 244.0 / 495.0


def monte_carlo_phase_error(n_trials, sigma_phase, n_pixels,
                            phase_design, target_far_field,
                            propagate_func=None, seed=42):
    """
    对超表面相位分布进行蒙特卡洛制造误差模拟。

    参数:
        n_trials:      模拟次数
        sigma_phase:   相位标准差（弧度）
        n_pixels:      每边像素数
        phase_design:  设计的理想相位剖面（2-D array）
        target_far_field: 目标远场分布（1-D 或 2-D）
        propagate_func: 传播函数，输入相位剖面，输出远场
    返回:
        results: dict 包含 errors, yield_rate, mean_error, std_error
    """
    np.random.seed(seed)
    phase_design = np.asarray(phase_design, dtype=float)
    target_far_field = np.asarray(target_far_field, dtype=complex)
    errors = np.zeros(n_trials)

    if propagate_func is None:
        # 默认传播函数：简化的傅里叶变换远场
        def propagate_func(phase_profile):
            t = np.exp(1j * phase_profile)
            far = np.fft.fftshift(np.fft.fft2(t))
            return far

    for trial in range(n_trials):
        noise = np.random.normal(0.0, sigma_phase, phase_design.shape)
        phase_noisy = phase_design + noise
        # 相位 wrap 到 [-π, π]
        phase_noisy = np.mod(phase_noisy + np.pi, 2.0 * np.pi) - np.pi
        far_noisy = propagate_func(phase_noisy)
        # 归一化后比较
        if np.max(np.abs(far_noisy)) > 1e-15:
            far_noisy = far_noisy / np.max(np.abs(far_noisy))
        if np.max(np.abs(target_far_field)) > 1e-15:
            target_norm = target_far_field / np.max(np.abs(target_far_field))
        else:
            target_norm = target_far_field
        diff = far_noisy - target_norm
        error = np.sqrt(np.mean(np.abs(diff) ** 2))
        errors[trial] = error

    mean_error = float(np.mean(errors))
    std_error = float(np.std(errors))
    median_error = float(np.median(errors))
    return {
        'errors': errors,
        'mean_error': mean_error,
        'std_error': std_error,
        'median_error': median_error,
        'min_error': float(np.min(errors)),
        'max_error': float(np.max(errors))
    }


def estimate_yield(errors, threshold):
    """
    根据模拟误差估计良率：P(error < threshold)。
    """
    errors = np.asarray(errors, dtype=float)
    if errors.shape[0] == 0:
        return 0.0
    n_pass = np.sum(errors < threshold)
    return float(n_pass) / float(errors.shape[0])


def gaussian_error_cdf(x, mu, sigma):
    """
    高斯累积分布函数（参考正态分布表）。
    使用误差函数 erf：
        Φ(x) = 0.5 [1 + erf((x-μ)/(σ√2))]
    """
    from math import erf, sqrt
    if sigma <= 0:
        sigma = 1e-15
    return 0.5 * (1.0 + erf((x - mu) / (sigma * sqrt(2.0))))


def tolerance_sensitivity_analysis(phase_design, param_ranges,
                                   propagate_func, target_far_field):
    """
    对多个制造参数进行灵敏度分析。
    param_ranges: dict，如 {'sigma_phase': [0.01, 0.05, 0.1, 0.2]}
    返回各参数下的误差统计。
    """
    results = {}
    for param_name, values in param_ranges.items():
        param_results = []
        for val in values:
            # 简化为单一参数变化
            res = monte_carlo_phase_error(
                n_trials=200,
                sigma_phase=val if param_name == 'sigma_phase' else 0.05,
                n_pixels=phase_design.shape[0],
                phase_design=phase_design,
                target_far_field=target_far_field,
                propagate_func=propagate_func,
                seed=42 + int(val * 100)
            )
            param_results.append({
                'param_value': val,
                'mean_error': res['mean_error'],
                'std_error': res['std_error']
            })
        results[param_name] = param_results
    return results
