# -*- coding: utf-8 -*-

import numpy as np


def craps_exact_probability():
    return 244.0 / 495.0


def monte_carlo_phase_error(n_trials, sigma_phase, n_pixels,
                            phase_design, target_far_field,
                            propagate_func=None, seed=42):
    np.random.seed(seed)
    phase_design = np.asarray(phase_design, dtype=float)
    target_far_field = np.asarray(target_far_field, dtype=complex)
    errors = np.zeros(n_trials)

    if propagate_func is None:

        def propagate_func(phase_profile):
            t = np.exp(1j * phase_profile)
            far = np.fft.fftshift(np.fft.fft2(t))
            return far

    for trial in range(n_trials):
        noise = np.random.normal(0.0, sigma_phase, phase_design.shape)
        phase_noisy = phase_design + noise

        phase_noisy = np.mod(phase_noisy + np.pi, 2.0 * np.pi) - np.pi
        far_noisy = propagate_func(phase_noisy)

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
    errors = np.asarray(errors, dtype=float)
    if errors.shape[0] == 0:
        return 0.0
    n_pass = np.sum(errors < threshold)
    return float(n_pass) / float(errors.shape[0])


def gaussian_error_cdf(x, mu, sigma):
    from math import erf, sqrt
    if sigma <= 0:
        sigma = 1e-15
    return 0.5 * (1.0 + erf((x - mu) / (sigma * sqrt(2.0))))


def tolerance_sensitivity_analysis(phase_design, param_ranges,
                                   propagate_func, target_far_field):
    results = {}
    for param_name, values in param_ranges.items():
        param_results = []
        for val in values:

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
