# -*- coding: utf-8 -*-

import numpy as np


def read_binary_hologram(filename, n_pixels):
    config = np.zeros(n_pixels, dtype=int)
    with open(filename, 'r') as f:
        lines = f.readlines()

    for line in lines[2:]:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            var_name = parts[0]
            value = int(parts[1])

            if var_name.startswith('x') or var_name.startswith('X'):
                idx_str = var_name[1:]
            else:
                idx_str = var_name
            idx = int(idx_str) - 1
            if 0 <= idx < n_pixels:
                config[idx] = 1 if value != 0 else 0
        except ValueError:
            continue
    return config


def write_binary_hologram(filename, config):
    config = np.asarray(config, dtype=int)
    n = config.shape[0]
    with open(filename, 'w') as f:
        f.write("# Binary hologram configuration\n")
        f.write(f"# Total pixels: {n}\n")
        for i in range(n):
            f.write(f"x{i+1} {config[i]}\n")


def binary_to_phase_config(binary_config, phase_levels=2):
    binary_config = np.asarray(binary_config, dtype=int)
    n = binary_config.shape[0]
    phases = np.zeros(n, dtype=float)
    if phase_levels <= 1:
        return phases
    for i in range(n):
        val = binary_config[i]

        val = val % phase_levels
        phases[i] = 2.0 * np.pi * val / phase_levels
    return phases


def phase_to_binary_config(phases, phase_levels=2):
    phases = np.asarray(phases, dtype=float)
    n = phases.size
    config = np.zeros(n, dtype=int)
    if phase_levels <= 1:
        return config
    phases_flat = phases.ravel()
    for i in range(n):

        phi = float(np.mod(phases_flat[i], 2.0 * np.pi))
        level = int(round(phi / (2.0 * np.pi / phase_levels))) % phase_levels
        config[i] = level
    return config


def validate_hologram_config(config, constraints):
    config = np.asarray(config, dtype=int)
    violations = []

    n_ones = int(np.sum(config))
    if 'max_ones' in constraints and n_ones > constraints['max_ones']:
        violations.append(f"ones count {n_ones} exceeds max {constraints['max_ones']}")
    if 'min_ones' in constraints and n_ones < constraints['min_ones']:
        violations.append(f"ones count {n_ones} below min {constraints['min_ones']}")

    if 'max_gradient' in constraints and config.ndim >= 2:
        max_grad = constraints['max_gradient']
        ny, nx = config.shape
        for i in range(ny):
            for j in range(nx - 1):
                if abs(int(config[i, j]) - int(config[i, j + 1])) > max_grad:
                    violations.append(f"gradient violation at ({i},{j})-({i},{j+1})")
        for i in range(ny - 1):
            for j in range(nx):
                if abs(int(config[i, j]) - int(config[i + 1, j])) > max_grad:
                    violations.append(f"gradient violation at ({i},{j})-({i+1},{j})")

    is_valid = len(violations) == 0
    return is_valid, violations


def gerchberg_saxton_iteration(target_amplitude, initial_phase, n_iter=20):
    target_amplitude = np.asarray(target_amplitude, dtype=float)
    initial_phase = np.asarray(initial_phase, dtype=float)
    if target_amplitude.shape != initial_phase.shape:
        raise ValueError("target_amplitude 与 initial_phase 形状必须一致")

    phase = initial_phase.copy()
    error_history = []
    for it in range(n_iter):

        E_space = np.exp(1j * phase)

        E_far = np.fft.fftshift(np.fft.fft2(E_space))
        amplitude_far = np.abs(E_far)

        if np.max(amplitude_far) > 1e-15:
            amplitude_far = amplitude_far / np.max(amplitude_far)

        phase_far = np.angle(E_far)
        E_far_new = target_amplitude * np.exp(1j * phase_far)

        E_space_new = np.fft.ifft2(np.fft.ifftshift(E_far_new))
        phase = np.angle(E_space_new)

        diff = amplitude_far - target_amplitude
        error = np.sqrt(np.mean(diff ** 2))
        error_history.append(error)

    return phase, error_history
