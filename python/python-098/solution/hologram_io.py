# -*- coding: utf-8 -*-
"""
hologram_io.py
基于 scip_solution_read（优化解文件读取），
实现超表面全息图二进制/多相位配置的读写与验证。

核心科学问题：
  全息超表面的设计可建模为组合优化问题：
      min_{b_i ∈ {0,1}} || E_target - Σ_i b_i h_i ||²
  其中 b_i 为二元相位状态（0 或 π），h_i 为第 i 个单元的点扩散函数。
  通过 SCIP/MIP 求解器得到最优配置后，需读取并验证解的物理可行性。

关键公式：
  1. 二元相位全息图:
       t_i ∈ {+1, -1}  对应相位 φ_i ∈ {0, π}
  2. 多相位全息图（Q 阶量化）:
       φ_i ∈ {0, 2π/Q, 2·2π/Q, ..., (Q-1)·2π/Q}
  3. Gerchberg-Saxton 迭代:
       空间域约束: |E_out| = A_target
       频域约束:    phase(E_far) = φ_design
  4. 约束验证:
       对于每个单元 i，|t_i| ≤ 1（被动系统能量守恒）
"""

import numpy as np


def read_binary_hologram(filename, n_pixels):
    """
    读取二进制全息图配置文件（参考 scip_solution_read）。
    文件格式：每行包含 "xi value"，表示第 i 个变量取值（0 或 1）。
    忽略前 2 行头部注释。

    参数:
        filename: 文件路径
        n_pixels: 总像素数 N = n_side²
    返回:
        config: 长度为 n_pixels 的 0/1 数组
    """
    config = np.zeros(n_pixels, dtype=int)
    with open(filename, 'r') as f:
        lines = f.readlines()
    # 忽略前 2 行
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
            # 提取变量索引，如 "x7" -> 7
            if var_name.startswith('x') or var_name.startswith('X'):
                idx_str = var_name[1:]
            else:
                idx_str = var_name
            idx = int(idx_str) - 1  # 1-based to 0-based
            if 0 <= idx < n_pixels:
                config[idx] = 1 if value != 0 else 0
        except ValueError:
            continue
    return config


def write_binary_hologram(filename, config):
    """
    写出二进制全息图配置文件。
    """
    config = np.asarray(config, dtype=int)
    n = config.shape[0]
    with open(filename, 'w') as f:
        f.write("# Binary hologram configuration\n")
        f.write(f"# Total pixels: {n}\n")
        for i in range(n):
            f.write(f"x{i+1} {config[i]}\n")


def binary_to_phase_config(binary_config, phase_levels=2):
    """
    将二元配置转换为多相位配置。
    phase_levels=2: φ ∈ {0, π}
    phase_levels=4: φ ∈ {0, π/2, π, 3π/2}
    若输入为多电平（0..Q-1），直接映射。
    """
    binary_config = np.asarray(binary_config, dtype=int)
    n = binary_config.shape[0]
    phases = np.zeros(n, dtype=float)
    if phase_levels <= 1:
        return phases
    for i in range(n):
        val = binary_config[i]
        # 将整数取值映射到 [0, 2π)
        val = val % phase_levels
        phases[i] = 2.0 * np.pi * val / phase_levels
    return phases


def phase_to_binary_config(phases, phase_levels=2):
    """
    将连续相位量化到最近的离散相位电平。
    """
    phases = np.asarray(phases, dtype=float)
    n = phases.size
    config = np.zeros(n, dtype=int)
    if phase_levels <= 1:
        return config
    phases_flat = phases.ravel()
    for i in range(n):
        # 归一化到 [0, 2π)
        phi = float(np.mod(phases_flat[i], 2.0 * np.pi))
        level = int(round(phi / (2.0 * np.pi / phase_levels))) % phase_levels
        config[i] = level
    return config


def validate_hologram_config(config, constraints):
    """
    验证全息图配置是否满足物理约束。

    constraints 字典可包含：
      - 'max_ones': 1 的最大数量（能量预算）
      - 'min_ones': 1 的最小数量
      - 'phase_levels': 允许的相位电平数
      - 'max_gradient': 相邻单元间的最大相位梯度（防止过陡相位）
    返回: (is_valid, violations)
    """
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
    """
    Gerchberg-Saxton 迭代算法（核心全息优化算法）。

    参数:
        target_amplitude: 目标远场幅度（2-D array，已归一化）
        initial_phase:    初始相位猜测（2-D array）
        n_iter:           迭代次数
    返回:
        phase: 优化后的相位剖面
        error_history: 每次迭代的误差列表
    """
    target_amplitude = np.asarray(target_amplitude, dtype=float)
    initial_phase = np.asarray(initial_phase, dtype=float)
    if target_amplitude.shape != initial_phase.shape:
        raise ValueError("target_amplitude 与 initial_phase 形状必须一致")

    phase = initial_phase.copy()
    error_history = []
    for it in range(n_iter):
        # 空间域：施加幅度约束（透射幅度为 1）
        E_space = np.exp(1j * phase)
        # 频域
        E_far = np.fft.fftshift(np.fft.fft2(E_space))
        amplitude_far = np.abs(E_far)
        # 归一化
        if np.max(amplitude_far) > 1e-15:
            amplitude_far = amplitude_far / np.max(amplitude_far)
        # 频域约束：替换幅度，保留相位
        phase_far = np.angle(E_far)
        E_far_new = target_amplitude * np.exp(1j * phase_far)
        # 逆变换回空间域
        E_space_new = np.fft.ifft2(np.fft.ifftshift(E_far_new))
        phase = np.angle(E_space_new)
        # 计算误差
        diff = amplitude_far - target_amplitude
        error = np.sqrt(np.mean(diff ** 2))
        error_history.append(error)

    return phase, error_history
