"""
诊断与后处理模块

PIC 模拟的物理量诊断:
- 能量守恒监控
- 动量守恒监控
- 相空间密度
- 傅里叶分析 (模式识别)
- 发射度计算
- 加热率估计

融合 1113 (模型鲁棒性分析) 与 719 (矩阵输出) 的思想
"""

import numpy as np
from typing import Dict, List, Tuple
try:
    from plasma_constants import (ELECTRON_MASS, ELECTRON_CHARGE,
                                  VACUUM_PERMITTIVITY, PI, BOLTZMANN_CONSTANT)
except ImportError:
    from plasma_constants import (ELECTRON_MASS, ELECTRON_CHARGE,
                                  VACUUM_PERMITTIVITY, PI, BOLTZMANN_CONSTANT)
try:
    from poisson_solver import energy_from_field
except ImportError:
    from poisson_solver import energy_from_field


def total_energy(x: np.ndarray, v: np.ndarray, E_field: np.ndarray,
                    weight: float, dx: float,
                    m: float = ELECTRON_MASS,
                    epsilon_0: float = VACUUM_PERMITTIVITY) -> float:
    """
    总能量:
        W = W_kin + W_field
        W_kin   = weight * m * sum(v^2) / 2
        W_field = epsilon_0 * sum(E^2) * dx / 2
    """
    W_kin = 0.5 * weight * m * np.sum(v**2)
    W_field = energy_from_field(E_field, dx, epsilon_0)
    return W_kin + W_field


def kinetic_energy(x: np.ndarray, v: np.ndarray, weight: float,
                      m: float = ELECTRON_MASS) -> float:
    """总动能"""
    return 0.5 * weight * m * np.sum(v**2)


def field_energy(E_field: np.ndarray, dx: float,
                    epsilon_0: float = VACUUM_PERMITTIVITY) -> float:
    """场能量"""
    return 0.5 * epsilon_0 * np.sum(E_field**2) * dx


def total_momentum(v: np.ndarray, weight: float,
                      m: float = ELECTRON_MASS) -> float:
    """
    总动量:
        P = weight * m * sum(v)
    """
    return weight * m * np.sum(v)


def energy_conservation_history(energy_history: np.ndarray) -> Dict:
    """
    能量守恒分析:
        relative_drift = (E_final - E_initial) / E_initial
        max_fluctuation = max|E - E_0| / E_0
    """
    E0 = energy_history[0]
    if abs(E0) < 1e-300:
        return {'relative_drift': 0.0, 'max_fluctuation': 0.0}

    relative = (energy_history - E0) / E0
    return {
        'relative_drift': relative[-1],
        'max_fluctuation': np.max(np.abs(relative)),
        'mean_drift': np.mean(relative),
        'std_drift': np.std(relative)
    }


def momentum_conservation_history(momentum_history: np.ndarray) -> Dict:
    """动量守恒分析"""
    P0 = momentum_history[0]
    if abs(P0) < 1e-300:
        return {'relative_drift': 0.0, 'max_fluctuation': np.max(np.abs(momentum_history))}
    relative = (momentum_history - P0) / P0
    return {
        'relative_drift': relative[-1],
        'max_fluctuation': np.max(np.abs(relative))
    }


def density_from_particles(x: np.ndarray, weight: float, N_grid: int,
                             dx: float, L: float, order: int = 1) -> np.ndarray:
    """
    从粒子位置计算密度 (PIC 沉积):
        rho(i) = sum_p weight * S(x_p - x_i)
    """
    rho = np.zeros(N_grid)
    for xp in x:
        xp_mod = xp % L
        i = int(xp_mod / dx)
        d = (xp_mod - i * dx) / dx
        if order == 0:
            rho[i % N_grid] += weight
        elif order == 1:
            rho[i % N_grid] += weight * (1.0 - d)
            rho[(i + 1) % N_grid] += weight * d
    rho /= dx
    return rho


def fourier_spectrum_density(rho: np.ndarray, dx: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    密度涨落的傅里叶谱:
        rho_hat(k) = FFT(rho - <rho>)

    Returns:
        (k_array, |rho_hat|)
    """
    N = len(rho)
    rho_fluct = rho - np.mean(rho)
    rho_hat = np.fft.fft(rho_fluct)
    k_array = np.fft.fftfreq(N, d=dx) * 2.0 * PI
    # 只取正频率
    n_pos = N // 2
    return k_array[:n_pos], np.abs(rho_hat[:n_pos])


def fourier_spectrum_field(E_field: np.ndarray, dx: float) -> Tuple[np.ndarray, np.ndarray]:
    """电场傅里叶谱"""
    N = len(E_field)
    E_hat = np.fft.fft(E_field)
    k_array = np.fft.fftfreq(N, d=dx) * 2.0 * PI
    n_pos = N // 2
    return k_array[:n_pos], np.abs(E_hat[:n_pos])


def mode_amplitude_time_series(E_history: np.ndarray, dx: float,
                                  mode: int) -> np.ndarray:
    """
    指定模式振幅随时间的演化:
        A_m(t) = |FFT(E(t))[mode]|
    """
    n_steps = E_history.shape[0]
    amplitude = np.zeros(n_steps)
    for t in range(n_steps):
        E_hat = np.fft.fft(E_history[t])
        amplitude[t] = np.abs(E_hat[mode])
    return amplitude


def growth_rate_from_amplitude(time: np.ndarray, amplitude: np.ndarray,
                                  t_start: float = None,
                                  t_end: float = None) -> float:
    """
    从振幅时间序列拟合增长率:
        A(t) ~ A_0 * exp(gamma * t)
        => log(A) = log(A_0) + gamma * t

    线性拟合 log(A) vs t, 斜率即为 gamma
    """
    if t_start is None:
        t_start = time[0]
    if t_end is None:
        t_end = time[-1]

    mask = (time >= t_start) & (time <= t_end) & (amplitude > 1e-300)
    if np.sum(mask) < 2:
        return 0.0

    t_fit = time[mask]
    a_fit = np.log(amplitude[mask])

    # 线性最小二乘
    n = len(t_fit)
    sum_t = np.sum(t_fit)
    sum_a = np.sum(a_fit)
    sum_t2 = np.sum(t_fit**2)
    sum_ta = np.sum(t_fit * a_fit)

    denom = n * sum_t2 - sum_t**2
    if abs(denom) < 1e-300:
        return 0.0

    gamma = (n * sum_ta - sum_t * sum_a) / denom
    return gamma


def emittance(x: np.ndarray, v: np.ndarray, weight: np.ndarray = None) -> float:
    """
    束发射度 (相空间面积):
        epsilon = sqrt(<x^2><v^2> - <xv>^2)
    """
    if weight is None:
        x_mean = np.mean(x)
        v_mean = np.mean(v)
        dx = x - x_mean
        dv = v - v_mean
        x2 = np.mean(dx**2)
        v2 = np.mean(dv**2)
        xv = np.mean(dx * dv)
    else:
        w_sum = np.sum(weight)
        x_mean = np.sum(weight * x) / w_sum
        v_mean = np.sum(weight * v) / w_sum
        dx = x - x_mean
        dv = v - v_mean
        x2 = np.sum(weight * dx**2) / w_sum
        v2 = np.sum(weight * dv**2) / w_sum
        xv = np.sum(weight * dx * dv) / w_sum

    eps_sq = x2 * v2 - xv**2
    return np.sqrt(max(eps_sq, 0.0))


def temperature_from_velocity(v: np.ndarray, weight: np.ndarray = None,
                                  m: float = ELECTRON_MASS) -> float:
    """
    从速度分布计算温度:
        T = m * (<v^2> - <v>^2) / k_B
    """
    if weight is None:
        v_mean = np.mean(v)
        v2_mean = np.mean(v**2)
    else:
        w_sum = np.sum(weight)
        v_mean = np.sum(weight * v) / w_sum
        v2_mean = np.sum(weight * v**2) / w_sum

    return m * (v2_mean - v_mean**2) / BOLTZMANN_CONSTANT


def heating_rate(energy_history: np.ndarray, dt: float,
                    t_start: int = None, t_end: int = None) -> float:
    """
    数值加热率 (线性拟合):
        dW/dt = slope of W(t)
    """
    if t_start is None:
        t_start = 0
    if t_end is None:
        t_end = len(energy_history)

    t = np.arange(t_start, t_end) * dt
    W = energy_history[t_start:t_end]

    if len(t) < 2:
        return 0.0

    n = len(t)
    slope = (n * np.sum(t * W) - np.sum(t) * np.sum(W)) / (n * np.sum(t**2) - np.sum(t)**2)
    return slope


def print_matrix_formatted(M: np.ndarray, name: str = 'Matrix',
                             precision: int = 4) -> str:
    """
    格式化矩阵输出 (from r8ge_print / magicsquare_print)
    """
    lines = [f"\n{name} ({M.shape[0]} x {M.shape[1]}):"]
    for i in range(M.shape[0]):
        row_str = "  ".join(f"{M[i, j]:{10 + precision}.{precision}e}" for j in range(M.shape[1]))
        lines.append(f"  [{row_str}]")
    return "\n".join(lines)


def robustness_metrics(energy_history: np.ndarray,
                         momentum_history: np.ndarray) -> Dict:
    """
    鲁棒性指标 (from 1113_model_robustness):
    - 能量漂移率
    - 动量漂移率
    - 最大瞬时涨落
    - 信噪比
    """
    energy_stats = energy_conservation_history(energy_history)
    momentum_stats = momentum_conservation_history(momentum_history)

    # 信噪比
    signal = np.mean(energy_history)
    noise = np.std(energy_history)
    snr = abs(signal) / (noise + 1e-300)

    return {
        'energy_drift': energy_stats['relative_drift'],
        'energy_max_fluctuation': energy_stats['max_fluctuation'],
        'momentum_drift': momentum_stats['relative_drift'],
        'momentum_max_fluctuation': momentum_stats['max_fluctuation'],
        'energy_snr': snr,
        'stable': abs(energy_stats['relative_drift']) < 0.05
    }


def phase_space_density_2d(x: np.ndarray, v: np.ndarray,
                               n_x_bins: int = 32, n_v_bins: int = 32,
                               weight: float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    2D 相空间密度 (x, v):
        f(x, v) = (1/(dx*dv)) * sum_p weight * delta(x - x_p) * delta(v - v_p)
    """
    x_bins = np.linspace(np.min(x), np.max(x), n_x_bins + 1)
    v_bins = np.linspace(np.min(v), np.max(v), n_v_bins + 1)

    f, _, _ = np.histogram2d(x, v, bins=[x_bins, v_bins])
    dx = x_bins[1] - x_bins[0]
    dv = v_bins[1] - v_bins[0]
    f *= weight / (dx * dv)

    x_centers = 0.5 * (x_bins[:-1] + x_bins[1:])
    v_centers = 0.5 * (v_bins[:-1] + v_bins[1:])

    return x_centers, v_centers, f
