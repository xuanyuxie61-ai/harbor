"""
photon_statistics.py - 单光子发射统计与蒙特卡洛模拟模块

融合原项目 1046_roulette_simulation（随机数生成与概率模拟）、
292_disk_distance（圆盘内均匀采样）与
713_maple_area（基于网格的面积估计）的核心算法，
用于计算量子点单光子源的二阶关联函数 g^(2)(tau) 与探测统计。

核心物理公式：
    - 二阶关联函数：
        g^(2)(tau) = <I(t) I(t+tau)> / <I(t)>^2
                   = <a^dagger(t) a^dagger(t+tau) a(t+tau) a(t)> / <a^dagger a>^2
    - 理想单光子源：g^(2)(0) -> 0
    - 泊松光源：g^(2)(0) = 1
    - 热光源：g^(2)(0) = 2
    - Hanbury Brown-Twiss 实验等效：
        g^(2)(tau) = 1 - exp(-(gamma_dot + kappa) |tau|)
"""

import numpy as np
from typing import Tuple, Dict
from utils import validate_array_1d


def disk_unit_sample(n: int = 1) -> np.ndarray:
    """
    在单位圆盘内均匀采样（源自 disk_unit_sample）：
    
        theta ~ Uniform(0, 2pi)
        r ~ sqrt(Uniform(0, 1))
    
    物理意义：模拟量子点中电偶极子的随机取向投影。
    """
    theta = 2.0 * np.pi * np.random.rand(n)
    r = np.sqrt(np.random.rand(n))
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.vstack([x, y])


def disk_distance_stats(n_samples: int = 10000) -> Tuple[float, float]:
    """
    估计单位圆盘内两随机点之间距离的均值与方差（源自 disk_distance_stats）。
    
    理论值：均值 = 128 / (45 pi) ≈ 0.9054
    """
    distances = np.zeros(n_samples, dtype=float)
    for i in range(n_samples):
        p = disk_unit_sample(1).ravel()
        q = disk_unit_sample(1).ravel()
        distances[i] = np.linalg.norm(p - q)
    mu = float(np.mean(distances))
    if n_samples > 1:
        var = float(np.var(distances, ddof=1))
    else:
        var = 0.0
    return mu, var


def second_order_correlation_weak_coupling(
    tau: np.ndarray,
    gamma_dot: float,
    kappa: float,
    g_coupling: float,
) -> np.ndarray:
    """
    弱耦合极限下二能级量子点-腔系统的二阶关联函数解析近似：
    
        g^(2)(tau) = 1 - exp( - (gamma_dot + kappa) |tau| )
                     + (g / (gamma_dot + kappa))^2 * ...
    
    此处采用简化模型（Purcell 增强的自发辐射）：
        g^(2)(tau) = 1 - exp( - Gamma_purcell |tau| )
    
    其中 Gamma_purcell = gamma_dot * F_p + kappa。
    为体现非理想单光子特性，加入弱背景项 b：
        g^(2)(0) = b / (1 + b)^2
    """
    tau = np.asarray(tau, dtype=float)
    if gamma_dot < 0 or kappa < 0:
        raise ValueError("Decay rates must be non-negative")
    Gamma = gamma_dot + kappa
    if Gamma < 1e-15:
        Gamma = 1e-15
    g2 = 1.0 - np.exp(-Gamma * np.abs(tau))
    # 加入真空 Rabi 振荡修正（当 g 不可忽略时）
    if g_coupling > 0.01 * Gamma:
        Omega = np.sqrt(np.maximum(g_coupling ** 2 - 0.25 * Gamma ** 2, 0.0))
        if Omega > 1e-15:
            g2 += 0.2 * (g_coupling / Gamma) ** 2 * np.cos(Omega * tau) * np.exp(-0.5 * Gamma * np.abs(tau))
    # 确保非负
    g2 = np.maximum(g2, 0.0)
    return g2


def antibunching_parameter(g2_0: float) -> str:
    """
    根据 g^(2)(0) 值判定光源类型：
        g^(2)(0) < 0.5   -> 强反聚束（单光子源）
        0.5 <= g^(2)(0) < 1.0 -> 弱反聚束
        g^(2)(0) >= 1.0  -> 聚束或泊松光源
    """
    if g2_0 < 0.5:
        return "strong_antibunching"
    elif g2_0 < 1.0:
        return "weak_antibunching"
    else:
        return "bunching_or_poissonian"


def monte_carlo_photon_detection(
    emission_rate: float,
    detection_efficiency: float,
    gate_time: float,
    n_trials: int = 10000,
) -> Dict[str, float]:
    """
    蒙特卡洛模拟光子探测统计（源自 roulette_simulation 的随机过程思想）。
    
    模型：
        - 发射过程：泊松过程，平均光子数 mu = rate * gate_time * efficiency
        - 探测器：二元结果（探测到 / 未探测到）
        - 多次试验统计探测概率分布
    
    返回统计量：
        mean_counts, variance, p_zero, p_single, p_multi
    """
    if emission_rate < 0 or detection_efficiency < 0 or gate_time < 0:
        raise ValueError("Physical parameters must be non-negative")
    if detection_efficiency > 1.0:
        detection_efficiency = 1.0
    mu = emission_rate * gate_time * detection_efficiency
    counts = np.random.poisson(mu, size=n_trials)
    mean_counts = float(np.mean(counts))
    var_counts = float(np.var(counts, ddof=1))
    p_zero = float(np.mean(counts == 0))
    p_single = float(np.mean(counts == 1))
    p_multi = float(np.mean(counts >= 2))
    return {
        "mean_counts": mean_counts,
        "variance": var_counts,
        "p_zero": p_zero,
        "p_single": p_single,
        "p_multi": p_multi,
        "singles_fraction": p_single / (1.0 - p_zero + 1e-15),
    }


def simulate_hanbury_brown_twiss(
    gamma_dot: float,
    kappa: float,
    g_coupling: float,
    measurement_time: float,
    n_bins: int = 200,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    模拟 Hanbury Brown-Twiss 实验的时间分辨符合计数 histogram。
    
    采用理论模型生成符合计数 histogram：
        C(tau) ~ g^(2)(tau) * exp(-Gamma tau)  (归一化)
    """
    if measurement_time <= 0:
        raise ValueError("measurement_time must be positive")
    tau_vals = np.linspace(-measurement_time, measurement_time, n_bins)
    g2_vals = second_order_correlation_weak_coupling(tau_vals, gamma_dot, kappa, g_coupling)
    # 加入泊松噪声模拟实验统计涨落
    noise = np.random.normal(0.0, 0.02 * np.max(g2_vals), size=n_bins)
    counts = np.maximum(g2_vals + noise, 0.0)
    return tau_vals, counts


def photon_indistinguishability_homodyne(
    pure_state_overlap: float,
    dephasing_rate: float,
    pulse_duration: float,
) -> float:
    """
    估算 HOM 干涉可见度（两光子不可区分度）：
    
        V = |<psi_1 | psi_2>|^2 * exp(-2 gamma_dephase * tau_pulse)
    
    对于理想单光子源，V -> 1。
    """
    if pure_state_overlap < 0 or pure_state_overlap > 1:
        raise ValueError("Overlap must be in [0, 1]")
    if dephasing_rate < 0 or pulse_duration < 0:
        raise ValueError("Rates and durations must be non-negative")
    V = (pure_state_overlap ** 2) * np.exp(-2.0 * dephasing_rate * pulse_duration)
    return float(V)


def detection_area_efficiency(
    detector_radius: float,
    emission_waist: float,
    distance: float,
) -> float:
    """
    基于几何光学的探测器接收效率估算：
    
        eta = A_detector / (pi (w(z))^2)
    
    其中光束在距离 z 处的半径：
        w(z) = w0 * sqrt(1 + (z / z_R)^2)
        z_R = pi w0^2 / lambda
    
    为简化，假设高斯光束在探测器处束腰为 w(z)。
    """
    if detector_radius <= 0 or emission_waist <= 0 or distance < 0:
        raise ValueError("Geometric parameters must be positive")
    # 简化：探测器面积与光束截面面积之比
    beam_radius = emission_waist * np.sqrt(1.0 + (distance / (np.pi * emission_waist ** 2 / 780e-9)) ** 2)
    eta = (detector_radius ** 2) / (beam_radius ** 2)
    return float(min(eta, 1.0))
