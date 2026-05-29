"""
thermodynamics.py
=================
分子动力学热力学量计算：比热、弹性常数、热膨胀系数等。

核心物理公式
------------
定容热容（能量涨落法）：

    C_V = (⟨E²⟩ - ⟨E⟩²) / (k_B T²)

对于 N 粒子经典系统，能量均分定理给出：

    C_V = (d·N/2) · k_B       （理想气体极限）

弹性常数（应力-应变关系）：

    C_{αβγδ} = (1/Ω) · ∂²E/∂ε_{αβ}∂ε_{γδ}|_{ε=0}

其中 Ω 为体积，ε 为应变张量。

对于各向同性二维系统，简化为：

    C_{11} = C_{22} = (1/Ω) · ∂²E/∂ε_{xx}²
    C_{12} = (1/Ω) · ∂²E/∂ε_{xx}∂ε_{yy}

热膨胀系数（通过双调和方程耦合）：

    α_L = (1/L) · dL/dT

热压系数：

    β = (1/P) · dP/dT

Green-Kubo 关系（输运性质）：

    κ = (1/(k_B T² Ω)) ∫_0^∞ ⟨J(t)·J(0)⟩ dt
"""

import numpy as np
from typing import Tuple
from potential_models import total_potential_lj


def heat_capacity_cv(energies: np.ndarray, temperature: float,
                     n_particles: int, dim: int = 2,
                     k_boltzmann: float = 1.0) -> Tuple[float, float]:
    """
    从能量时间序列计算定容热容 C_V。
    
    公式: C_V = var(E) / (k_B T²)
    
    参数:
        energies: 总能量时间序列
        temperature: 系统温度
        n_particles: 粒子数
        dim: 空间维度
        k_boltzmann: 玻尔兹曼常数（自然单位制下取 1）
    
    返回:
        (C_V, C_V_dulong) 其中 C_V_dulong = d·N·k_B/2
    """
    # TODO: 实现定容热容计算
    # 提示: 使用能量涨落法 C_V = var(E) / (k_B T²)
    raise NotImplementedError("Hole_3: 请补全 heat_capacity_cv 的热力学公式实现")


def elastic_constants_from_fluctuations(stress_history: np.ndarray,
                                        volume: float,
                                        temperature: float,
                                        k_boltzmann: float = 1.0) -> np.ndarray:
    """
    使用应力涨落计算弹性常数矩阵（2D 各向同性近似）。
    
    对于 2D：
        C_{11} = (1/Ω) [ ⟨σ_{xx}²⟩ - ⟨σ_{xx}⟩² ] / (k_B T) + ...
    
    更精确地，使用应力自关联的零频响应：
        C_{αβγδ} = ⟨σ_{αβ}⟩⟨σ_{γδ}⟩/(Nk_BT) + ...
    
    这里采用简化版本：直接拟合应力-应变关系。
    """
    if len(stress_history) < 2:
        return np.eye(2)

    # 提取应力分量（假设 stress_history 存储的是标量压强或迹）
    # 如果输入是标量压强，扩展为各向同性应力张量
    if stress_history.ndim == 1:
        # 简化为各向同性情况
        mean_p = np.mean(stress_history)
        var_p = np.var(stress_history, ddof=1)
        # 压缩模量估计
        bulk_modulus = volume * var_p / (k_boltzmann * temperature) if temperature > 1e-12 else 0.0
        C = np.array([[bulk_modulus, 0.0],
                      [0.0, bulk_modulus]])
    else:
        # 2×2 应力张量时间序列 (n_steps, 2, 2)
        n_steps = stress_history.shape[0]
        sigma = np.mean(stress_history, axis=0)
        # 计算涨落
        delta_sigma = stress_history - sigma
        # 弹性常数张量 C_{αβγδ}
        C = np.zeros((2, 2, 2, 2))
        for alpha in range(2):
            for beta in range(2):
                for gamma in range(2):
                    for delta in range(2):
                        cov = np.mean(delta_sigma[:, alpha, beta] *
                                       delta_sigma[:, gamma, delta])
                        if temperature > 1e-12:
                            C[alpha, beta, gamma, delta] = cov * volume / (k_boltzmann * temperature)
        # 压缩为 2×2 Voigt 表示
        C_voigt = np.zeros((2, 2))
        C_voigt[0, 0] = C[0, 0, 0, 0]
        C_voigt[1, 1] = C[1, 1, 1, 1]
        C_voigt[0, 1] = C[0, 0, 1, 1]
        C_voigt[1, 0] = C_voigt[0, 1]
        C = C_voigt
    return C


def elastic_constants_strain_derivative(positions: np.ndarray,
                                         epsilon: float = 1.0,
                                         sigma: float = 1.0,
                                         rcut: float = 2.5,
                                         volume: float = 1.0,
                                         strain_perturbation: float = 1e-4) -> np.ndarray:
    """
    通过有限应变法计算弹性常数。
    
    施加微小应变 ε，计算能量变化：
        E(ε) = E(0) + Ω·σ·ε + (Ω/2)·ε·C·ε + O(ε³)
    
    对于 2D 各向同性系统：
        C_{11} = (1/Ω) · [E(ε_xx) - 2E(0) + E(-ε_xx)] / ε_xx²
    """
    n, d = positions.shape
    C = np.zeros((d, d))
    eps = strain_perturbation

    # 未应变能量
    e0 = total_potential_lj(positions, epsilon, sigma, rcut)

    for alpha in range(d):
        # 沿 x_α 方向施加拉伸
        pos_plus = positions.copy()
        pos_minus = positions.copy()
        for i in range(n):
            pos_plus[i, alpha] *= (1.0 + eps)
            pos_minus[i, alpha] *= (1.0 - eps)

        e_plus = total_potential_lj(pos_plus, epsilon, sigma, rcut)
        e_minus = total_potential_lj(pos_minus, epsilon, sigma, rcut)

        C[alpha, alpha] = (e_plus - 2.0 * e0 + e_minus) / (eps ** 2 * volume)

    return C


def radial_distribution_function(positions: np.ndarray,
                                  box_size: float,
                                  n_bins: int = 50,
                                  rcut: float = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算径向分布函数 g(r)。
    
    g(r) = (V / N²) · ⟨ Σ_i Σ_{j≠i} δ(r - r_{ij}) / (2πr·Δr)⟩   (2D)
    
    参数:
        positions: N×d 坐标
        box_size: 模拟盒子边长
        n_bins: 直方图分箱数
        rcut: 最大距离（默认 box_size/2）
    
    返回:
        (r_bins, g_r)
    """
    n, d = positions.shape
    if rcut is None:
        rcut = box_size / 2.0

    dr = rcut / n_bins
    r_bins = np.linspace(0.5 * dr, rcut - 0.5 * dr, n_bins)
    hist = np.zeros(n_bins)

    for i in range(n):
        for j in range(i + 1, n):
            rij = positions[j] - positions[i]
            # 最小镜像约定
            rij -= box_size * np.round(rij / box_size)
            r = np.linalg.norm(rij)
            if r < rcut:
                bin_idx = int(r / dr)
                if 0 <= bin_idx < n_bins:
                    hist[bin_idx] += 2.0  # i-j 和 j-i 各计一次

    # 归一化
    volume = box_size ** d
    shell_volumes = np.zeros(n_bins)
    for k in range(n_bins):
        r_inner = k * dr
        r_outer = (k + 1) * dr
        if d == 2:
            shell_volumes[k] = np.pi * (r_outer ** 2 - r_inner ** 2)
        elif d == 3:
            shell_volumes[k] = (4.0 / 3.0) * np.pi * (r_outer ** 3 - r_inner ** 3)
        else:
            shell_volumes[k] = 2.0 * (r_outer - r_inner)  # 1D

    # g(r) = V/(N(N-1)) * hist / shell_volume
    norm = volume / (n * (n - 1)) if n > 1 else 1.0
    g_r = norm * hist / (shell_volumes + 1e-30)
    return r_bins, g_r


def thermal_expansion_estimate(lengths: np.ndarray,
                                temperatures: np.ndarray) -> float:
    """
    线性热膨胀系数估计（最小二乘法）。
    
    α_L = (1/L₀) · dL/dT
    
    假设 L(T) ≈ L₀(1 + α_L · T)
    """
    if len(lengths) < 2 or len(temperatures) < 2:
        return 0.0
    # 线性拟合
    T_mean = np.mean(temperatures)
    L_mean = np.mean(lengths)
    numerator = np.sum((temperatures - T_mean) * (lengths - L_mean))
    denominator = np.sum((temperatures - T_mean) ** 2)
    if abs(denominator) < 1e-30:
        return 0.0
    dL_dT = numerator / denominator
    L0 = L_mean - dL_dT * T_mean
    if abs(L0) < 1e-30:
        L0 = 1e-30
    return dL_dT / L0


def entropy_from_energy_distribution(energies: np.ndarray,
                                      temperature: float,
                                      n_bins: int = 20,
                                      k_boltzmann: float = 1.0) -> float:
    """
    从能量分布估计熵（基于直方图法）。
    
    S = -k_B Σ_i p_i log(p_i)
    
    其中 p_i 为能量落在第 i 个区间的概率。
    """
    if len(energies) < 2:
        return 0.0
    e_min, e_max = np.min(energies), np.max(energies)
    if abs(e_max - e_min) < 1e-30:
        return 0.0

    counts, _ = np.histogram(energies, bins=n_bins, range=(e_min, e_max))
    probs = counts / len(energies)
    # 只考虑非零概率
    probs = probs[probs > 1e-30]
    entropy = -k_boltzmann * np.sum(probs * np.log(probs))
    return entropy
