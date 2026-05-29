"""
碎片质量产额的蒙特卡洛统计模型
================================
融合原始项目:
  - 1327_triangle01_monte_carlo: 单位三角形蒙特卡洛积分

科学背景:
---------
裂变碎片质量分布 Y(A) 可通过多种统计模型计算。
本模块实现基于以下模型的蒙特卡洛采样：

1. 高斯型质量分布（对称裂变主导时）:
   Y(A) = (1/√(2πσ²)) exp[ -(A - A_c)² / (2σ²) ]

2. 双峰分布（不对称裂变）:
   Y(A) = w₁ Y₁(A) + w₂ Y₂(A)
   其中 Y₁, Y₂ 为分别以 A_L 和 A_H 为中心的高斯。

3. 五维相空间积分:
   Y(A) ∝ ∫ dβ₂ dβ₃ dβ₄ dβ₅ dΔ · δ(A - A(β₃))
         · exp[ -V(β₂,β₃,β₄,β₅,Δ) / T ]
         · J(β₂,β₃,β₄,β₅,Δ)
   
   其中 J 为集体坐标到物理空间的 Jacobian 因子。

蒙特卡洛估计:
   Y(A) ≈ (1/N) Σ_{i=1}^N δ(A - A(β₃^i))
   其中 {β^i} 按 exp(-V/T) 重要性采样。

三角形蒙特卡洛方法用于二维截面 (β₂, β₃) 的相空间积分，
这是原始 triangle01_monte_carlo 的核心算法在核物理中的映射。
"""

import numpy as np
from typing import Callable, Tuple


def triangle01_area() -> float:
    """
    单位三角形面积 = 1/2.
    顶点: (0,0), (1,0), (0,1).
    """
    return 0.5


def triangle01_sample(n: int) -> np.ndarray:
    """
    在单位三角形内均匀采样 n 个点.
    
    使用变换法:
    u₁, u₂ ~ Uniform[0,1]
    若 u₁ + u₂ > 1，则映射到 (1-u₁, 1-u₂)
    """
    samples = np.random.rand(n, 2)
    mask = samples[:, 0] + samples[:, 1] > 1.0
    samples[mask] = 1.0 - samples[mask]
    return samples


def triangle_monte_carlo_integral(
    f: Callable[[np.ndarray], np.ndarray],
    n_samples: int = 10000,
) -> float:
    """
    单位三角形上的蒙特卡洛积分 (改编自 triangle01_monte_carlo.m).
    
    ∫∫_Δ f(x,y) dx dy ≈ Area · (1/N) Σ_i f(x_i, y_i)
    """
    area = triangle01_area()
    samples = triangle01_sample(n_samples)
    values = f(samples)
    return area * np.mean(values)


def gaussian_mass_distribution(
    mass_centers: np.ndarray,
    A_peak: float,
    sigma: float,
) -> np.ndarray:
    """
    单峰高斯质量分布.
    
    Y(A) = (1/√(2πσ²)) exp[ -(A - A_peak)² / (2σ²) ]
    """
    if sigma <= 0:
        sigma = 1.0
    return (1.0 / (sigma * np.sqrt(2.0 * np.pi))) * np.exp(
        -0.5 * ((mass_centers - A_peak) / sigma) ** 2
    )


def bimodal_mass_distribution(
    mass_centers: np.ndarray,
    A_light: float,
    A_heavy: float,
    sigma_light: float,
    sigma_heavy: float,
    weight_ratio: float = 1.0,
) -> np.ndarray:
    """
    双峰碎片质量分布（不对称裂变）.
    
    Y(A) = w_L G(A; A_L, σ_L) + w_H G(A; A_H, σ_H)
    其中 w_L + w_H = 1，由 weight_ratio = w_L/w_H 决定.
    """
    if sigma_light <= 0:
        sigma_light = 2.0
    if sigma_heavy <= 0:
        sigma_heavy = 2.0
    
    w_total = 1.0 + weight_ratio
    w_L = weight_ratio / w_total
    w_H = 1.0 / w_total
    
    Y_L = gaussian_mass_distribution(mass_centers, A_light, sigma_light)
    Y_H = gaussian_mass_distribution(mass_centers, A_heavy, sigma_heavy)
    
    return w_L * Y_L + w_H * Y_H


def importance_sampling_mc_yield(
    mass_number: int,
    charge_number: int,
    excitation_energy: float,
    n_samples: int = 50000,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    基于五维相空间重要性采样的碎片质量产额计算.
    
    采样策略：使用 Metropolis-Hastings 算法在集体坐标空间中
    按 Boltzmann 权重 exp(-V/T) 采样。
    """
    from potential_energy_surface import potential_energy
    from collective_coordinates import (
        mass_asymmetry_to_fragment_mass,
        collective_coordinate_bounds,
        clip_to_physical_domain,
    )
    from diffusion_coefficient import nuclear_temperature
    
    T = nuclear_temperature(excitation_energy, mass_number)
    if T < 0.1:
        T = 0.1
    
    bounds = collective_coordinate_bounds(mass_number)
    
    # Metropolis-Hastings 采样
    q_current = np.array([0.3, 0.0, 0.0, 0.0, 1.0])
    q_current = clip_to_physical_domain(q_current, bounds)
    V_current = potential_energy(q_current, mass_number, charge_number)
    
    masses = []
    n_accepted = 0
    n_burn = min(1000, n_samples // 10)
    total_steps = n_samples + n_burn
    
    step_size = np.array([0.1, 0.15, 0.08, 0.04, 0.3])
    
    for step in range(total_steps):
        q_proposal = q_current + step_size * np.random.randn(5)
        q_proposal = clip_to_physical_domain(q_proposal, bounds)
        V_proposal = potential_energy(q_proposal, mass_number, charge_number)
        
        delta_V = V_proposal - V_current
        # Metropolis 接受准则
        accept = False
        if delta_V < 0:
            accept = True
        else:
            if np.random.rand() < np.exp(-delta_V / T):
                accept = True
        
        if accept:
            q_current = q_proposal
            V_current = V_proposal
            n_accepted += 1
        
        if step >= n_burn:
            beta3 = q_current[1]
            A_L, A_H = mass_asymmetry_to_fragment_mass(beta3, mass_number)
            # 对称性：随机选择记录轻或重碎片
            if np.random.rand() < 0.5:
                masses.append(A_L)
            else:
                masses.append(A_H)
    
    masses = np.array(masses)
    
    # 构建直方图
    A_min = max(1.0, mass_number * 0.25)
    A_max = mass_number * 0.75
    n_bins = min(80, n_samples // 200)
    n_bins = max(n_bins, 20)
    bins = np.linspace(A_min, A_max, n_bins + 1)
    counts, edges = np.histogram(masses, bins=bins)
    bin_width = edges[1] - edges[0]
    
    # 归一化
    total_counts = np.sum(counts)
    if total_counts > 0:
        counts = counts / (total_counts * bin_width)
    
    mass_centers = 0.5 * (edges[:-1] + edges[1:])
    return mass_centers, counts


def scission_point_yield_model(
    mass_number: int,
    charge_number: int,
    excitation_energy: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    基于断裂点统计模型的碎片质量产额.
    
    假设在断裂点，质量分布由势能面的曲率决定:
    σ_A² ≈ T / (∂²V/∂β₃²)|_{scission}
    
    双峰位置由势能面在 β₃ 方向的极小值决定。
    """
    from potential_energy_surface import (
        potential_energy_1d,
        fission_barrier_height,
    )
    from diffusion_coefficient import nuclear_temperature
    from collective_coordinates import mass_asymmetry_to_fragment_mass
    
    T = nuclear_temperature(excitation_energy, mass_number)
    barrier = fission_barrier_height(mass_number, charge_number)
    
    # 搜索 β₃ 方向的势能极小值
    beta3_grid = np.linspace(-1.0, 1.0, 200)
    
    # 近似：固定 β₂=1.0（断裂点），计算 V 随 β₃ 的变化
    def V_beta3(b3):
        from potential_energy_surface import potential_energy
        q = np.array([1.0, b3, 0.0, 0.0, 0.0])
        return potential_energy(q, mass_number, charge_number)
    
    V_vals = np.array([V_beta3(b3) for b3 in beta3_grid])
    
    # 寻找双峰结构
    dV = np.gradient(V_vals, beta3_grid)
    ddV = np.gradient(dV, beta3_grid)
    
    # 极小值点: dV=0, ddV>0
    minima_idx = []
    for i in range(1, len(beta3_grid) - 1):
        if dV[i - 1] < 0 and dV[i + 1] > 0 and ddV[i] > 0:
            minima_idx.append(i)
    
    if len(minima_idx) >= 2:
        # 不对称裂变
        idx_L = minima_idx[0]
        idx_H = minima_idx[-1]
        beta3_L = beta3_grid[idx_L]
        beta3_H = beta3_grid[idx_H]
        A_L, _ = mass_asymmetry_to_fragment_mass(beta3_L, mass_number)
        _, A_H = mass_asymmetry_to_fragment_mass(beta3_H, mass_number)
        
        # TODO(Hole_2): 由势能面在 β₃ 方向的曲率计算碎片质量分布的宽度。
        # 科学背景：断裂点处质量分布宽度 σ_A 与势能曲率 κ = ∂²V/∂β₃² 的关系为
        #   σ_β₃ = √(T / κ)   （热涨落导致的形变展宽）
        # 再由 β₃ → A 的映射关系（见 collective_coordinates.mass_asymmetry_to_fragment_mass
        # 中使用的 C3_ASYMMETRY = √(7/4π)），将 σ_β₃ 转换为质量空间宽度 σ_A。
        # 注意：此处的转换系数必须与 collective_coordinates 中的转换公式保持一致。
        raise NotImplementedError("Hole_2: scission_point 质量宽度计算待修复")
        sigma_A_L = 3.0  # 占位
        sigma_A_H = 3.0  # 占位
    else:
        # 对称裂变近似
        A_L = mass_number * 0.4
        A_H = mass_number * 0.6
        sigma_A_L = 3.0
        sigma_A_H = 3.0
    
    mass_centers = np.linspace(mass_number * 0.2, mass_number * 0.8, 100)
    yield_dist = bimodal_mass_distribution(
        mass_centers, A_L, A_H, sigma_A_L, sigma_A_H, weight_ratio=1.0
    )
    
    # 归一化
    total = np.trapezoid(yield_dist, mass_centers)
    if total > 0:
        yield_dist = yield_dist / total
    
    return mass_centers, yield_dist
