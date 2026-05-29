"""
核裂变构型空间的质心 Voronoi 镶嵌划分
======================================
融合原始项目:
  - 239_cvt_1_movie: 质心 Voronoi 镶嵌 (CVT) 迭代算法

科学背景:
---------
在多维集体坐标空间中，裂变构型可以划分为 Voronoi 单元:
  C_i = { q ∈ ℝ⁵ : ||q - z_i|| ≤ ||q - z_j||, ∀j≠i }
其中 {z_i} 为生成点（generator）。

质心 Voronoi 镶嵌 (CVT) 要求每个生成点 z_i 恰好是其对应
单元 C_i 的质心：
  z_i = ∫_{C_i} q ρ(q) dq / ∫_{C_i} ρ(q) dq

在核物理中，密度函数取为 Boltzmann 权重:
  ρ(q) = exp( -V(q) / T )

CVT 划分的优势：
1. 最优量化：最小化能量泛函
   E({z_i}) = Σ_i ∫_{C_i} ||q - z_i||² ρ(q) dq
2. 自适应网格：在势能低谷处自动聚集更多单元
3. 降维分析：每个单元可视为一个"微观态"

本模块实现基于 Lloyd 算法的 CVT 迭代，
用于集体坐标空间的自适应离散化。
"""

import numpy as np
from typing import Tuple


def cvt_iterate_2d(
    generators: np.ndarray,
    n_samples: int,
    density_func,
    bounds: dict,
    max_iter: int = 50,
    tol: float = 1e-4,
) -> Tuple[np.ndarray, float, float]:
    """
    2D 集体坐标空间 (β₂, β₃) 的 CVT 迭代 (改编自 cvt_iterate.m).
    
    参数:
        generators: (n, 2) 初始生成点数组
        n_samples: 每轮采样数
        density_func: 密度函数 ρ(β₂, β₃)
        bounds: 坐标范围字典
        max_iter: 最大迭代次数
        tol: 收敛容差
    返回:
        (优化后的生成点, 总位移, 离散能量)
    """
    n_gen = len(generators)
    if n_gen < 1:
        raise ValueError("need at least one generator")
    
    ndim = 2
    z = generators.copy()
    b2_min, b2_max = bounds['beta2']
    b3_min, b3_max = bounds['beta3']
    
    for iteration in range(max_iter):
        # 按密度重要性采样
        samples = np.zeros((n_samples, ndim))
        accepted = 0
        max_attempts = n_samples * 100
        attempts = 0
        
        while accepted < n_samples and attempts < max_attempts:
            attempts += 1
            s = np.array([
                np.random.uniform(b2_min, b2_max),
                np.random.uniform(b3_min, b3_max),
            ])
            rho = density_func(s)
            # 接受-拒绝采样
            if np.random.rand() < rho:
                samples[accepted] = s
                accepted += 1
        
        if accepted < n_samples // 2:
            # fallback: 均匀采样
            samples = np.random.rand(n_samples, 2)
            samples[:, 0] = b2_min + samples[:, 0] * (b2_max - b2_min)
            samples[:, 1] = b3_min + samples[:, 1] * (b3_max - b3_min)
        else:
            samples = samples[:accepted]
        
        # 寻找每个样本的最近生成点
        z_new = np.zeros_like(z)
        counts = np.zeros(n_gen)
        energy = 0.0
        
        for s in samples:
            # 最近邻搜索
            dists = np.sum((z - s) ** 2, axis=1)
            nearest = np.argmin(dists)
            z_new[nearest] += s
            counts[nearest] += 1
            energy += dists[nearest]
        
        # 计算新质心
        for j in range(n_gen):
            if counts[j] > 0:
                z_new[j] /= counts[j]
            else:
                z_new[j] = z[j]  # 未收到样本则保持原位
        
        # 裁剪到物理域
        z_new[:, 0] = np.clip(z_new[:, 0], b2_min, b2_max)
        z_new[:, 1] = np.clip(z_new[:, 1], b3_min, b3_max)
        
        # 计算位移
        diff = np.sum(np.sqrt(np.sum((z_new - z) ** 2, axis=1)))
        energy = energy / len(samples) if len(samples) > 0 else 0.0
        
        z = z_new
        
        if diff < tol:
            break
    
    return z, diff, energy


def cvt_partition_fission_space(
    mass_number: int,
    charge_number: int,
    excitation_energy: float,
    n_generators: int = 20,
    n_samples: int = 5000,
) -> Tuple[np.ndarray, float, float]:
    """
    对裂变构型空间 (β₂, β₃) 进行 CVT 自适应划分.
    
    返回优化后的生成点坐标及收敛信息.
    """
    from potential_energy_surface import potential_energy
    from diffusion_coefficient import nuclear_temperature
    from collective_coordinates import collective_coordinate_bounds
    
    T = nuclear_temperature(excitation_energy, mass_number)
    if T < 0.1:
        T = 0.1
    
    bounds = collective_coordinate_bounds(mass_number)
    
    # 初始生成点：均匀分布
    b2_min, b2_max = bounds['beta2']
    b3_min, b3_max = bounds['beta3']
    
    generators = np.zeros((n_generators, 2))
    generators[:, 0] = np.random.uniform(b2_min, b2_max, n_generators)
    generators[:, 1] = np.random.uniform(b3_min, b3_max, n_generators)
    
    def density_func(s):
        q = np.array([s[0], s[1], 0.0, 0.0, 0.0])
        V = potential_energy(q, mass_number, charge_number)
        rho = np.exp(-V / T)
        return float(np.clip(rho, 0.0, 1.0))
    
    z_opt, diff, energy = cvt_iterate_2d(
        generators, n_samples, density_func, bounds, max_iter=30, tol=1e-3
    )
    
    return z_opt, diff, energy
