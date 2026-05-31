"""
超声造影剂微泡扩散与布朗运动模拟模块

基于种子项目 119_brownian_motion_simulation 的核心算法，
为超声造影成像提供微泡在生物组织中的随机扩散模拟。

物理模型:
微泡在血管中的运动受以下因素支配:
1. 布朗运动: 随机热涨落导致的扩散
2. 声辐射力: 超声波对微泡的推进作用
3. 粘滞阻力: Stokes阻力 F_drag = 6πηrv

扩散系数（Stokes-Einstein关系）:
    D = k_B·T / (6πηr)
其中:
    k_B = 1.38×10⁻²³ J/K（Boltzmann常数）
    T: 绝对温度 (K)
    η: 流体动力粘度 (Pa·s)
    r: 微泡半径 (m)

随机位移的均方根:
    <Δx²> = 2D·Δt  (1D)
    <Δr²> = 6D·Δt  (3D)

超声造影剂微泡的典型参数:
    - 半径: 1-5 μm
    - 壳层: 磷脂或白蛋白
    - 共振频率: 1-10 MHz
"""

import numpy as np
from typing import Tuple, Optional


BOLTZMANN = 1.380649e-23  # J/K
TEMPERATURE_BODY = 310.15  # K (37°C)
VISCOSITY_BLOOD = 3.5e-3   # Pa·s (血液动力粘度)


def diffusion_coefficient(radius: float, temperature: float = TEMPERATURE_BODY,
                          viscosity: float = VISCOSITY_BLOOD) -> float:
    """计算微泡的扩散系数（Stokes-Einstein公式）。
    
    公式: D = k_B·T / (6πηr)
    
    参数:
        radius: 微泡半径 (m)
        temperature: 温度 (K)
        viscosity: 动力粘度 (Pa·s)
    
    返回:
        D: 扩散系数 (m²/s)
    """
    if radius <= 0:
        raise ValueError("半径必须为正")
    if viscosity <= 0:
        raise ValueError("粘度必须为正")
    
    D = BOLTZMANN * temperature / (6.0 * np.pi * viscosity * radius)
    return D


def brownian_step(n_particles: int, dim: int, D: float, dt: float) -> np.ndarray:
    """生成布朗运动的随机步长。
    
    Wiener过程增量:
        dW ~ N(0, √(2D·dt)·I)
    
    每个空间维度的步长:
        Δx = √(2D·dt) · Z,  Z ~ N(0,1)
    
    参数:
        n_particles: 粒子数
        dim: 空间维度 (1, 2, 或 3)
        D: 扩散系数
        dt: 时间步长 (s)
    
    返回:
        step: (n_particles, dim) 随机位移数组
    """
    if dt < 0:
        raise ValueError("时间步长必须非负")
    
    sigma = np.sqrt(2.0 * D * dt)
    step = sigma * np.random.randn(n_particles, dim)
    return step


def simulate_microbubble_diffusion(n_particles: int = 1000,
                                   radius: float = 2.5e-6,
                                   n_steps: int = 1000,
                                   dt: float = 1e-6,
                                   domain_size: float = 5e-3,
                                   temperature: float = TEMPERATURE_BODY,
                                   viscosity: float = VISCOSITY_BLOOD,
                                   acoustic_force: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """模拟微泡在超声场中的扩散运动。
    
    在超声辐射力作用下，微泡运动方程为:
        m·d²r/dt² = F_acoustic + F_Brownian - F_drag
    
    对于过阻尼极限（微泡质量极小），惯性项可忽略:
        dr/dt = (F_acoustic + F_Brownian) / γ
    其中 γ = 6πηr 为Stokes摩擦系数。
    
    参数:
        n_particles: 微泡数量
        radius: 微泡半径 (m)
        n_steps: 时间步数
        dt: 时间步长 (s)
        domain_size: 模拟域边长 (m)
        temperature: 温度 (K)
        viscosity: 动力粘度 (Pa·s)
        acoustic_force: (dim,) 或 (n_steps, dim) 声辐射力 (N)
    
    返回:
        trajectory: (n_steps+1, n_particles, dim) 粒子轨迹
        msd: (n_steps+1,) 均方位移
        D: 扩散系数
    """
    dim = 2  # 2D模拟
    D = diffusion_coefficient(radius, temperature, viscosity)
    
    # 初始化位置：均匀分布在域内
    trajectory = np.zeros((n_steps + 1, n_particles, dim))
    trajectory[0] = np.random.uniform(0, domain_size, (n_particles, dim))
    
    # Stokes摩擦系数
    gamma = 6.0 * np.pi * viscosity * radius
    
    # 处理声辐射力
    if acoustic_force is None:
        force = np.zeros(dim)
    elif acoustic_force.ndim == 1:
        force = acoustic_force
    else:
        force = np.zeros(dim)
    
    for step in range(n_steps):
        # 布朗步长
        brownian = brownian_step(n_particles, dim, D, dt)
        
        # 声辐射力引起的漂移
        if acoustic_force is not None and acoustic_force.ndim == 2:
            force = acoustic_force[step % len(acoustic_force)]
        drift = (force / gamma) * dt
        
        # 更新位置
        trajectory[step + 1] = trajectory[step] + brownian + drift
        
        # 反射边界条件
        for d in range(dim):
            # 下边界反射
            mask_low = trajectory[step + 1, :, d] < 0
            trajectory[step + 1, mask_low, d] = -trajectory[step + 1, mask_low, d]
            
            # 上边界反射
            mask_high = trajectory[step + 1, :, d] > domain_size
            trajectory[step + 1, mask_high, d] = 2 * domain_size - trajectory[step + 1, mask_high, d]
    
    # 计算均方位移 (MSD)
    msd = np.zeros(n_steps + 1)
    initial_pos = trajectory[0]
    for step in range(n_steps + 1):
        displacements = trajectory[step] - initial_pos
        squared_displacements = np.sum(displacements**2, axis=1)
        msd[step] = np.mean(squared_displacements)
    
    return trajectory, msd, D


def acoustic_radiation_force(frequency: float = 5e6,
                              pressure_amplitude: float = 1e5,
                              bubble_radius: float = 2.5e-6,
                              c0: float = 1540.0,
                              rho0: float = 1000.0) -> float:
    """计算超声辐射力（Bjerknes力）。
    
    初级Bjerknes力（时间平均）:
        F = -V·∇<p²> / (2ρ₀c₀²)
    
    对于平面波近似:
        F ≈ 4πr³·k·p₀² / (3ρ₀c₀²)
    
    其中 k = 2πf/c₀ 为波数。
    
    参数:
        frequency: 超声频率 (Hz)
        pressure_amplitude: 声压幅值 (Pa)
        bubble_radius: 微泡半径 (m)
        c0: 声速 (m/s)
        rho0: 介质密度 (kg/m³)
    
    返回:
        F: 辐射力幅值 (N)
    """
    k = 2.0 * np.pi * frequency / c0
    volume = 4.0 / 3.0 * np.pi * bubble_radius**3
    
    F = volume * k * pressure_amplitude**2 / (3.0 * rho0 * c0**2)
    return F


def concentration_profile(trajectory: np.ndarray, domain_size: float,
                          n_bins: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    """计算微泡浓度的空间分布。
    
    用于与超声成像的回波强度进行关联分析。
    
    参数:
        trajectory: (n_steps+1, n_particles, dim) 轨迹数组
        domain_size: 域边长
        n_bins: 直方图bins数
    
    返回:
        bin_edges: 边界坐标
        concentration: 浓度分布
    """
    final_positions = trajectory[-1]  # 最终时刻的位置
    
    # 2D直方图
    hist, xedges, yedges = np.histogram2d(
        final_positions[:, 0], final_positions[:, 1],
        bins=n_bins, range=[[0, domain_size], [0, domain_size]]
    )
    
    # 归一化为浓度
    bin_area = (domain_size / n_bins)**2
    concentration = hist / bin_area
    
    return xedges, yedges, concentration
