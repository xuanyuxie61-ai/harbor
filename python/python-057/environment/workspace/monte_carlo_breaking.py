"""
monte_carlo_breaking.py
基于蒙特卡洛方法的内波破碎概率模拟

融合项目:
- 438_flies_simulation: 圆盘上的随机距离 → 随机相位内波叠加
- 137_casino_simulation: 乘法随机过程 → 能量级联的随机衰减
- 655_leaf_chaos: 迭代函数系统(IFS) → 混合斑块的分形结构

核心科学:
海洋内波破碎是一个随机过程，受多种因素控制:
1. 随机相位叠加导致的局部剪切增强
2. 能量级联中的随机耗散
3. 混合斑块的分形分布

数学模型:
1. 随机相位叠加:
    u(z,t) = Σ A_k · sin(k_m z + φ_k) · cos(ω_k t + θ_k)
    
2. 能量级联 (乘法随机过程):
    E_{n+1} = E_n · X_n
    其中 X_n 为随机能量转移系数
    
3. 混合斑块分形:
    使用IFS生成混合效率的空间分布
    
破碎判据:
    当局部Ri < 0.25 或 波陡 ε > ε_c 时发生破碎
"""

import numpy as np


def random_phase_superposition(n_modes=20, z=None, t=0.0,
                                N=0.01, f=1.0e-4):
    """
    随机相位内波模态叠加
    
    内波速度场表示为多个模态的随机叠加:
        u(z,t) = Σ_{m=1}^{M} A_m · sin(mπ z/H) · cos(ω_m t + θ_m)
    
    参数:
        n_modes: 模态数量
        z: 深度坐标 [m]
        t: 时间 [s]
        N: 浮力频率 [rad/s]
        f: 科里奥利参数 [rad/s]
    
    返回:
        u: 水平速度 [m/s]
        shear: 垂向剪切 [1/s]
        Ri: Richardson数
    """
    if z is None:
        z = np.linspace(-200, 0, 101)
    
    z = np.asarray(z)
    H = np.max(z) - np.min(z)  # 水深
    
    # 模态振幅 (Garrett-Munk谱)
    m = np.arange(1, n_modes + 1)
    A_m = 0.5 / m  # 振幅随模态衰减
    
    # 随机相位
    theta_m = np.random.uniform(0, 2.0 * np.pi, n_modes)
    phi_m = np.random.uniform(0, 2.0 * np.pi, n_modes)
    
    # 频率 (内波色散关系近似)
    k_h = 2.0 * np.pi / 1000.0  # 固定水平波数
    k_m = m * np.pi / H  # 垂向波数
    # HOLE 2: 计算内波模态频率 omega_m
    # 提示: 应使用与 ocean_physics.internal_wave_dispersion 一致的色散关系
    # ω_m = sqrt( (N²·k_h² + f²·k_m²) / (k_h² + k_m²) )
    # 建议: 从 ocean_physics 导入并调用 internal_wave_dispersion
    raise NotImplementedError("待实现: 内波模态频率计算")
    
    # 速度场叠加
    u = np.zeros_like(z)
    for i in range(n_modes):
        u += A_m[i] * np.sin(k_m[i] * (z - np.min(z))) * \
             np.cos(omega_m[i] * t + theta_m[i])
    
    # 垂向剪切
    shear = np.zeros_like(z)
    for i in range(n_modes):
        shear += A_m[i] * k_m[i] * np.cos(k_m[i] * (z - np.min(z))) * \
                 np.cos(omega_m[i] * t + theta_m[i])
    
    # Richardson数
    N2 = N**2
    shear_sq = shear**2
    Ri = np.where(shear_sq > 1.0e-12, N2 / shear_sq, 1.0e6)
    Ri = np.clip(Ri, 0.0, 100.0)
    
    return u, shear, Ri


def monte_carlo_breaking_probability(n_realizations=1000,
                                      n_modes=20,
                                      n_depths=101,
                                      N=0.01):
    """
    蒙特卡洛估计内波破碎概率
    
    通过多次随机相位实现，统计满足破碎判据的比例:
        P_break = N_break / N_total
    
    参数:
        n_realizations: 随机实现次数
        n_modes: 每实现的模态数
        n_depths: 深度分辨率
        N: 浮力频率 [rad/s]
    
    返回:
        P_break: 破碎概率
        P_break_z: 深度依赖的破碎概率
        z: 深度坐标
    """
    z = np.linspace(-200, 0, n_depths)
    n_break = np.zeros(n_depths)
    
    for _ in range(n_realizations):
        _, shear, Ri = random_phase_superposition(n_modes, z, t=0.0, N=N)
        
        # 破碎判据: Ri < 0.25
        breaking_mask = Ri < 0.25
        n_break += breaking_mask.astype(float)
    
    P_break_z = n_break / n_realizations
    P_break = np.mean(P_break_z)
    
    return P_break, P_break_z, z


def energy_cascade_simulation(E0=1.0, n_steps=1000,
                               growth_factor=1.05,
                               dissipation_factor=0.97):
    """
    内波能量级联的乘法随机过程模拟
    
    受赌场模拟(casino_simulation)启发，能量级联可建模为:
        E_{n+1} = E_n · ξ_n
    
    其中 ξ_n 为随机能量转移系数:
        ξ = growth_factor (概率 p)
        ξ = dissipation_factor (概率 1-p)
    
    参数:
        E0: 初始能量
        n_steps: 步数
        growth_factor: 能量增长因子
        dissipation_factor: 能量耗散因子
    
    返回:
        E_history: 能量历史
        breaking_events: 破碎事件索引
    """
    E_history = np.zeros(n_steps)
    E = E0
    E_history[0] = E
    
    # 破碎阈值: 能量超过临界值
    E_critical = 5.0 * E0
    breaking_events = []
    
    # 随机转移概率 (偏向耗散)
    p_growth = 0.45
    
    for n in range(1, n_steps):
        coin = np.random.rand()
        
        if coin < p_growth:
            E *= growth_factor
        else:
            E *= dissipation_factor
        
        # 边界处理
        E = max(E, 1.0e-6)
        
        E_history[n] = E
        
        # 检测破碎
        if E > E_critical:
            breaking_events.append(n)
            # 破碎后能量重置
            E = E0 * 0.5
    
    return E_history, breaking_events


def mixing_patch_ifs(n_points=5000, n_iterations=10):
    """
    使用迭代函数系统(IFS)生成分形混合斑块分布
    
    基于leaf_chaos项目的IFS思想，构建混合效率的分形分布:
    
    4个仿射变换:
        x' = A_i · x + b_i
    
    变换概率: p = [0.3, 0.3, 0.25, 0.15]
    
    参数:
        n_points: 生成点数
        n_iterations: 初始迭代次数 (丢弃)
    
    返回:
        x, y: 混合斑块坐标 (归一化)
        intensity: 混合强度
    """
    # 仿射变换矩阵 (4个变换)
    A = [
        np.array([[0.5, 0.0], [0.0, 0.5]]),
        np.array([[0.5, 0.0], [0.0, 0.5]]),
        np.array([[0.4, 0.1], [-0.1, 0.4]]),
        np.array([[0.3, -0.2], [0.2, 0.3]]),
    ]
    
    b = [
        np.array([0.0, 0.0]),
        np.array([0.5, 0.0]),
        np.array([0.25, 0.5]),
        np.array([0.6, 0.4]),
    ]
    
    p = [0.3, 0.3, 0.25, 0.15]
    
    # 初始点
    point = np.random.rand(2)
    
    # 预热迭代
    for _ in range(n_iterations):
        idx = np.random.choice(4, p=p)
        point = A[idx] @ point + b[idx]
    
    # 生成点云
    points = np.zeros((n_points, 2))
    intensities = np.zeros(n_points)
    
    for i in range(n_points):
        idx = np.random.choice(4, p=p)
        point = A[idx] @ point + b[idx]
        points[i, :] = point
        
        # 混合强度与变换相关
        intensities[i] = 0.5 + 0.5 * idx / 3.0
    
    # 边界处理
    points[:, 0] = np.clip(points[:, 0], 0.0, 1.0)
    points[:, 1] = np.clip(points[:, 1], 0.0, 1.0)
    
    return points[:, 0], points[:, 1], intensities
