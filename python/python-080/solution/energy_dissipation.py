"""
energy_dissipation.py
气泡崩溃能量耗散路径优化

核心物理模型:
1. 气泡崩溃总能量:
   E_total = E_kinetic + E_potential + E_surface + E_acoustic

2. 动能:
   E_k = 2π ρ R³ (dR/dt)²

3. 势能（压力功）:
   E_p = (4/3)π R³ (p_∞ - p_v)

4. 表面能:
   E_s = 4π σ R²

5. 声能耗散:
   E_acoustic = ∫_0^t (p_wall² / (ρ c)) * 4π R² dτ

6. 能量耗散路径优化:
   将气泡崩溃过程中的能量分配视为一个组合优化问题。
   将时间离散为 N 个阶段，每个阶段选择能量流向（动能/声能/热耗散）。
   使用随机搜索（类比 TSP）寻找近似最优路径。

映射来源:
- 1367_tsp_random: 随机搜索旅行商问题 → 能量分配路径优化
"""

import numpy as np
from numpy.linalg import norm


def bubble_energy_budget(R, dRdt, p_inf, p_v, sigma, rho, c_sound):
    """
    计算气泡各能量分量。

    参数:
        R: 当前半径 [m]
        dR/dt: 径向速度 [m/s]
    返回:
        energies: 字典，包含各能量分量 [J]
    """
    # 动能
    E_k = 2.0 * np.pi * rho * (R ** 3) * (dRdt ** 2)

    # 势能（压力体积功）
    E_p = (4.0 / 3.0) * np.pi * (R ** 3) * (p_inf - p_v)

    # 表面能
    E_s = 4.0 * np.pi * sigma * (R ** 2)

    # 声能辐射功率
    # 近似: dE_acoustic/dt = (p_wall)² * 4πR² / (ρc)
    p_wall = p_v - 2.0 * sigma / R - 4.0 * 1.002e-3 * dRdt / R
    P_acoustic = (p_wall ** 2) * 4.0 * np.pi * (R ** 2) / (rho * c_sound + 1e-30)

    return {
        'kinetic': E_k,
        'potential': E_p,
        'surface': E_s,
        'acoustic_power': P_acoustic,
        'total': E_k + E_p + E_s
    }


def energy_dissipation_path(R_history, dRdt_history, dt, p_inf, p_v, sigma, rho, c_sound):
    """
    计算气泡崩溃全过程的能量耗散时间序列。

    参数:
        R_history: 半径历史数组
        dRdt_history: 速度历史数组
        dt: 时间步长
    返回:
        energy_history: 每个时间步的能量分量字典列表
    """
    energy_history = []
    cumulative_acoustic = 0.0

    for R, dRdt in zip(R_history, dRdt_history):
        energies = bubble_energy_budget(R, dRdt, p_inf, p_v, sigma, rho, c_sound)
        cumulative_acoustic += energies['acoustic_power'] * dt
        energies['cumulative_acoustic'] = cumulative_acoustic
        energy_history.append(energies)

    return energy_history


def collapse_efficiency(R0, R_min, p_inf, p_v, sigma, rho, c_sound):
    """
    计算气泡崩溃效率。
    η = E_kinetic_at_Rmin / E_total_initial

    理想球形崩溃的近似解析解（Rayleigh 解）:
    t_collapse ≈ 0.915 R0 * sqrt(ρ / (p_∞ - p_v))
    E_kinetic_max ≈ (4/3)π R0³ (p_∞ - p_v)
    """
    E_initial = bubble_energy_budget(R0, 0.0, p_inf, p_v, sigma, rho, c_sound)
    E_total_0 = E_initial['total']

    # 近似最大动能（忽略表面张力和粘性）
    E_k_max_approx = (4.0 / 3.0) * np.pi * (R0 ** 3) * (p_inf - p_v)

    if E_total_0 > 1e-30:
        eta = E_k_max_approx / E_total_0
    else:
        eta = 0.0
    return min(eta, 1.0)


def random_search_energy_allocation(N_stages, energy_budget, efficiency_weights, num_samples=5000):
    """
    使用随机搜索优化能量分配路径。
    对应 1367_tsp_random 的旅行商随机搜索框架。

    问题建模:
    - 将崩溃过程分为 N_stages 个阶段
    - 每个阶段 i，能量可以分配给: 动能(k), 声能(a), 热耗散(h)
    - 总预算约束: Σ E_i = energy_budget
    - 目标: 最大化总效率 = Σ w_k * E_kinetic + w_a * E_acoustic - w_h * E_heat

    参数:
        N_stages: 阶段数
        energy_budget: 总能量预算 [J]
        efficiency_weights: [w_k, w_a, w_h] 权重
        num_samples: 随机采样数
    返回:
        best_allocation: 最优分配矩阵 N_stages x 3
        best_cost: 最优成本（负效率）
    """
    w_k, w_a, w_h = efficiency_weights
    best_cost = -np.inf
    best_allocation = None

    for _ in range(num_samples):
        # 随机生成能量分配比例
        ratios = np.random.dirichlet(np.ones(3), size=N_stages)
        allocation = ratios * (energy_budget / N_stages)

        # 计算总效率
        total_kinetic = np.sum(allocation[:, 0])
        total_acoustic = np.sum(allocation[:, 1])
        total_heat = np.sum(allocation[:, 2])

        # 效率函数（带惩罚项，确保物理合理性）
        efficiency = (w_k * total_kinetic + w_a * total_acoustic - w_h * total_heat)
        # 惩罚：各阶段能量不能为负
        if np.any(allocation < 0):
            efficiency -= 1e10

        if efficiency > best_cost:
            best_cost = efficiency
            best_allocation = allocation.copy()

    return best_allocation, best_cost


def optimize_collapse_parameters(p_inf_range, R0_range, p_v, sigma, rho, c_sound,
                                  num_samples=1000):
    """
    在给定参数范围内搜索最优崩溃条件。
    返回使崩溃效率最高的 (p_inf, R0) 组合。
    """
    best_efficiency = -1.0
    best_params = None
    results = []

    for _ in range(num_samples):
        p_inf = np.random.uniform(p_inf_range[0], p_inf_range[1])
        R0 = np.random.uniform(R0_range[0], R0_range[1])

        eta = collapse_efficiency(R0, 1e-6, p_inf, p_v, sigma, rho, c_sound)
        results.append((p_inf, R0, eta))

        if eta > best_efficiency:
            best_efficiency = eta
            best_params = (p_inf, R0)

    return best_params, best_efficiency, np.array(results)


def energy_spectrum_analysis(R_history, dRdt_history, dt):
    """
    对气泡半径和速度历史进行频谱分析，识别特征振荡频率。
    使用离散傅里叶变换（DFT）。
    """
    N = len(R_history)
    if N < 4:
        return np.array([]), np.array([])

    freqs = np.fft.rfftfreq(N, d=dt)
    R_fft = np.abs(np.fft.rfft(R_history - np.mean(R_history)))
    v_fft = np.abs(np.fft.rfft(dRdt_history))

    return freqs, R_fft, v_fft
