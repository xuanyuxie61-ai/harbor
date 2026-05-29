"""
transport_optimizer.py
拓扑半金属中的输运优化问题

凝聚态物理背景：
在Weyl半金属中，电子输运受拓扑保护的手征反常（Chiral Anomaly）影响，
产生负磁阻效应。输运通道的选择可以建模为约束优化问题。

核心物理模型：
考虑N个可能的输运通道，每个通道有：
- 电导增益 g_i（与Berry曲率相关）
- 能量消耗 e_i（与散射率相关）
- 总能量预算 E_budget

优化目标：在总能量约束下最大化总电导
    max sum_i g_i * x_i
    s.t. sum_i e_i * x_i <= E_budget
         0 <= x_i <= 1

这正是连续（有理数）背包问题！

基于种子项目627_knapsack_rational的核心贪心算法：
1. 按"电导密度"g_i/e_i排序
2. 优先选择电导密度最高的通道
3. 若预算不足，取最后一个通道的部分比例

数学上，这等价于寻找最优的费米面切片选择策略，
使得在有限能量窗口内最大化拓扑保护的输运贡献。
"""

import numpy as np
from typing import Tuple


def knapsack_rational(n: int, budget: float, gains: np.ndarray,
                      costs: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """
    有理数背包问题求解器
    
    基于种子项目627_knapsack_rational的核心算法。
    
    假设输入已按增益密度（gain/cost）降序排列。
    
    Parameters
    ----------
    n : int
        项目数量
    budget : float
        总预算
    gains : np.ndarray, shape (n,)
        各项目的增益（非负）
    costs : np.ndarray, shape (n,)
        各项目的成本（非负）
    
    Returns
    -------
    x : np.ndarray, shape (n,)
        选择比例：0（不选）、1（全选）、或(0,1)（部分选）
    total_cost : float
    total_gain : float
    """
    x = np.zeros(n)
    total_cost = 0.0
    total_gain = 0.0
    
    for i in range(n):
        if costs[i] < 1e-15:
            # 零成本项目，全部选取
            x[i] = 1.0
            total_gain += gains[i]
            continue
        
        if budget <= total_cost + 1e-15:
            x[i] = 0.0
        elif total_cost + costs[i] <= budget:
            x[i] = 1.0
            total_cost += costs[i]
            total_gain += gains[i]
        else:
            # 部分选取最后一个项目
            remaining = budget - total_cost
            x[i] = remaining / costs[i]
            total_cost = budget
            total_gain += gains[i] * x[i]
    
    return x, total_cost, total_gain


def sort_by_profit_density(gains: np.ndarray, costs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    按增益密度降序排列
    
    增益密度 = gain / cost
    
    Parameters
    ----------
    gains : np.ndarray
    costs : np.ndarray
    
    Returns
    -------
    gains_sorted : np.ndarray
    costs_sorted : np.ndarray
    """
    # 避免除零
    safe_costs = np.where(costs < 1e-15, 1e-15, costs)
    density = gains / safe_costs
    
    # 降序排列
    idx = np.argsort(-density)
    return gains[idx], costs[idx]


def transport_channel_selection(n_channels: int, e_fermi: float,
                                 energy_window: float,
                                 ham, bz_sampler) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    选择最优输运通道
    
    在Weyl半金属中，输运通道对应k空间中不同方向的Berry曲率极值方向。
    
    Parameters
    ----------
    n_channels : int
        通道数
    e_fermi : float
        Fermi能级
    energy_window : float
        能量窗口
    ham : WeylHamiltonian
    bz_sampler : callable
        k点采样函数
    
    Returns
    -------
    x : np.ndarray
        最优选择比例
    total_cost : float
    total_gain : float
    """
    # 采样k点
    k_points = bz_sampler(n_channels)
    
    gains = np.zeros(n_channels)
    costs = np.zeros(n_channels)
    
    for i in range(n_channels):
        k = k_points[i]
        energies, _ = ham.eigenproblem(k)
        
        # 增益：与能带梯度相关（高群速度 = 高增益）
        gap = abs(energies[1] - energies[0])
        gains[i] = 1.0 / (gap + 0.01)  # 能隙越小增益越高
        
        # 成本：与能量偏离Fermi面的程度相关
        e_avg = 0.5 * (energies[0] + energies[1])
        costs[i] = abs(e_avg - e_fermi) + 0.01
    
    # 按增益密度排序
    gains_sorted, costs_sorted = sort_by_profit_density(gains, costs)
    
    # 求解背包问题
    budget = energy_window * n_channels
    x, total_cost, total_gain = knapsack_rational(n_channels, budget, gains_sorted, costs_sorted)
    
    return x, total_cost, total_gain


def chiral_anomaly_conductance(ham, k_points: np.ndarray, e_field: np.ndarray,
                                b_field: np.ndarray, band_index: int = 0) -> np.ndarray:
    """
    计算手征反常修正的电导
    
    在平行电场和磁场下，Weyl半金属出现负磁阻：
        sigma(E||B) ~ sigma_0 * (1 + C * |E·B|)
    
    其中C与Berry曲率相关。
    
    Parameters
    ----------
    ham : WeylHamiltonian
    k_points : np.ndarray, shape (N, 3)
    e_field : np.ndarray, shape (3,)
    b_field : np.ndarray, shape (3,)
    band_index : int
    
    Returns
    -------
    conductance : np.ndarray, shape (N,)
    """
    from berry_curvature import berry_curvature_numeric
    
    N = k_points.shape[0]
    conductance = np.zeros(N)
    
    e_dot_b = np.dot(e_field, b_field)
    e_norm = np.linalg.norm(e_field)
    b_norm = np.linalg.norm(b_field)
    
    for i in range(N):
        k = k_points[i]
        omega = berry_curvature_numeric(ham, k, band_index)
        
        # TODO Hole_3: 从Berry曲率张量提取矢量并计算手征反常修正电导
        # 步骤:
        #   1. 将反对称Berry曲率张量 Omega[3,3] 转换为矢量形式:
        #      omega_vec = (Omega_yz, Omega_zx, Omega_xy) = (Omega[1,2], Omega[2,0], Omega[0,1])
        #   2. 计算手征反常修正因子:
        #      anomaly_factor = 1.0 + 0.5 * |E·B| / (|E||B|) * |omega_vec · B|
        #   3. 存储到 conductance[i]
        raise NotImplementedError("Hole_3: 手征反常电导计算待实现")
    
    return conductance
