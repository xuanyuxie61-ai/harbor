# -*- coding: utf-8 -*-
"""
transition_predictor.py
高超声速边界层转捩位置预测与优化

核心算法来源：
- tsp_moler (traveler): 旅行商问题启发式求解（2-opt 与点插入）

物理背景：
高超声速飞行器表面不同展向位置的转捩位置构成一条"转捩前沿"(transition front)。
在考虑展向变化、表面粗糙度分布、热斑效应时，
转捩前沿的预测可建模为沿展向站位的优化问题：

    寻找展向站位序列 z_1, z_2, ..., z_m，使得
        Σ_i |x_t(z_i) - x_t(z_{i+1})| + 惩罚项(粗糙度, 热流)
    最小

这类似于 TSP：在不同展向站位"访问"转捩位置，寻找最光滑的转捩前沿曲线。
同时结合 e^N 方法计算各站位转捩位置。
"""

import numpy as np
from math import sqrt


def e_n_method(Re_x_array, alpha_i_array, N_cr=9.0):
    """
    e^N 转捩预测方法。

    沿流向积分扰动放大因子:
        N(x) = -∫_{x_0}^{x} α_i(s) ds

    当 N(x_t) = N_cr 时发生转捩。
    典型临界值: N_cr ≈ 7~11（低噪声风洞 N_cr ≈ 9，飞行环境 N_cr ≈ 12~14）。

    参数:
        Re_x_array (np.ndarray): 流向当地雷诺数
        alpha_i_array (np.ndarray): 空间增长率（负值表示增长）
        N_cr (float): 临界 N 因子

    返回:
        tuple: (Re_xt, N_profile) 转捩雷诺数与 N 剖面
    """
    Re = np.asarray(Re_x_array)
    ai = np.asarray(alpha_i_array)
    n = len(Re)
    N = np.zeros(n)

    for i in range(1, n):
        dRe = Re[i] - Re[i - 1]
        if dRe <= 0:
            N[i] = N[i - 1]
            continue
        # 梯形积分
        N[i] = N[i - 1] - 0.5 * dRe * (ai[i] + ai[i - 1])

    # 查找 N = N_cr 的位置
    Re_xt = None
    for i in range(1, n):
        if N[i - 1] < N_cr <= N[i] or N[i] < N_cr <= N[i - 1]:
            # 线性插值
            frac = (N_cr - N[i - 1]) / (N[i] - N[i - 1])
            Re_xt = Re[i - 1] + frac * (Re[i] - Re[i - 1])
            break

    if Re_xt is None:
        Re_xt = Re[-1] if N[-1] >= N_cr else Re[0]

    return Re_xt, N


def compute_growth_rate_profile(Re_x, Ma=6.0, Re_unit=1e6, Tw_Te=1.0):
    """
    简化的空间增长率剖面模型（用于 e^N 积分）。

    基于 Mack 第二模态的经验关联:
        α_i ≈ -C * (Re/Re_0)^p * exp(-(Re/Re_max)^q)

    参数:
        Re_x (np.ndarray): 当地雷诺数
        Ma (float): 马赫数
        Re_unit (float): 单位雷诺数
        Tw_Te (float): 壁温比

    返回:
        np.ndarray: α_i 剖面
    """
    Re = np.asarray(Re_x)
    # 经验参数（随马赫数与壁温变化）
    C = 0.002 * (Ma / 6.0) ** 1.5 * (Tw_Te ** (-0.3))
    p = 0.5
    q = 2.0
    Re_max = 3e6 * (Ma / 6.0) ** (-0.8) * (Tw_Te ** 0.4)

    alpha_i = -C * (Re / 1e6) ** p * np.exp(-(Re / Re_max) ** q)
    return alpha_i


def transition_front_cost(positions, penalties):
    """
    转捩前沿的总成本函数（类比 TSP 路径长度）。

    成本 = Σ |Δx_t| + λ Σ penalty_i

    参数:
        positions (np.ndarray): 展向站位的转捩位置序列
        penalties (np.ndarray): 各站位的附加惩罚（粗糙度、热斑等）

    返回:
        float: 总成本
    """
    path_diff = np.sum(np.abs(np.diff(positions)))
    penalty_sum = np.sum(penalties)
    return path_diff + penalty_sum


def optimize_transition_front(spanwise_positions, initial_xt, penalties,
                               max_iter=5000, lambda_penalty=0.5):
    """
    基于 tsp_moler (traveler) 启发式算法的转捩前沿优化。

    展向站位已固定，问题转化为：
    在固定站位顺序下，通过局部交换与插入优化转捩位置的分配
    （处理多工况、多模态竞争导致的转捩位置跳变）。

    参数:
        spanwise_positions (np.ndarray): 展向坐标 z_i
        initial_xt (np.ndarray): 初始转捩位置猜测
        penalties (np.ndarray): 惩罚项
        max_iter (int): 最大迭代次数
        lambda_penalty (float): 惩罚权重

    返回:
        tuple: (optimized_xt, cost_history)
    """
    n = len(initial_xt)
    xt = initial_xt.copy()
    cost = transition_front_cost(xt, lambda_penalty * penalties)
    cost_history = [cost]

    for _ in range(max_iter):
        # 2-opt 局部搜索：交换两个展向站位的转捩位置
        i, j = np.random.randint(0, n, size=2)
        if i == j:
            continue

        xt_new = xt.copy()
        xt_new[i], xt_new[j] = xt_new[j], xt_new[i]
        cost_new = transition_front_cost(xt_new, lambda_penalty * penalties)

        if cost_new < cost:
            xt = xt_new
            cost = cost_new
            cost_history.append(cost)
            continue

        # 单点插入：将一个站位的转捩位置移动到另一位置附近
        i = np.random.randint(0, n)
        j = np.random.randint(0, n - 1)
        xt_new = np.delete(xt, i)
        xt_new = np.insert(xt_new, j, xt[i])
        cost_new = transition_front_cost(xt_new, lambda_penalty * penalties)

        if cost_new < cost:
            xt = xt_new
            cost = cost_new
            cost_history.append(cost)

    return xt, cost_history


def multi_station_transition_prediction(Ma, Re_unit, Tw_Te, Tu,
                                         z_stations, roughness_array,
                                         N_cr=9.0):
    """
    多展向站位的转捩位置预测。

    对每个站位:
        1. 基于当地参数计算增长率剖面
        2. 积分 e^N
        3. 粗糙度修正（依据 h_s/δ 增加 N 的有效值或降低 N_cr）

    参数:
        Ma (float): 马赫数
        Re_unit (float): 单位雷诺数
        Tw_Te (float): 壁温比
        Tu (float): 湍流度
        z_stations (np.ndarray): 展向站位 [m]
        roughness_array (np.ndarray): 相对粗糙度 k_s/δ
        N_cr (float): 临界 N 因子

    返回:
        dict: 各站位转捩位置与统计信息
    """
    n_stations = len(z_stations)
    Re_xt = np.zeros(n_stations)
    N_profiles = []

    for i in range(n_stations):
        # 当地雷诺数范围
        Re_x = np.linspace(1e5, 1e7, 500)

        # 增长率剖面
        ai = compute_growth_rate_profile(Re_x, Ma, Re_unit, Tw_Te)

        # 粗糙度修正：有效 N_cr 降低
        # 经验：ΔN_cr ≈ -3.0 * (k_s/δ)^0.5
        ksd = roughness_array[i]
        N_cr_eff = max(2.0, N_cr - 3.0 * (ksd ** 0.5))

        Re_t, N_prof = e_n_method(Re_x, ai, N_cr_eff)
        Re_xt[i] = Re_t
        N_profiles.append(N_prof)

    # 计算转捩前沿的光滑性指标
    smoothness = np.sum(np.diff(Re_xt) ** 2)

    return {
        'z_stations': z_stations,
        'Re_xt': Re_xt,
        'N_profiles': N_profiles,
        'smoothness': smoothness,
        'mean_Re_xt': np.mean(Re_xt),
        'std_Re_xt': np.std(Re_xt, ddof=1)
    }


def receptivity_coefficient(Ma, Tw_Te, Tu, F=2.5e-6):
    """
    感受性系数估计。

    自由流声学扰动进入边界层时，感受性系数大致满足:
        C_rec ∝ Tu * Ma^2 * (Tw/Te)^{-0.5} * f(F)

    其中 F = 2πf ν / u_e^2 为无量纲频率。

    参数:
        Ma (float): 马赫数
        Tw_Te (float): 壁温比
        Tu (float): 湍流度
        F (float): 无量纲频率

    返回:
        float: 初始扰动幅值 A_0 的估计
    """
    C_rec = Tu * (Ma ** 2) * (Tw_Te ** (-0.5)) * np.exp(-F ** 2 / 1e-11)
    return max(C_rec, 1e-10)
