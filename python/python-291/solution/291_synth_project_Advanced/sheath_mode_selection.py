"""
sheath_mode_selection.py
========================
模式选择与最优子集选择模块（背包问题）。

本模块融合种子项目 623_knapsack_brute 和 625_knapsack_greedy 的
背包算法，用于等离子体鞘层稳定性分析中的最优模式选择。

物理背景：
    在多维鞘层稳定性分析中，需要选择一组 (m, n, p, ...) 模式
    来表征不稳定谱。计算成本随模式数指数增长，因此需要在
    有限计算预算内选择信息量最大的模式组合。

    将模式选择建模为背包问题：
        - 物品: 候选模式 i
        - 价值: 模式 i 的增长率 |γ_i| (信息量)
        - 重量: 模式 i 的计算代价 C_i
        - 容量: 总计算预算 B

    目标：max Σ v_i x_i  subject to  Σ w_i x_i ≤ B

核心算法：
    1. 贪心背包 (按 v/w 排序)
    2. 暴力搜索背包 (动态规划 + 子集枚举)
    3. FPTAS 近似
"""

import numpy as np
from itertools import combinations
from typing import Tuple, List, Optional
import math


def knapsack_brute(
    values: np.ndarray,
    weights: np.ndarray,
    capacity: float,
) -> Tuple[float, float, np.ndarray]:
    """
    暴力搜索求解 0-1 背包问题
    （源自种子项目 623_knapsack_brute: knapsack_brute）

    枚举所有 2^n 个子集，找最优解。

    参数：
        values: shape (n,) 各物品的价值
        weights: shape (n,) 各物品的重量
        capacity: 背包容量

    返回：
        max_value: 最优总价值
        total_weight: 最优总重量
        selection: shape (n,) 选择向量 (0 或 1)
    """
    n = len(values)
    if n > 25:
        raise ValueError(f"暴力搜索仅适用于 n≤25, 当前 n={n}")

    best_value = 0.0
    best_weight = 0.0
    best_selection = np.zeros(n, dtype=int)

    for mask in range(1 << n):
        total_v = 0.0
        total_w = 0.0
        selection = np.zeros(n, dtype=int)

        for i in range(n):
            if mask & (1 << i):
                total_v += values[i]
                total_w += weights[i]
                selection[i] = 1

        if total_w <= capacity and total_v > best_value:
            best_value = total_v
            best_weight = total_w
            best_selection = selection.copy()

    return best_value, best_weight, best_selection


def knapsack_greedy(
    values: np.ndarray,
    weights: np.ndarray,
    capacity: float,
) -> Tuple[float, float, np.ndarray]:
    """
    贪心算法求解 0-1 背包问题
    （源自种子项目 625_knapsack_greedy: knapsack_greedy）

    按价值/重量比 (v/w) 降序排列物品，
    依次放入不超过容量的物品。

    参数：
        values: shape (n,) 各物品的价值
        weights: shape (n,) 各物品的重量
        capacity: 背包容量

    返回：
        max_value: 总价值
        total_weight: 总重量
        selection: shape (n,) 选择向量
    """
    n = len(values)

    # 按 v/w 降序排列
    ratios = np.where(weights > 1e-15, values / weights, 0.0)
    order = np.argsort(-ratios)

    selection = np.zeros(n, dtype=int)
    total_v = 0.0
    total_w = 0.0

    for idx in order:
        if total_w + weights[idx] <= capacity:
            selection[idx] = 1
            total_v += values[idx]
            total_w += weights[idx]

    return total_v, total_w, selection


def knapsack_dp(
    values: np.ndarray,
    weights: np.ndarray,
    capacity: float,
    discretization: int = 1000,
) -> Tuple[float, float, np.ndarray]:
    """
    动态规划求解背包问题

    将连续容量离散化为 discretization 个等级

    参数：
        values: 价值数组
        weights: 重量数组
        capacity: 容量
        discretization: 离散化等级

    返回：
        同 knapsack_brute
    """
    n = len(values)
    W = discretization
    w_scale = capacity / W

    # DP 表
    dp = np.zeros(W + 1)
    choice = np.zeros((n, W + 1), dtype=bool)

    for i in range(n):
        wi = max(1, int(round(weights[i] / w_scale)))
        vi = values[i]

        for w in range(W, wi - 1, -1):
            if dp[w - wi] + vi > dp[w]:
                dp[w] = dp[w - wi] + vi
                choice[i, w] = True

    # 回溯
    selection = np.zeros(n, dtype=int)
    w = np.argmax(dp)
    total_v = dp[w]
    total_w = w * w_scale

    for i in range(n - 1, -1, -1):
        if choice[i, w]:
            selection[i] = 1
            w -= max(1, int(round(weights[i] / w_scale)))

    return total_v, total_w, selection


def sheath_mode_selection(
    mode_frequencies: np.ndarray,
    growth_rates: np.ndarray,
    computational_cost: np.ndarray,
    budget: float,
    method: str = "greedy",
) -> dict:
    """
    鞘层不稳定模式选择

    给定候选模式的频率、增长率和计算代价，
    选择最优模式子集使总信息量最大化。

    价值 = |增长率| (不稳定性的信息量)
    重量 = 计算代价

    参数：
        mode_frequencies: shape (n_modes,) 模式频率
        growth_rates: shape (n_modes,) 增长率 (虚部)
        computational_cost: shape (n_modes,) 各模式计算代价
        budget: 计算预算
        method: "greedy" | "brute" | "dp"

    返回：
        result: 结果字典
    """
    n_modes = len(mode_frequencies)

    # 价值 = 增长率幅度
    values = np.abs(growth_rates)
    # 归一化价值到正整数范围
    if np.max(values) > 1e-15:
        values_int = np.maximum(1, (values / np.max(values) * 100).astype(int))
    else:
        values_int = np.ones(n_modes, dtype=int)

    # 重量 = 计算代价
    weights = computational_cost.astype(float)

    if method == "greedy":
        total_v, total_w, selection = knapsack_greedy(
            values_int.astype(float), weights, budget
        )
    elif method == "brute":
        if n_modes > 20:
            # 太多，使用 dp 替代
            total_v, total_w, selection = knapsack_dp(
                values_int.astype(float), weights, budget
            )
        else:
            total_v, total_w, selection = knapsack_brute(
                values_int.astype(float), weights, budget
            )
    elif method == "dp":
        total_v, total_w, selection = knapsack_dp(
            values_int.astype(float), weights, budget
        )
    else:
        raise ValueError(f"未知方法: {method}")

    selected_indices = np.where(selection == 1)[0]
    n_selected = len(selected_indices)

    return {
        'n_total': n_modes,
        'n_selected': n_selected,
        'selected_indices': selected_indices,
        'selected_frequencies': mode_frequencies[selected_indices] if n_selected > 0 else np.array([]),
        'selected_growth_rates': growth_rates[selected_indices] if n_selected > 0 else np.array([]),
        'total_value': total_v,
        'total_weight': total_w,
        'budget': budget,
        'utilization': total_w / budget if budget > 0 else 0.0,
        'method': method,
    }


def generate_synthetic_modes(
    n_modes: int = 30,
    omega_ci: float = 0.1,
    omega_pi: float = 1.0,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    生成合成的模式数据

    模拟鞘层不稳定谱

    参数：
        n_modes: 模式数
        omega_ci: 离子回旋频率
        omega_pi: 离子等离子体频率
        seed: 随机种子

    返回：
        frequencies: 模式频率
        growth_rates: 增长率
        costs: 计算代价
    """
    rng = np.random.RandomState(seed)

    frequencies = np.zeros(n_modes)
    growth_rates = np.zeros(n_modes)
    costs = np.zeros(n_modes)

    for i in range(n_modes):
        m = rng.randint(1, 10)
        n = rng.randint(0, 5)
        frequencies[i] = m * omega_ci + n * omega_pi + rng.randn() * 0.1

        # 增长率：与频率非线性相关
        growth_rates[i] = 0.01 * rng.exponential() * (1 + 0.5 * np.sin(frequencies[i]))

        # 计算代价：与频率平方成正比
        costs[i] = 1.0 + 0.5 * frequencies[i]**2 + rng.exponential()

    return frequencies, growth_rates, costs
