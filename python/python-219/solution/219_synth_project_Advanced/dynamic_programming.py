"""
dynamic_programming.py
======================

动态规划与离散阶段决策模块。

融合种子项目:
  - 624_knapsack_dynamic : 0/1 背包问题的动态规划解法

在最优控制中, 该模块用于:
  1. 多级火箭阶段选择 (哪些发动机点火)
  2. 燃料分配决策 (离散化的最优资源分配)
  3. 多目标访问顺序的分支定界
  4. 混合整数最优控制的松弛与舍入

数学公式:
---------
1. 0/1 背包 DP:
     dp[i, w] = max(dp[i-1, w], dp[i-1, w-w_i] + v_i)
     其中 v_i 为物品价值, w_i 为重量, w 为容量

2. 多级火箭阶段选择 (类比背包):
     设 n 个可选发动机, 第 i 个提供推力 T_i, 消耗燃料 m_i
     目标: 在总燃料约束 M 下最大化总 Delta_v
     Delta_v = sum_{i in S} I_sp_i * g_0 * ln(m_0_i / m_f_i)

3. 燃料分配 DP:
     设总燃料 M 分配给 K 个阶段, 每阶段消耗 m_k
     目标: 最大化终端速度 v_f
     约束: sum m_k <= M,  m_k >= 0

4. 分支定界 (用于多目标 TSP):
     下界 = 最小生成树 / 分配问题松弛
"""

from __future__ import annotations

from typing import List, Tuple


# ===========================================================================
# 1. 0/1 背包动态规划 (来自 624_knapsack_dynamic)
# ===========================================================================
def knapsack_dp_table(
    values: List[int], weights: List[int], capacity: int
) -> List[List[int]]:
    """构造背包 DP 表.

    dp[i, w] = 前 i 个物品, 容量 w 下的最大价值

    递推:
        dp[i, w] = max(dp[i-1, w], dp[i-1, w - w_i] + v_i)  if w >= w_i
                 = dp[i-1, w]                                 otherwise

    Parameters
    ----------
    values : List[int]
        物品价值 v_i (长度 n).
    weights : List[int]
        物品重量 w_i (长度 n).
    capacity : int
        背包容量 K.

    Returns
    -------
    List[List[int]]
        (n+1) x (K+1) DP 表.
    """
    n = len(values)
    if n != len(weights):
        raise ValueError("values 和 weights 长度不一致")
    if capacity < 0:
        raise ValueError(f"容量 {capacity} 不能为负")

    # dp[i][w] = 前 i 个物品, 容量 w 的最大价值
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        v_i = values[i - 1]
        w_i = weights[i - 1]
        for w in range(capacity + 1):
            if w_i <= w:
                dp[i][w] = max(dp[i - 1][w], dp[i - 1][w - w_i] + v_i)
            else:
                dp[i][w] = dp[i - 1][w]

    return dp


def knapsack_dp(
    values: List[int], weights: List[int], capacity: int
) -> Tuple[List[int], int]:
    """0/1 背包问题求解 (返回选中物品列表和最大价值).

    算法 (来自 624_knapsack_dynamic):
        1. 构造 DP 表 dp[i, w]
        2. 从 dp[n, K] 回溯, 找出选中的物品

    Parameters
    ----------
    values : List[int]
        物品价值.
    weights : List[int]
        物品重量.
    capacity : int
        背包容量.

    Returns
    -------
    (selected, max_value) : (List[int], int)
        选中的物品索引 (1-based) 和最大价值.
    """
    dp = knapsack_dp_table(values, weights, capacity)
    n = len(values)

    # 回溯
    selected: List[int] = []
    w = capacity
    for i in range(n, 0, -1):
        if dp[i][w] != dp[i - 1][w]:
            # 物品 i 被选中
            selected.append(i)
            w -= weights[i - 1]

    selected.reverse()
    max_value = dp[n][capacity]
    return selected, max_value


# ===========================================================================
# 2. 多级火箭阶段选择 (背包类比)
# ===========================================================================
def rocket_stage_selection(
    thrusts: List[float],
    fuel_consumptions: List[int],
    fuel_capacity: int,
    I_sp_values: List[float],
) -> Tuple[List[int], float]:
    """多级火箭阶段选择 (0/1 背包).

    问题: 从 n 个可选发动机中选择子集 S, 使得:
        总燃料消耗 <= fuel_capacity
        总 Delta_v 最大化

    Delta_v 近似 (简化):
        Delta_v ≈ sum_{i in S} I_sp_i * g_0 * ln(1 + fuel_i / m_dry)
               ≈ sum_{i in S} c_i * fuel_i  (线性化)

    其中 c_i = I_sp_i * g_0 / m_dry 为效率系数.

    为使用整数 DP, 将 Delta_v 缩放为整数.

    Parameters
    ----------
    thrusts : List[float]
        各发动机推力 [N] (此处未直接使用, 仅用于记录).
    fuel_consumptions : List[int]
        各发动机燃料消耗 [kg] (整数).
    fuel_capacity : int
        总燃料容量 [kg].
    I_sp_values : List[float]
        各发动机比冲 [s].

    Returns
    -------
    (selected_stages, total_delta_v) : (List[int], float)
        选中的阶段索引 (1-based) 和总 Delta_v [m/s].
    """
    n = len(thrusts)
    if not (n == len(fuel_consumptions) == len(I_sp_values)):
        raise ValueError("输入列表长度不一致")

    g0 = 9.81
    m_dry = 500.0  # 假设干质量

    # 计算每个阶段的 Delta_v 贡献 (线性化)
    delta_v_contribs = [
        I_sp_values[i] * g0 * (fuel_consumptions[i] / m_dry)
        for i in range(n)
    ]

    # 缩放为整数 (乘以 100 并四舍五入)
    scale = 100
    values_int = [int(round(dv * scale)) for dv in delta_v_contribs]

    # 求解 0/1 背包
    selected, max_val_int = knapsack_dp(values_int, fuel_consumptions, fuel_capacity)

    # 计算真实 Delta_v
    total_delta_v = sum(delta_v_contribs[i - 1] for i in selected)

    return selected, total_delta_v


# ===========================================================================
# 3. 燃料分配 DP (连续松弛 + 离散化)
# ===========================================================================
def fuel_allocation_dp(
    n_stages: int,
    total_fuel: int,
    stage_efficiency: List[float],
) -> Tuple[List[int], float]:
    """燃料分配问题 (DP 求解).

    问题: 将 total_fuel 分配给 n_stages 个阶段, 每阶段分配 m_k,
          最大化总效能 sum_{k=1}^{n_stages} eta_k * m_k,
          约束 sum m_k <= total_fuel, m_k >= 0.

    这是无界背包的变体, 可用 DP 求解.

    Parameters
    ----------
    n_stages : int
        阶段数.
    total_fuel : int
        总燃料 [kg].
    stage_efficiency : List[float]
        各阶段效率系数 eta_k.

    Returns
    -------
    (allocation, total_performance) : (List[int], float)
        各阶段燃料分配和总效能.
    """
    if n_stages <= 0:
        raise ValueError("阶段数必须为正")
    if total_fuel < 0:
        raise ValueError("总燃料不能为负")
    if len(stage_efficiency) != n_stages:
        raise ValueError("效率系数数量不匹配")

    # DP: dp[w] = 容量 w 下的最大效能
    dp = [0.0] * (total_fuel + 1)
    choice = [0] * (total_fuel + 1)  # 记录最后选择的阶段

    for w in range(1, total_fuel + 1):
        for k in range(n_stages):
            # 每阶段每次消耗 1 kg (简化)
            val = dp[w - 1] + stage_efficiency[k]
            if val > dp[w]:
                dp[w] = val
                choice[w] = k + 1

    # 回溯
    allocation = [0] * n_stages
    w = total_fuel
    while w > 0:
        k = choice[w]
        allocation[k - 1] += 1
        w -= 1

    return allocation, dp[total_fuel]


# ===========================================================================
# 自检
# ===========================================================================
def self_check() -> None:
    """动态规划模块自检."""
    print("[Knapsack DP] 经典 0/1 背包:")
    values = [60, 100, 120]
    weights = [10, 20, 30]
    capacity = 50
    selected, max_val = knapsack_dp(values, weights, capacity)
    print(f"  物品价值: {values}")
    print(f"  物品重量: {weights}")
    print(f"  容量: {capacity}")
    print(f"  选中: {selected}, 最大价值: {max_val}")

    print("[Rocket Stage] 多级火箭阶段选择:")
    thrusts = [5000.0, 8000.0, 3000.0, 6000.0]
    fuel_cons = [200, 400, 150, 300]
    fuel_cap = 700
    I_sp = [300.0, 320.0, 280.0, 310.0]
    sel_stages, total_dv = rocket_stage_selection(thrusts, fuel_cons, fuel_cap, I_sp)
    print(f"  选中阶段: {sel_stages}")
    print(f"  总 Delta_v: {total_dv:.2f} m/s")

    print("[Fuel Allocation] 燃料分配:")
    n_stages = 3
    total_fuel = 100
    eff = [1.0, 1.2, 0.9]
    alloc, perf = fuel_allocation_dp(n_stages, total_fuel, eff)
    print(f"  分配: {alloc}")
    print(f"  总效能: {perf:.2f}")


if __name__ == "__main__":
    self_check()
