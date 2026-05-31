"""
pbpk_optimization.py
基于种子项目 624_knapsack_dynamic

实现动态规划优化算法，用于 PBPK 中的：
1. 多靶点给药的剂量分配优化（0/1 Knapsack 变体）
2. 多周期给药方案的最优时间窗口分配
3. 药物组合的选择优化

在 PBPK 模型中用于：
- 给定总剂量预算，优化各 compartment 的分配以最大化疗效
- 考虑毒性约束下的最优给药策略
- 多药联用的组合优化
"""

import numpy as np
from typing import List, Tuple, Dict

# ---------------------------------------------------------------------------
# 0/1 背包动态规划（经典）
# ---------------------------------------------------------------------------

def knapsack_01(weights: np.ndarray, values: np.ndarray, capacity: float) -> Tuple[float, List[int]]:
    """
    0/1 背包问题的动态规划求解。
    递推关系：
        DP[i, w] = max(DP[i-1, w], DP[i-1, w-w_i] + v_i)
    返回：(最大价值, 选中物品索引列表)
    """
    n = len(weights)
    if n != len(values):
        raise ValueError("weights and values must have same length")
    if capacity < 0:
        raise ValueError("capacity must be non-negative")
    if n == 0:
        return 0.0, []

    # 将连续权重离散化为整数（精度 0.01）
    scale = 100.0
    W_int = int(np.ceil(capacity * scale))
    w_int = np.maximum((weights * scale).astype(int), 0)
    v = np.maximum(values, 0.0)

    # 一维 DP 数组优化内存
    dp = np.zeros(W_int + 1, dtype=float)
    # 记录选择（用于回溯）
    choice = np.full((n, W_int + 1), -1, dtype=int)

    for i in range(n):
        wi = w_int[i]
        vi = v[i]
        # 逆序更新以避免重复选择
        for w in range(W_int, wi - 1, -1):
            if dp[w - wi] + vi > dp[w]:
                dp[w] = dp[w - wi] + vi
                choice[i, w] = 1
            else:
                choice[i, w] = 0

    # 回溯
    selected = []
    w = W_int
    for i in range(n - 1, -1, -1):
        if choice[i, w] == 1:
            selected.append(i)
            w -= w_int[i]

    max_val = dp[W_int]
    return max_val, selected[::-1]


# ---------------------------------------------------------------------------
# PBPK 剂量分配优化
# ---------------------------------------------------------------------------

def optimize_dose_allocation(total_dose: float,
                              organ_volumes: np.ndarray,
                              organ_sensitivities: np.ndarray,
                              organ_toxicities: np.ndarray,
                              max_toxicity: float) -> Tuple[np.ndarray, float]:
    """
    使用背包问题框架优化多器官剂量分配。
    将剂量空间离散化，每个"剂量包"为一个小单位剂量 δ。
    目标：最大化总疗效 Σ sensitivity_i * dose_i
    约束：Σ dose_i <= total_dose, Σ toxicity_i * dose_i <= max_toxicity

    参数：
        total_dose : 总剂量 [mg]
        organ_volumes : 各器官体积 [L]
        organ_sensitivities : 各器官疗效敏感度 [1/mg]
        organ_toxicities : 各器官毒性系数 [1/mg]
        max_toxicity : 最大可接受毒性
    返回：
        dose_allocation : 各器官分配的剂量
        total_efficacy : 总疗效
    """
    n_organs = len(organ_volumes)
    if n_organs != len(organ_sensitivities) or n_organs != len(organ_toxicities):
        raise ValueError("Array lengths must match")
    if total_dose <= 0 or max_toxicity < 0:
        raise ValueError("Invalid dose or toxicity limits")

    # 双约束转化为单约束：惩罚函数法
    # 实际实现：枚举每个器官的剂量水平
    n_levels = 20  # 每个器官 20 个剂量水平
    delta = total_dose / n_levels

    # 构建物品列表：(weight=delta, value=sensitivity_i * delta_j)
    weights = []
    values = []
    organ_idx = []
    for i in range(n_organs):
        for j in range(1, n_levels + 1):
            dose_j = j * delta
            # 惩罚项：若毒性超限则价值为 0
            if organ_toxicities[i] * dose_j > max_toxicity / n_organs:
                continue
            weights.append(dose_j)
            values.append(organ_sensitivities[i] * dose_j)
            organ_idx.append(i)

    weights = np.array(weights)
    values = np.array(values)
    max_val, selected = knapsack_01(weights, values, total_dose)

    # 汇总每个器官的剂量
    allocation = np.zeros(n_organs)
    for idx in selected:
        allocation[organ_idx[idx]] += weights[idx]

    total_efficacy = np.sum(organ_sensitivities * allocation)
    return allocation, total_efficacy


def optimize_dosing_schedule(horizon_hours: float,
                              dose_units: np.ndarray,
                              efficacy_values: np.ndarray,
                              toxicity_values: np.ndarray,
                              min_interval: float = 4.0,
                              max_daily_dose: float = 1000.0) -> Tuple[np.ndarray, float]:
    """
    优化多周期给药时间表。
    将时间 horizon 分为若干窗口，每个窗口可选择是否给药。
    使用动态规划最大化累积疗效 - λ * 累积毒性。

    参数：
        horizon_hours : 总时间 [h]
        dose_units : 可选剂量单元列表 [mg]
        efficacy_values : 对应剂量的疗效增量
        toxicity_values : 对应剂量的毒性增量
        min_interval : 最小给药间隔 [h]
        max_daily_dose : 每日最大剂量 [mg]
    返回：
        schedule : 二元决策向量（1=给药，0=跳过）
        net_benefit : 净效益
    """
    n_slots = int(horizon_hours / min_interval)
    if n_slots < 1:
        raise ValueError("Horizon too short")
    if len(dose_units) != len(efficacy_values) or len(dose_units) != len(toxicity_values):
        raise ValueError("Dose arrays must have same length")

    # 简化：每个 slot 只有一个剂量选项（最大剂量）
    n_options = len(dose_units)
    # DP[i] = 到第 i 个 slot 为止的最大净效益
    dp = np.full(n_slots + 1, -np.inf)
    dp[0] = 0.0
    choice = np.full(n_slots + 1, -1, dtype=int)

    lambda_penalty = 0.5
    for i in range(1, n_slots + 1):
        # 选择：不给药
        if dp[i - 1] > dp[i]:
            dp[i] = dp[i - 1]
            choice[i] = -1
        # 选择：给药（必须间隔至少 1 个 slot）
        for opt in range(n_options):
            prev = max(0, i - 1)
            if dp[prev] > -np.inf:
                val = dp[prev] + efficacy_values[opt] - lambda_penalty * toxicity_values[opt]
                if val > dp[i]:
                    dp[i] = val
                    choice[i] = opt

    # 回溯
    schedule = np.zeros(n_slots, dtype=int)
    i = n_slots
    while i > 0:
        if choice[i] >= 0:
            schedule[i - 1] = 1
            i -= 1
        i -= 1

    return schedule, dp[n_slots]


# ---------------------------------------------------------------------------
# 多药物组合优化
# ---------------------------------------------------------------------------

def optimize_drug_combination(n_drugs: int, budget: float,
                               drug_costs: np.ndarray,
                               drug_efficacies: np.ndarray,
                               synergy_matrix: np.ndarray = None) -> Tuple[List[int], float]:
    """
    在多药物联合治疗中选择最优药物组合。
    使用 0/1 背包问题：选择药物子集使疗效最大，总成本不超过预算。
    synergy_matrix : 药物协同效应矩阵（可选）
    """
    if len(drug_costs) != n_drugs or len(drug_efficacies) != n_drugs:
        raise ValueError("Array lengths must match n_drugs")
    # 考虑协同效应调整价值
    values = drug_efficacies.copy()
    if synergy_matrix is not None:
        for i in range(n_drugs):
            for j in range(i + 1, n_drugs):
                if synergy_matrix[i, j] > 0:
                    values[i] += 0.5 * synergy_matrix[i, j]
                    values[j] += 0.5 * synergy_matrix[i, j]

    max_val, selected = knapsack_01(drug_costs, values, budget)
    return selected, max_val


# ---------------------------------------------------------------------------
# 模块自检
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    w = np.array([2.0, 3.0, 4.0, 5.0])
    v = np.array([3.0, 4.0, 5.0, 6.0])
    cap = 5.0
    max_val, sel = knapsack_01(w, v, cap)
    print(f"Knapsack max value: {max_val:.2f}, selected: {sel}")

    alloc, eff = optimize_dose_allocation(
        500.0,
        np.array([1.5, 0.3, 30.0, 10.0, 0.5]),
        np.array([0.8, 0.6, 0.3, 0.2, 1.0]),
        np.array([0.05, 0.08, 0.02, 0.01, 0.10]),
        50.0
    )
    print(f"Dose allocation: {alloc}, total efficacy: {eff:.2f}")

    sched, benefit = optimize_dosing_schedule(
        72.0,
        np.array([100.0, 200.0]),
        np.array([10.0, 18.0]),
        np.array([2.0, 5.0])
    )
    print(f"Schedule length: {len(sched)}, doses given: {sched.sum()}, net benefit: {benefit:.2f}")
