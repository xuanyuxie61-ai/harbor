"""
portfolio_knapsack.py
0/1 背包问题求解与信用组合优化
应用于在风险资本约束下的最优信用资产配置

原项目映射: 628_knapsack_values
科学问题: 银行或资产管理公司在配置信用资产时，面临资本充足率约束
(Basel III 框架下的风险加权资产 RWA 约束)。
每个信用敞口 i 具有:
    - 预期收益 v_i (expected return)
    - 风险资本消耗 w_i (capital charge / RWA)
    - 违约概率 PD_i 和违约损失率 LGD_i

组合优化问题:
    max sum_i v_i * s_i
    s.t. sum_i w_i * s_i <= K  (资本约束)
         sum_i PD_i * LGD_i * EAD_i * s_i <= VaR_limit  (风险价值约束)
         s_i in {0, 1}

这是一个多维 0/1 背包问题。本模块实现动态规划求解器，
并注入信用风险特有的风险调整收益指标 (RAROC) 作为目标函数。
"""

import numpy as np
from typing import Tuple, List, Optional


def knapsack_01_dp(values: np.ndarray, weights: np.ndarray, capacity: int) -> Tuple[float, np.ndarray]:
    """
    经典 0/1 背包问题的动态规划求解

    动态规划递推:
        dp[i][w] = max( dp[i-1][w], dp[i-1][w-weights[i]] + values[i] )
    空间优化后使用一维数组:
        dp[w] = max( dp[w], dp[w-weights[i]] + values[i] )  (逆序遍历 w)

    Parameters:
        values: 价值数组 (n,)
        weights: 重量数组 (n,)，非负整数
        capacity: 背包容量

    Returns:
        max_value: 最大价值
        selection: 0/1 选择向量
    """
    n = len(values)
    weights = np.asarray(weights, dtype=int)
    capacity = int(capacity)

    if np.any(weights < 0):
        raise ValueError("重量不能为负")
    if capacity < 0:
        raise ValueError("容量不能为负")

    # 动态规划数组
    dp = np.zeros(capacity + 1, dtype=float)
    # 记录选择以回溯
    keep = np.zeros((n, capacity + 1), dtype=bool)

    for i in range(n):
        wi = weights[i]
        vi = values[i]
        if wi > capacity:
            continue
        for w in range(capacity, wi - 1, -1):
            if dp[w - wi] + vi > dp[w]:
                dp[w] = dp[w - wi] + vi
                keep[i, w] = True

    # 回溯
    selection = np.zeros(n, dtype=bool)
    w = capacity
    for i in range(n - 1, -1, -1):
        if keep[i, w]:
            selection[i] = True
            w -= weights[i]

    return dp[capacity], selection


def credit_portfolio_optimization(
    expected_returns: np.ndarray,
    capital_charges: np.ndarray,
    pd_values: np.ndarray,
    lgd_values: np.ndarray,
    ead_values: np.ndarray,
    total_capital: float,
    var_limit: float,
    resolution: int = 1000
) -> Tuple[np.ndarray, dict]:
    """
    双约束信用组合优化
    将双约束转化为单约束背包问题，通过将风险约束融入权重:
        effective_weight_i = capital_charges_i + lambda * risk_exposure_i
    其中 lambda 为风险厌恶系数，通过枚举调整。

    数学模型:
        max sum_i mu_i * s_i
        s.t. sum_i C_i * s_i <= C_total
             sum_i PD_i * LGD_i * EAD_i * s_i <= VaR_lim
             s_i in {0, 1}

    Parameters:
        expected_returns: 预期收益 (n,)
        capital_charges: 资本占用 (n,)
        pd_values: 违约概率 (n,)
        lgd_values: 违约损失率 (n,)
        ead_values: 风险敞口 (n,)
        total_capital: 总可用资本
        var_limit: 风险价值上限
        resolution: DP 离散化精度

    Returns:
        selection: 最优选择向量
        metrics: 组合指标字典
    """
    n = len(expected_returns)
    risk_exposure = pd_values * lgd_values * ead_values

    # 将连续权重离散化为整数
    max_weight = max(total_capital, var_limit)
    scale = resolution / max_weight

    # 尝试不同的 lambda 来平衡两个约束
    best_selection = None
    best_score = -np.inf
    best_metrics = None

    for lam in np.linspace(0.0, 5.0, 21):
        effective_weights = capital_charges + lam * risk_exposure
        # 归一化并离散化
        w_int = np.maximum((effective_weights * scale).astype(int), 1)
        cap_int = int(total_capital * scale)

        max_val, sel = knapsack_01_dp(expected_returns, w_int, cap_int)

        # 检查原始约束满足情况
        total_cap = np.sum(capital_charges * sel)
        total_risk = np.sum(risk_exposure * sel)
        total_return = np.sum(expected_returns * sel)

        if total_cap <= total_capital and total_risk <= var_limit:
            # 用 RAROC = 收益 / (资本 + 风险) 评分
            score = total_return / (total_cap + total_risk + 1e-8)
            if score > best_score:
                best_score = score
                best_selection = sel
                best_metrics = {
                    "total_return": total_return,
                    "total_capital": total_cap,
                    "total_risk": total_risk,
                    "raroc": score,
                    "lambda": lam
                }

    if best_selection is None:
        # 退化为仅资本约束
        w_int = np.maximum((capital_charges * scale).astype(int), 1)
        cap_int = int(total_capital * scale)
        _, best_selection = knapsack_01_dp(expected_returns, w_int, cap_int)
        total_cap = np.sum(capital_charges * best_selection)
        total_risk = np.sum(risk_exposure * best_selection)
        total_return = np.sum(expected_returns * best_selection)
        best_metrics = {
            "total_return": total_return,
            "total_capital": total_cap,
            "total_risk": total_risk,
            "raroc": total_return / (total_cap + total_risk + 1e-8),
            "lambda": 0.0
        }

    return best_selection, best_metrics


def generate_credit_portfolio_data(n_assets: int = 24, seed: int = 42) -> dict:
    """
    生成测试用的信用组合数据
    参照经典背包测试实例的结构
    """
    np.random.seed(seed)
    # 收益与风险正相关 (高风险高收益)
    base_return = np.random.uniform(0.5, 3.0, n_assets)
    capital = np.random.uniform(10, 100, n_assets)
    pd_vals = np.random.uniform(0.01, 0.15, n_assets)
    lgd_vals = np.random.uniform(0.2, 0.6, n_assets)
    ead_vals = np.random.uniform(50, 500, n_assets)

    return {
        "expected_returns": base_return,
        "capital_charges": capital,
        "pd_values": pd_vals,
        "lgd_values": lgd_vals,
        "ead_values": ead_vals
    }


def test_portfolio_knapsack():
    """测试信用组合优化"""
    data = generate_credit_portfolio_data(n_assets=15)
    sel, metrics = credit_portfolio_optimization(
        data["expected_returns"],
        data["capital_charges"],
        data["pd_values"],
        data["lgd_values"],
        data["ead_values"],
        total_capital=500.0,
        var_limit=2000.0,
        resolution=500
    )
    assert np.any(sel), "未选择任何资产"
    assert metrics["total_capital"] <= 500.0 * 1.01, "资本约束违反"
    print(f"portfolio_knapsack test passed. selected={np.sum(sel)}, RAROC={metrics['raroc']:.4f}")


if __name__ == "__main__":
    test_portfolio_knapsack()
