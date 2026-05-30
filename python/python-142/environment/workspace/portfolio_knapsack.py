
import numpy as np
from typing import Tuple, List, Optional


def knapsack_01_dp(values: np.ndarray, weights: np.ndarray, capacity: int) -> Tuple[float, np.ndarray]:
    n = len(values)
    weights = np.asarray(weights, dtype=int)
    capacity = int(capacity)

    if np.any(weights < 0):
        raise ValueError("重量不能为负")
    if capacity < 0:
        raise ValueError("容量不能为负")


    dp = np.zeros(capacity + 1, dtype=float)

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
    n = len(expected_returns)
    risk_exposure = pd_values * lgd_values * ead_values


    max_weight = max(total_capital, var_limit)
    scale = resolution / max_weight


    best_selection = None
    best_score = -np.inf
    best_metrics = None

    for lam in np.linspace(0.0, 5.0, 21):
        effective_weights = capital_charges + lam * risk_exposure

        w_int = np.maximum((effective_weights * scale).astype(int), 1)
        cap_int = int(total_capital * scale)

        max_val, sel = knapsack_01_dp(expected_returns, w_int, cap_int)


        total_cap = np.sum(capital_charges * sel)
        total_risk = np.sum(risk_exposure * sel)
        total_return = np.sum(expected_returns * sel)

        if total_cap <= total_capital and total_risk <= var_limit:

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
    np.random.seed(seed)

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
