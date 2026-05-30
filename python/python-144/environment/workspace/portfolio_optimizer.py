
import numpy as np
from scipy.optimize import minimize


def markowitz_min_variance(Sigma: np.ndarray,
                           target_return: float = None,
                           mu: np.ndarray = None) -> dict:
    n = Sigma.shape[0]
    if Sigma.shape != (n, n):
        raise ValueError("markowitz_min_variance: Sigma 必须是方阵。")

    Sigma_reg = Sigma + 1e-6 * np.eye(n)

    def objective(w):
        return 0.5 * w @ Sigma_reg @ w

    def grad(w):
        return Sigma_reg @ w

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    if target_return is not None and mu is not None:
        constraints.append({"type": "eq", "fun": lambda w: w @ mu - target_return})

    bounds = [(0.0, 1.0) for _ in range(n)]
    w0 = np.ones(n) / n
    result = minimize(objective, w0, jac=grad, method="SLSQP",
                      bounds=bounds, constraints=constraints,
                      options={"ftol": 1e-12, "maxiter": 1000})
    w = result.x
    w = np.maximum(w, 0.0)
    w = w / np.sum(w)
    risk = np.sqrt(w @ Sigma_reg @ w)
    out = {"weights": w, "risk": float(risk), "success": result.success}
    if mu is not None:
        out["expected_return"] = float(w @ mu)
    return out


def risk_parity_weights(Sigma: np.ndarray,
                        risk_budget: np.ndarray = None,
                        max_iter: int = 1000,
                        tol: float = 1e-9) -> dict:
    n = Sigma.shape[0]
    if Sigma.shape != (n, n):
        raise ValueError("risk_parity_weights: Sigma 必须是方阵。")
    if risk_budget is None:
        risk_budget = np.ones(n) / n
    risk_budget = np.asarray(risk_budget, dtype=float)
    if not np.isclose(np.sum(risk_budget), 1.0):
        risk_budget = risk_budget / np.sum(risk_budget)
    if np.any(risk_budget <= 0):
        raise ValueError("risk_parity_weights: 风险预算必须为正数。")


    Sigma_reg = Sigma + 1e-8 * np.eye(n)


    x = np.sqrt(np.diag(Sigma_reg))
    x = x / np.sum(x)
    if np.any(x <= 0):
        x = np.ones(n) / n













    raise NotImplementedError("Hole 1: 风险平价CCD迭代核心待实现")

    w = x / np.sum(x)
    port_var = w @ Sigma_reg @ w
    if port_var < 1e-15:
        port_var = 1e-15
    port_risk = np.sqrt(port_var)
    marginal_risk = Sigma_reg @ w
    rc = w * marginal_risk / port_risk


    vol = np.sqrt(np.diag(Sigma_reg))
    dr = (w @ vol) / port_risk if port_risk > 1e-15 else 0.0

    return {
        "weights": w,
        "risk_contributions": rc,
        "risk": float(port_risk),
        "diversification_ratio": float(dr),
        "iterations": iteration + 1,
        "converged": iteration < max_iter - 1,
    }


def herfindahl_risk_concentration(rc: np.ndarray) -> float:
    total = np.sum(rc)
    if total < 1e-15:
        return 1.0
    shares = rc / total
    return float(np.sum(shares ** 2))


def effective_number_of_bets(rc: np.ndarray) -> float:
    total = np.sum(rc)
    if total < 1e-15:
        return 1.0
    p = rc / total
    p = p[p > 1e-15]
    entropy = -np.sum(p * np.log(p))
    return float(np.exp(entropy))


def risk_parity_with_budget_constraints(Sigma: np.ndarray,
                                         lower: np.ndarray = None,
                                         upper: np.ndarray = None,
                                         risk_budget: np.ndarray = None,
                                         max_iter: int = 1000) -> dict:
    n = Sigma.shape[0]
    if risk_budget is None:
        risk_budget = np.ones(n) / n
    if lower is None:
        lower = np.zeros(n)
    if upper is None:
        upper = np.ones(n)

    Sigma_reg = Sigma + 1e-8 * np.eye(n)
    x = np.ones(n) / n
    alpha_step = 0.1

    for iteration in range(max_iter):
        grad = Sigma_reg @ x - risk_budget / np.maximum(x, 1e-15)
        x_new = x - alpha_step * grad

        x_new = np.clip(x_new, lower, upper)

        x_new = np.maximum(x_new, 0)
        s = np.sum(x_new)
        if s > 0:
            x_new = x_new / s
        if np.linalg.norm(x_new - x) < 1e-9:
            break
        x = x_new

    w = x / np.sum(x)
    port_risk = np.sqrt(max(w @ Sigma_reg @ w, 1e-15))
    rc = w * (Sigma_reg @ w) / port_risk
    return {
        "weights": w,
        "risk_contributions": rc,
        "risk": float(port_risk),
        "iterations": iteration + 1,
    }
