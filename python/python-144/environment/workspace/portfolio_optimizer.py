"""
portfolio_optimizer.py
投资组合优化核心模块：风险平价、最小方差与分散化比率优化。

科学背景：
传统马科维茨均值-方差优化对参数估计误差极度敏感，
尤其在资产维度较高时会产生极端权重集中（Markowitz诅咒）。
风险平价（Risk Parity）策略通过使各资产对组合总风险的边际贡献相等，
实现真正的风险分散化。本项目实现了基于牛顿迭代与循环坐标下降（CCD）
的鲁棒风险平价求解器，并引入分散化比率作为辅助目标函数。

核心数学模型：
1. 马科维茨最小方差组合：
    min_w  w^T Σ w
    s.t.   w^T 1 = 1, w ≥ 0。

2. 风险平价组合（Roncalli, 2013）：
    定义资产 i 的风险贡献为
        RC_i(w) = w_i (Σ w)_i / sqrt(w^T Σ w)。
    风险平价要求
        RC_i(w) = b_i * σ(w)，
    其中 b_i 为预设风险预算（通常取等权重 b_i = 1/n）。
    等价于求解非线性方程组：
        w_i (Σ w)_i = b_i (w^T Σ w)。

3. 分散化比率（Choueifaty & Coignard, 2008）：
        DR(w) = (w^T σ) / sqrt(w^T Σ w)。
    最大化 DR(w) 等价于寻找"最分散化"的组合。
"""

import numpy as np
from scipy.optimize import minimize


def markowitz_min_variance(Sigma: np.ndarray,
                           target_return: float = None,
                           mu: np.ndarray = None) -> dict:
    """
    求解带非负约束的最小方差组合。

    优化问题：
        min_w   0.5 * w^T Σ w
        s.t.    Σ_i w_i = 1
                w_i ≥ 0。

    参数
    ----------
    Sigma : np.ndarray, shape (n, n)
        协方差矩阵（半正定）。
    target_return : float, optional
        目标收益率约束（若提供，则增加 w^T μ = target_return）。
    mu : np.ndarray, optional
        预期收益率向量。

    返回
    -------
    dict
        包含 'weights'、'risk'、'expected_return'（若 mu 提供）的字典。
    """
    n = Sigma.shape[0]
    if Sigma.shape != (n, n):
        raise ValueError("markowitz_min_variance: Sigma 必须是方阵。")
    # 正则化
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
    """
    基于循环坐标下降（CCD）算法求解风险平价组合权重。

    迭代格式（Spinu, 2013）：
    对每个坐标 i，更新
        x_i^{k+1} = sqrt( b_i * (Σ_{j≠i} Σ_{ij} x_j^k )^{-1} )，
    其中 x_i = w_i / sqrt(w^T Σ w)。
    最终权重 w_i = x_i / Σ_j x_j。

    该算法具有线性收敛速度，且对初始化不敏感。

    参数
    ----------
    Sigma : np.ndarray, shape (n, n)
        协方差矩阵。
    risk_budget : np.ndarray, shape (n,)
        风险预算向量，默认等风险预算。
    max_iter : int
        最大迭代次数。
    tol : float
        收敛容差。

    返回
    -------
    dict
        包含 'weights'、'risk_contributions'、'risk'、'diversification_ratio'。
    """
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

    # 正则化协方差矩阵
    Sigma_reg = Sigma + 1e-8 * np.eye(n)

    # 初始化
    x = np.sqrt(np.diag(Sigma_reg))
    x = x / np.sum(x)
    if np.any(x <= 0):
        x = np.ones(n) / n

    # === HOLE 1 BEGIN ===
    # TODO: 实现风险平价循环坐标下降（CCD）核心迭代
    # 科学知识：风险平价循环坐标下降算法（Spinu, 2013）
    # 迭代格式：x_i^{k+1} = sqrt( b_i / (Σ_{j≠i} Σ_{ij} x_j^k) )
    # 其中 b_i = risk_budget[i]，Σ = Sigma_reg，n = Sigma.shape[0]
    # 每轮迭代需：
    #   1) 对每坐标 i，计算 ax = Σ_{j≠i} Σ_{ij} x_j，更新 x_i = sqrt(b_i / max(ax, 1e-15))
    #   2) 归一化 x = x / sum(x)
    #   3) 若 ||x - x_old||_1 < tol 则收敛退出
    # 可用变量：Sigma_reg, x, risk_budget, max_iter, tol, n
    # 需更新变量：x, iteration（实际迭代次数）
    # === HOLE 1 END ===
    raise NotImplementedError("Hole 1: 风险平价CCD迭代核心待实现")

    w = x / np.sum(x)
    port_var = w @ Sigma_reg @ w
    if port_var < 1e-15:
        port_var = 1e-15
    port_risk = np.sqrt(port_var)
    marginal_risk = Sigma_reg @ w
    rc = w * marginal_risk / port_risk

    # 分散化比率
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
    """
    计算风险贡献的Herfindahl集中度指数：

        H = Σ_i (RC_i / σ_p)^2。

    对风险平价组合，H ≈ 1/n；对完全集中组合，H = 1。
    """
    total = np.sum(rc)
    if total < 1e-15:
        return 1.0
    shares = rc / total
    return float(np.sum(shares ** 2))


def effective_number_of_bets(rc: np.ndarray) -> float:
    """
    计算有效赌注数（Effective Number of Bets, ENB）：

        ENB = exp( - Σ_i p_i log p_i )，
    其中 p_i = RC_i / Σ_j RC_j。

    该指标由 Meucci (2009) 提出，衡量组合的真实分散化程度。
    ENB = n 表示完全风险平价；ENB = 1 表示完全集中。
    """
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
    """
    带上下界约束的风险平价优化（投影梯度法）。

    优化问题：
        min_x   0.5 x^T Σ x - Σ_i b_i log(x_i)
        s.t.    l_i ≤ x_i ≤ u_i。

    该目标函数为凸函数，其最优解满足风险平价条件。
    """
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
        # 投影到可行域
        x_new = np.clip(x_new, lower, upper)
        # 再投影到单纯形（近似）
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
