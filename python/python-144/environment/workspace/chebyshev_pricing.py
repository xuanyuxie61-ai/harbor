"""
chebyshev_pricing.py
基于Chebyshev谱方法的高精度风险度量计算模块。

融入的原项目核心算法：
- 161_chebyshev_matrix: Chebyshev微分矩阵与Chebyshev网格

科学背景：
在金融工程中，投资组合的风险度量（VaR、CVaR）依赖于收益率分布的分位数。
传统线性插值精度低，Chebyshev谱方法在光滑函数上具有指数收敛性，
能够高精度地重构收益率的累积分布函数（CDF）并计算其导数（PDF）。
"""

import numpy as np
from scipy.special import gamma as scipy_gamma


def chebyshev_grid(n: int) -> np.ndarray:
    """
    生成 n+1 个Chebyshev节点：
        x_i = cos(π i / n),  i = 0, 1, ..., n。

    这些节点是区间 [-1, 1] 上使插值误差最小化的点，
    满足极小极大性质：
        max |ω_n(x)| = 2^{1-n} / n!
    其中 ω_n(x) = ∏ (x - x_i)。
    """
    if n < 0:
        raise ValueError("chebyshev_grid: n 必须为非负整数。")
    if n == 0:
        return np.array([1.0])
    return np.cos(np.pi * np.arange(n + 1) / n)


def chebyshev_diff_matrix(n: int) -> np.ndarray:
    """
    构造 Chebyshev 谱微分矩阵 D，满足
        w = D @ v
    其中 v 为网格函数值，w 为其在Chebyshev节点上的导数值。

    数学推导（Trefethen, Spectral Methods in MATLAB）：
    设 x_i = cos(π i / n)，c_0 = c_n = 2，c_i = 1 (1 ≤ i ≤ n-1)。
    非对角元：
        D_{ij} = (c_i / c_j) * (-1)^{i+j} / (x_i - x_j),   i ≠ j。
    对角元通过行和为零确定：
        D_{ii} = - Σ_{j≠i} D_{ij}。
    """
    if n < 0:
        raise ValueError("chebyshev_diff_matrix: n 必须为非负整数。")
    if n == 0:
        return np.zeros((1, 1))
    x = chebyshev_grid(n)
    c = np.ones(n + 1)
    c[0] = 2.0
    c[-1] = 2.0
    c = c * ((-1.0) ** np.arange(n + 1))
    X = np.tile(x[:, np.newaxis], (1, n + 1))
    dX = X - X.T
    # 非对角元
    D = (c[:, np.newaxis] / c[np.newaxis, :]) / (dX + np.eye(n + 1))
    # 对角元：行和为零
    D = D - np.diag(np.sum(D, axis=1))
    return D


def chebyshev_barycentric_interpolate(x_grid: np.ndarray, v: np.ndarray,
                                       x_query: np.ndarray) -> np.ndarray:
    """
    使用重心Lagrange插值公式在Chebyshev节点上进行插值：

        p(x) = Σ_{j=0}^n [w_j / (x - x_j)] v_j / Σ_{j=0}^n [w_j / (x - x_j)]

    其中 Chebyshev 权重 w_0 = w_n = 0.5，w_j = (-1)^j (1 < j < n)。
    该公式具有 O(n) 计算复杂度且在多项式空间内数值稳定。
    """
    n = len(x_grid) - 1
    w = np.ones(n + 1) * ((-1.0) ** np.arange(n + 1))
    w[0] = 0.5
    w[-1] = 0.5 * ((-1.0) ** n)

    # 避免除以零
    x_query = np.asarray(x_query).reshape(-1)
    result = np.zeros_like(x_query, dtype=float)
    for i, xq in enumerate(x_query):
        exact = np.isclose(xq, x_grid)
        if np.any(exact):
            result[i] = v[np.argmax(exact)]
            continue
        weights = w / (xq - x_grid)
        result[i] = np.dot(weights, v) / np.sum(weights)
    return result


def spectral_var_cvar(returns: np.ndarray, alpha: float = 0.05,
                       n_cheb: int = 64) -> dict:
    """
    基于Chebyshev谱方法计算投资组合的 VaR 与 CVaR。

    数学模型：
    设收益率样本为 {r_i}_{i=1}^N，经验CDF为
        F_N(r) = (1/N) Σ_{i=1}^N I(r_i ≤ r)。
    VaR_α 定义为 F_N 的 α-分位数：
        VaR_α = inf{ r ∈ ℝ | F_N(r) ≥ α }。
    CVaR_α（Expected Shortfall）定义为
        CVaR_α = E[ r | r ≤ VaR_α ]
               = (1/α) ∫_{-∞}^{VaR_α} r dF_N(r)。

    算法步骤：
    1. 将收益率样本映射到 [-1, 1] 区间。
    2. 在Chebyshev节点上构造经验CDF的插值。
    3. 利用谱微分矩阵计算PDF，验证非负性。
    4. 通过重心插值高精度求解 F_N(r) = α 的根，得到 VaR。
    5. 数值积分计算 CVaR。

    参数
    ----------
    returns : np.ndarray
        收益率样本序列。
    alpha : float
        置信水平，默认 0.05（5%尾部）。
    n_cheb : int
        Chebyshev节点数，默认 64。

    返回
    -------
    dict
        包含 'VaR'、'CVaR'、'mean'、'std'、'spectral_nodes'、
        'cdf_values' 的风险度量字典。
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError("spectral_var_cvar: alpha 必须在 (0, 1) 区间内。")
    if returns.size < 10:
        raise ValueError("spectral_var_cvar: 样本量不足（至少10个）。")

    r_min = np.min(returns)
    r_max = np.max(returns)
    # 扩展边界以确保插值稳定性
    margin = 0.1 * max(abs(r_max), abs(r_min), 1e-6)
    r_lo = r_min - margin
    r_hi = r_max + margin

    # 仿射变换到 [-1, 1]
    def to_std(r):
        return (2.0 * r - (r_hi + r_lo)) / (r_hi - r_lo)

    def from_std(x):
        return 0.5 * ((r_hi - r_lo) * x + (r_hi + r_lo))

    # Chebyshev节点（标准区间）
    x_nodes = chebyshev_grid(n_cheb)
    r_nodes = from_std(x_nodes)

    # 经验CDF在节点处的值
    cdf_nodes = np.array([np.mean(returns <= r) for r in r_nodes])
    # 单调性修正
    cdf_nodes = np.maximum.accumulate(np.minimum.accumulate(cdf_nodes))
    cdf_nodes = np.clip(cdf_nodes, 0.0, 1.0)

    # 谱微分矩阵计算PDF
    D = chebyshev_diff_matrix(n_cheb)
    # 链式法则：d/dr = d/dx * dx/dr = d/dx * 2/(r_hi - r_lo)
    pdf_nodes = D @ cdf_nodes * (2.0 / (r_hi - r_lo))
    pdf_nodes = np.maximum(pdf_nodes, 0.0)  # 非负性截断

    # 使用重心插值求解 F(r) = alpha
    # 二分搜索结合插值
    x_left, x_right = -1.0, 1.0
    for _ in range(60):
        x_mid = 0.5 * (x_left + x_right)
        f_mid = chebyshev_barycentric_interpolate(x_nodes, cdf_nodes,
                                                   np.array([x_mid]))[0]
        if f_mid < alpha:
            x_left = x_mid
        else:
            x_right = x_mid
        if x_right - x_left < 1e-14:
            break
    x_var = 0.5 * (x_left + x_right)
    var_val = from_std(x_var)

    # 计算 CVaR：对尾部区域数值积分
    # 在Chebyshev节点上计算条件期望
    tail_mask = r_nodes <= var_val
    cvar_val = None
    if not np.any(tail_mask):
        cvar_val = var_val
    else:
        # 梯形法则在Chebyshev节点上积分（节点非均匀）
        x_tail = x_nodes[tail_mask]
        r_tail = r_nodes[tail_mask]
        pdf_tail = pdf_nodes[tail_mask]
        # 排序确保单调
        order = np.argsort(x_tail)
        x_tail = x_tail[order]
        r_tail = r_tail[order]
        pdf_tail = pdf_tail[order]
        # 积分 ∫ r * f(r) dr，利用 dr = (r_hi - r_lo)/2 dx
        integrand = r_tail * pdf_tail * ((r_hi - r_lo) / 2.0)
        integral = np.trapezoid(integrand, x_tail)
        # CDF在var处的值（可能不完全等于alpha）
        cdf_at_var = chebyshev_barycentric_interpolate(
            x_nodes, cdf_nodes, np.array([x_var]))[0]
        if cdf_at_var < 1e-12:
            cvar_val = var_val
        else:
            cvar_val = integral / cdf_at_var

    # 数值鲁棒性：若谱积分异常，回退到经验估计
    tail_returns = returns[returns <= var_val]
    if len(tail_returns) == 0:
        empirical_cvar = var_val
    else:
        empirical_cvar = np.mean(tail_returns)
    if (cvar_val is None or not np.isfinite(cvar_val)
            or abs(cvar_val) > 10 * max(abs(var_val), 1e-6)
            or abs(cvar_val) < 1e-12):
        cvar_val = empirical_cvar

    return {
        "VaR": float(var_val),
        "CVaR": float(cvar_val),
        "mean": float(np.mean(returns)),
        "std": float(np.std(returns)),
        "alpha": alpha,
        "spectral_nodes": r_nodes,
        "cdf_values": cdf_nodes,
        "pdf_values": pdf_nodes,
    }


def circle01_monomial_integral(e: np.ndarray) -> float:
    """
    计算单位圆周上的单变量积分：
        ∫_{S^1} x^{e_1} y^{e_2} ds。

    解析公式（Davis & Rabinowitz, 1984）：
    若 e_1 或 e_2 为奇数，积分值为 0。
    否则
        I = 2 * Γ((e_1+1)/2) * Γ((e_2+1)/2) / Γ((e_1+e_2+2)/2)。

    参数
    ----------
    e : np.ndarray, shape (2,)
        指数向量。

    返回
    -------
    float
        积分值。
    """
    if np.any(e < 0):
        raise ValueError("circle01_monomial_integral: 指数必须为非负整数。")
    if np.any(e % 2 == 1):
        return 0.0
    val = 2.0
    for i in range(2):
        val *= scipy_gamma(0.5 * (e[i] + 1))
    val /= scipy_gamma(0.5 * np.sum(e + 1))
    return float(val)
