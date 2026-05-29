"""
simplex_search.py
单纯形格点枚举与高维组合优化搜索模块。

融入的原项目核心算法：
- 054_asa299: 单纯形格点枚举（simplex_lattice_point_next）
- 1236_tet_mesh_quality: 四面体质量评估（体积、行列式）

科学背景：
在投资组合优化中，当需要考虑整数约束（如最小交易单位）或
需要系统性地探索可行域时，单纯形格点枚举提供了一种遍历
标准单纯形上所有有理点的方法。此外，四面体体积可作为
高维分散度的几何度量：协方差矩阵对应的并行多面体体积
越大，组合的风险分散度越高。
"""

import numpy as np
from itertools import combinations


def simplex_lattice_points(n: int, t: int) -> np.ndarray:
    """
    枚举 n 维标准单纯形中所有整数格点，满足
        x_i ≥ 0,   Σ_i x_i = t。

    算法（Chasalow & Brand, AS 299）：
    从 x = (t, 0, ..., 0) 开始，按逆字典序生成下一个格点。
    对当前点 x，找到最右边满足 x_j > 0 的分量（j < n），
    令 x_j ← x_j - 1，x_{j+1} ← t - Σ_{i=1}^j x_i，
    x_{j+2} = ... = x_n = 0。

    参数
    ----------
    n : int
        维度（资产数）。
    t : int
        格点特征数（总和）。

    返回
    -------
    np.ndarray, shape (m, n)
        所有格点构成的矩阵，其中 m = C(n+t-1, t)。
    """
    if n < 1:
        raise ValueError("simplex_lattice_points: n 必须为正整数。")
    if t < 0:
        raise ValueError("simplex_lattice_points: t 必须为非负整数。")
    points = []
    x = np.zeros(n, dtype=int)
    x[0] = t
    points.append(x.copy())
    if n == 1:
        return np.array(points, dtype=int)
    while True:
        # 寻找最右边满足 x[j] > 0 的分量（j < n-1）
        j = n - 1
        for i in range(n - 2, -1, -1):
            if x[i] > 0:
                j = i
                break
        if j == n - 1:
            break
        x[j] -= 1
        x[j + 1] = t - np.sum(x[:j + 1])
        x[j + 2:] = 0
        points.append(x.copy())
    return np.array(points, dtype=int)


def simplex_volume(points: np.ndarray) -> float:
    """
    计算单纯形的体积。

    对 d 维空间中的 d+1 个点 {p_0, p_1, ..., p_d}，
    体积公式为
        V = |det([p_1-p_0, p_2-p_0, ..., p_d-p_0])| / d!。

    参数
    ----------
    points : np.ndarray, shape (d+1, d)
        单纯形顶点。

    返回
    -------
    float
        体积。
    """
    d = points.shape[1]
    if points.shape[0] != d + 1:
        raise ValueError("simplex_volume: 顶点数必须等于维度+1。")
    M = np.zeros((d, d))
    for i in range(d):
        M[:, i] = points[i + 1, :] - points[0, :]
    det = np.linalg.det(M)
    vol = abs(det) / np.math.factorial(d)
    return float(vol)


def covariance_simplex_volume(Sigma: np.ndarray) -> float:
    """
    将协方差矩阵视为线性变换，计算其对应的并行多面体体积。

    对 d×d 正定矩阵 Σ，Cholesky 分解 Σ = L L^T，
    则 L 的列向量张成的并行多面体体积为 det(L) = sqrt(det(Σ))。
    该体积可解释为"风险空间"中的广义体积：体积越大，
    风险椭球覆盖的范围越广，潜在的分散化空间越大。
    """
    d = Sigma.shape[0]
    if Sigma.shape != (d, d):
        raise ValueError("covariance_simplex_volume: 输入必须是方阵。")
    try:
        L = np.linalg.cholesky(Sigma)
        vol = np.prod(np.diag(L))
    except np.linalg.LinAlgError:
        eigvals = np.linalg.eigvalsh(Sigma)
        vol = np.sqrt(np.prod(np.maximum(eigvals, 1e-15)))
    return float(vol)


def tet_quality_indicator_from_cov(Sigma: np.ndarray) -> dict:
    """
    将四面体质量指标映射到协方差矩阵的质量评估。

    对 3×3 子矩阵，计算其对应的四面体质量指标：
    - 体积-边长比（Radius Ratio）
    - 条件数

    返回字典包含多种质量度量。
    """
    d = Sigma.shape[0]
    # 取前3个资产（若不足3则取最小维度）
    k = min(3, d)
    sub = Sigma[:k, :k]
    det_sub = np.linalg.det(sub)
    trace_sub = np.trace(sub)
    cond = np.linalg.cond(sub)
    return {
        "sub_determinant": float(det_sub),
        "sub_trace": float(trace_sub),
        "condition_number": float(cond),
        "quality_score": float(det_sub / (trace_sub ** k + 1e-15)),
    }


def lattice_portfolio_search(n_assets: int, t: int,
                              Sigma: np.ndarray,
                              mu: np.ndarray = None) -> dict:
    """
    在单纯形格点上搜索最优投资组合权重。

    对格点 w = x / t（其中 x 为单纯形格点），
    计算组合风险 σ(w) = sqrt(w^T Σ w)，
    返回风险最小的组合。

    若提供 mu，则计算夏普比率近似并返回最大夏普比率组合。

    参数
    ----------
    n_assets : int
        资产数。
    t : int
        格点特征数（越大搜索越精细）。
    Sigma : np.ndarray
        协方差矩阵。
    mu : np.ndarray, optional
        预期收益率。

    返回
    -------
    dict
        最优格点组合及其风险/收益指标。
    """
    if n_assets < 1 or t < 1:
        raise ValueError("lattice_portfolio_search: n_assets 和 t 必须为正整数。")
    points = simplex_lattice_points(n_assets, t)
    weights = points / float(t)
    best_idx = -1
    best_score = np.inf
    best_sharpe = -np.inf

    Sigma_reg = Sigma + 1e-8 * np.eye(n_assets)
    risks = np.sqrt(np.maximum(np.sum(weights @ Sigma_reg * weights, axis=1), 1e-15))

    if mu is not None:
        returns = weights @ mu
        sharpes = returns / risks
        best_idx = int(np.argmax(sharpes))
        best_sharpe = float(sharpes[best_idx])
        best_score = float(risks[best_idx])
    else:
        best_idx = int(np.argmin(risks))
        best_score = float(risks[best_idx])

    return {
        "optimal_weights": weights[best_idx, :],
        "optimal_risk": best_score,
        "optimal_sharpe": best_sharpe,
        "n_points_evaluated": len(points),
        "grid_resolution": t,
    }


def mesh_base_one(element_node: np.ndarray, node_num: int) -> np.ndarray:
    """
    检测并修正元素定义的索引基准（0-based 转 1-based）。

    来自 tet_mesh_quality 的 mesh_base_one 思想。
    """
    node_min = np.min(element_node)
    node_max = np.max(element_node)
    if node_min == 0 and node_max == node_num - 1:
        return element_node + 1
    elif node_min == 1 and node_max == node_num:
        return element_node
    else:
        raise ValueError("mesh_base_one: 无法识别的索引类型。")
