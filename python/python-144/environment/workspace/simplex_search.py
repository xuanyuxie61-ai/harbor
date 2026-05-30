
import numpy as np
from itertools import combinations


def simplex_lattice_points(n: int, t: int) -> np.ndarray:
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
    d = Sigma.shape[0]

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
    node_min = np.min(element_node)
    node_max = np.max(element_node)
    if node_min == 0 and node_max == node_num - 1:
        return element_node + 1
    elif node_min == 1 and node_max == node_num:
        return element_node
    else:
        raise ValueError("mesh_base_one: 无法识别的索引类型。")
