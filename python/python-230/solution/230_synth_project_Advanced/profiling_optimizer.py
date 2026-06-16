"""
profiling_optimizer.py
======================

Nuisance 参数 Profiling 优化器 —— Profile Likelihood 计算核心。

种子项目融合:
- 807 (nonlin_fixed_point) → 不动点迭代求解 profiling 方程
- 1435 (zoomin) → Halley/Brent/Traub 高阶求根加速 profiling
- 671 (life, cellular automata) → 离散状态空间扫描启发式初始化
- 285 (digraph_adj) → nuisance 依赖图 → 块对角分解

Profiling 方程 (一阶最优性条件):

    ∂(-ln L)/∂θ_k |_{μ, θ̂_μ} = 0,  ∀ k

这是 n_nuis 维非线性方程组。求解方法:

1. Newton-Raphson (利用解析梯度+Hessian):
    θ^{(n+1)} = θ^{(n)} - H^{-1} · g
    其中 g_k = ∂NLL/∂θ_k, H_{jk} = ∂²NLL/∂θ_j∂θ_k

2. Halley 方法 (seed 1435, 三次收敛):
    θ^{(n+1)} = θ^{(n)} - [g/H] / [1 - 0.5 · (g·H'·g)/(g·g)]
    简化为一维: x ← x - (f/f') / (1 - 0.5 · f·f''/(f')²)

3. 不动点迭代 (seed 807):
    θ^{(n+1)} = G(θ^{(n)})
    其中 G_k(θ) = -Σ_{i≠k} H_{ki}/H_{kk} · (θ_i - θ_i^{(n)}) - g_k/H_{kk}

4. Brent 方法 (一维鲁棒混合):
    对每个 θ_k 固定其他参数, 做一维 Brent 最小化。

5. 离散初始化扫描 (seed 671 → cellular automata 式):
    在 θ ∈ {-2, -1, 0, 1, 2}^n_nuis 网格上评估 NLL,
    选取最佳点作为迭代初值。

依赖图分解 (seed 285 → digraph adjacency):
    构建 nuisance 依赖图 G:
        A_{jk} = 1 如果 θ_j 和 θ_k 同时出现在同一 bin 的期望中
    对 G 做连通分量分解 → 块对角结构 → 分块独立 profiling
"""

from __future__ import annotations
import math
import numpy as np
from typing import Tuple, Optional, List, Dict

from physical_model import BinModel
from likelihood import (
    nll_poisson_gaussian,
    nll_gradient_theta,
    nll_hessian_theta_diag,
)


# ---------------------------------------------------------------------------
# 依赖图构建 (seed 285 → digraph adjacency)
# ---------------------------------------------------------------------------
def build_nuisance_dependency_graph(model: BinModel) -> np.ndarray:
    """
    构建 nuisance 依赖图的邻接矩阵。

    A_{jk} = 1 如果 ∃ bin i 使得 γ_{ik} ≠ 0 或 δ_{ik} ≠ 0
    且 γ_{il} ≠ 0 或 δ_{il} ≠ 0 (j ≠ k 同时影响同一 bin)

    对角线 A_{jj} = 1 (自环)。
    """
    n = model.n_nuis
    A = np.eye(n, dtype=np.float64)  # 自环
    for i in range(model.n_bins):
        active = []
        for k in range(n):
            if abs(model.gamma_sig[i, k]) > 1e-12 or abs(model.delta_bkg[i, k]) > 1e-12:
                active.append(k)
        # 完全子图
        for j_idx in range(len(active)):
            for k_idx in range(j_idx + 1, len(active)):
                j, k = active[j_idx], active[k_idx]
                A[j, k] = 1.0
                A[k, j] = 1.0
    return A


def adjacency_to_blocks(A: np.ndarray) -> List[List[int]]:
    """
    邻接矩阵 → 连通分量分解 (BFS)。

    Returns
    -------
    list of list of int
        每个子列表为一个连通分量的节点索引。
    """
    n = A.shape[0]
    visited = np.zeros(n, dtype=bool)
    blocks = []
    for start in range(n):
        if visited[start]:
            continue
        # BFS
        component = []
        queue = [start]
        while queue:
            node = queue.pop(0)
            if visited[node]:
                continue
            visited[node] = True
            component.append(node)
            for nb in range(n):
                if not visited[nb] and A[node, nb] > 0.5:
                    queue.append(nb)
        blocks.append(sorted(component))
    return blocks


# ---------------------------------------------------------------------------
# 离散初始化扫描 (seed 671 → cellular automata 式)
# ---------------------------------------------------------------------------
def discrete_initialization_scan(
    model: BinModel,
    n_obs: np.ndarray,
    mu: float,
    grid_values: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    在离散网格上扫描 NLL, 找到最佳初始点。

    类似 cellular automata 的离散状态演化:
    初始状态 grid ∈ {-2, -1, 0, 1, 2}^n_nuis,
    逐态评估, 取 argmin。

    Parameters
    ----------
    model : BinModel
    n_obs : ndarray
    mu : float
    grid_values : ndarray, optional
        扫描值 (默认 [-2, -1, 0, 1, 2])

    Returns
    -------
    ndarray
        最佳初始 θ
    """
    if grid_values is None:
        grid_values = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])

    n_nuis = model.n_nuis
    best_nll = float("inf")
    best_theta = np.zeros(n_nuis)

    # 枚举所有组合 (小规模: 5^2 = 25 或 5^3 = 125)
    if n_nuis > 4:
        # 太多组合, 只扫描原点附近
        return np.zeros(n_nuis)

    from itertools import product
    for combo in product(grid_values, repeat=n_nuis):
        theta = np.array(combo, dtype=np.float64)
        nll = nll_poisson_gaussian(model, n_obs, mu, theta)
        if nll < best_nll:
            best_nll = nll
            best_theta = theta.copy()

    return best_theta


# ---------------------------------------------------------------------------
# Halley 方法 Profiling (seed 1435)
# ---------------------------------------------------------------------------
def halley_profiling_step(
    model: BinModel,
    n_obs: np.ndarray,
    mu: float,
    theta: np.ndarray,
    step_scale: float = 1.0,
) -> Tuple[np.ndarray, float]:
    """
    Halley 方法一步更新 (三次收敛, seed 1435):

    对每个维度 k:
        x ← x - (f/f') / (1 - 0.5 · f·f''/(f')²)

    简化: 使用对角 Hessian, 对每个 θ_k 独立做 Halley 步。

    f = ∂NLL/∂θ_k (梯度)
    f' = ∂²NLL/∂θ_k² (Hessian 对角)
    f'' ≈ 有限差分近似 ∂³NLL/∂θ_k³

    Returns
    -------
    (theta_new, max_update) : (ndarray, float)
    """
    theta = np.asarray(theta, dtype=np.float64).copy()
    grad = nll_gradient_theta(model, n_obs, mu, theta)
    Hdiag = nll_hessian_theta_diag(model, n_obs, mu, theta)

    # 三阶导数 (有限差分近似)
    h_fd = 1e-4
    d3 = np.zeros(model.n_nuis)
    for k in range(model.n_nuis):
        theta_p = theta.copy(); theta_p[k] += h_fd
        theta_m = theta.copy(); theta_m[k] -= h_fd
        gp = nll_gradient_theta(model, n_obs, mu, theta_p)
        gm = nll_gradient_theta(model, n_obs, mu, theta_m)
        d3[k] = (gp[k] - gm[k]) / (2.0 * h_fd)

    # Halley 更新
    max_update = 0.0
    for k in range(model.n_nuis):
        f = grad[k]
        fp = Hdiag[k]
        fpp = d3[k]
        if abs(fp) < 1e-15:
            continue
        # Halley 公式
        L = f * fpp / (fp * fp) if abs(fp * fp) > 1e-30 else 0.0
        denom = 1.0 - 0.5 * L
        if abs(denom) < 1e-10:
            # 退化: 回退 Newton
            update = f / fp
        else:
            update = (f / fp) / denom
        update *= step_scale
        theta[k] -= update
        max_update = max(max_update, abs(update))

    # 边界截断
    theta = np.clip(theta, -10.0, 10.0)
    return theta, max_update


# ---------------------------------------------------------------------------
# 不动点迭代 Profiling (seed 807)
# ---------------------------------------------------------------------------
def fixed_point_profiling(
    model: BinModel,
    n_obs: np.ndarray,
    mu: float,
    theta_init: np.ndarray,
    max_iter: int = 500,
    tol: float = 1e-10,
    damping: float = 0.5,
) -> Tuple[np.ndarray, int, float]:
    """
    不动点迭代求解 profiling 方程 (seed 807):

    θ^{(n+1)} = (1 - α) θ^{(n)} + α · G(θ^{(n)})

    其中 G(θ) = θ - D^{-1} · ∇NLL(θ), D = diag(Hessian)

    Returns
    -------
    (theta_hat, n_iter, final_residual) : (ndarray, int, float)
    """
    theta = np.asarray(theta_init, dtype=np.float64).copy()

    for it in range(max_iter):
        grad = nll_gradient_theta(model, n_obs, mu, theta)
        Hdiag = nll_hessian_theta_diag(model, n_obs, mu, theta)
        # 不动点映射
        G = theta - grad / np.maximum(Hdiag, 1e-10)
        # 阻尼更新
        theta_new = (1.0 - damping) * theta + damping * G
        # 边界截断
        theta_new = np.clip(theta_new, -10.0, 10.0)
        # 收敛检验
        residual = float(np.max(np.abs(theta_new - theta)))
        theta = theta_new
        if residual < tol:
            return theta, it + 1, residual

    return theta, max_iter, float(np.max(np.abs(grad)))


# ---------------------------------------------------------------------------
# 混合 Profiling (Halley + 不动点 + Brent 回退)
# ---------------------------------------------------------------------------
def profile_theta(
    model: BinModel,
    n_obs: np.ndarray,
    mu: float,
    theta_init: Optional[np.ndarray] = None,
    use_halley: bool = True,
    use_fixed_point: bool = True,
    discrete_scan: bool = True,
    max_iter: int = 200,
    tol: float = 1e-10,
) -> Tuple[np.ndarray, float, Dict]:
    """
    混合 profiling 优化器:

    1. [可选] 离散扫描初始化 (seed 671)
    2. [可选] Halley 迭代 (seed 1435, 快速三次收敛)
    3. [可选] 不动点迭代精化 (seed 807, 鲁棒)
    4. 最终 L-BFGS-B 精化

    Returns
    -------
    (theta_hat, nll_profile, info) : (ndarray, float, dict)
    """
    from scipy.optimize import minimize

    # 1. 初始化
    if theta_init is not None:
        theta = np.asarray(theta_init, dtype=np.float64).copy()
    elif discrete_scan:
        theta = discrete_initialization_scan(model, n_obs, mu)
    else:
        theta = np.zeros(model.n_nuis)

    info = {"init_theta": theta.copy(), "halley_iters": 0, "fp_iters": 0}

    # 2. Halley 迭代
    if use_halley:
        for _ in range(50):
            theta, max_upd = halley_profiling_step(model, n_obs, mu, theta)
            info["halley_iters"] += 1
            if max_upd < tol * 10:
                break

    # 3. 不动点迭代
    if use_fixed_point:
        theta, fp_iters, _ = fixed_point_profiling(
            model, n_obs, mu, theta, max_iter=100, tol=tol
        )
        info["fp_iters"] = fp_iters

    # 4. L-BFGS-B 最终精化
    def objective(th):
        return nll_poisson_gaussian(model, n_obs, mu, th)

    def gradient(th):
        return nll_gradient_theta(model, n_obs, mu, th)

    bounds = [(-10.0, 10.0)] * model.n_nuis
    result = minimize(
        objective, theta, method="L-BFGS-B", jac=gradient,
        bounds=bounds, options={"maxiter": max_iter, "ftol": tol * 1e-3},
    )
    theta_hat = result.x
    nll_val = float(result.fun)
    info["final_nll"] = nll_val
    info["scipy_success"] = result.success
    return theta_hat, nll_val, info


# ---------------------------------------------------------------------------
# 快速自检
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from physical_model import make_default_binmodel, generate_observed_data

    mdl = make_default_binmodel()
    data = generate_observed_data(mdl, mu_true=0.0, seed=230)
    mu_test = 1.0

    # 依赖图
    A = build_nuisance_dependency_graph(mdl)
    print(f"依赖图邻接矩阵:\n{A}")
    blocks = adjacency_to_blocks(A)
    print(f"连通分量: {blocks}")

    # 离散扫描
    theta_init = discrete_initialization_scan(mdl, data, mu_test)
    print(f"离散扫描最佳初始: {theta_init}")

    # 混合 profiling
    theta_hat, nll_prof, info = profile_theta(mdl, data, mu_test)
    print(f"Profiled θ̂({mu_test}): {theta_hat}")
    print(f"Profiled NLL: {nll_prof:.6f}")
    print(f"优化信息: halley={info['halley_iters']}, fp={info['fp_iters']}")
