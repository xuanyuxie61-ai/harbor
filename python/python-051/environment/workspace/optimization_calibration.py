"""
optimization_calibration.py
===========================
海洋生态系统参数自动标定与最优观测路径规划模块。

融合算法
--------
1. Nelder-Mead 单纯形法（源自 797_nelder_mead）：
   用于标定 NPZD 模型的高维生物参数。
   目标函数为模型输出与"观测"初级生产力之间的 L² 误差：
       J(θ) = ∫_Ω (PP_model(x;θ) - PP_obs(x))² dV
   其中 θ = (V_max, K_N, g_max, K_P, m_P, m_Z, β, γ, r_D)。

   Nelder-Mead 操作（反射、扩张、收缩、压缩）：
       x_r = x̄ + ρ(x̄ - x_worst)
       x_e = x̄ + ρξ(x̄ - x_worst)
       x_c = x̄ + ργ(x̄ - x_worst)   (外部收缩)
       x_c = x̄ - γ(x̄ - x_worst)    (内部收缩)
       x_i = x_best + σ(x_i - x_best)  (整体压缩)
   标准参数：ρ=1, ξ=2, γ=0.5, σ=0.5。

2. TSP 下降启发式（源自 1364_tsp_descent）：
   将自主水下航行器（AUV）的最优采样站部署建模为 TSP，
   通过 transposition 与 reversal 下降搜索最小化总航程。

   目标：min Σ_{k=1}^n d(p_k, p_{k+1})
   其中 p_{n+1} = p_1，d 为站点间欧氏距离。
"""

import numpy as np


# ---------------------------------------------------------------------------
# Nelder-Mead 优化器（源自 797_nelder_mead）
# ---------------------------------------------------------------------------

def nelder_mead_optimize(func, x0, rho=1.0, xi=2.0, gam=0.5, sig=0.5,
                         tol=1e-6, max_feval=500):
    """
    Nelder-Mead 单纯形法最小化标量函数 func(x)。

    参数
    ----
    func : callable
        目标函数 f(x) -> float
    x0 : ndarray (m+1, m)
        初始单纯形，每行为一个顶点
    """
    x = x0.copy().astype(float)
    n_dim = x.shape[1]
    n_vert = x.shape[0]
    if n_vert != n_dim + 1:
        raise ValueError("单纯形顶点数必须为 n_dim + 1")

    def evaluate_simplex(pts):
        return np.array([func(pts[i, :]) for i in range(pts.shape[0])])

    f = evaluate_simplex(x)
    n_feval = n_vert

    # 排序
    idx = np.argsort(f)
    f = f[idx]
    x = x[idx, :]

    converged = False
    diverged = False

    while not converged and not diverged:
        x_bar = np.mean(x[:-1, :], axis=0)  # 除最差外重心

        # 反射
        x_r = (1.0 + rho) * x_bar - rho * x[-1, :]
        f_r = func(x_r)
        n_feval += 1

        if f[0] <= f_r <= f[-2]:
            x[-1, :] = x_r
            f[-1] = f_r
        elif f_r < f[0]:
            # 扩张
            x_e = (1.0 + rho * xi) * x_bar - rho * xi * x[-1, :]
            f_e = func(x_e)
            n_feval += 1
            if f_e < f_r:
                x[-1, :] = x_e
                f[-1] = f_e
            else:
                x[-1, :] = x_r
                f[-1] = f_r
        elif f[-2] <= f_r < f[-1]:
            # 外部收缩
            x_c = (1.0 + rho * gam) * x_bar - rho * gam * x[-1, :]
            f_c = func(x_c)
            n_feval += 1
            if f_c <= f_r:
                x[-1, :] = x_c
                f[-1] = f_c
            else:
                x, f = _shrink_simplex(x, f, func, sig)
                n_feval += n_dim
        else:
            # 内部收缩
            x_c = (1.0 - gam) * x_bar + gam * x[-1, :]
            f_c = func(x_c)
            n_feval += 1
            if f_c < f[-1]:
                x[-1, :] = x_c
                f[-1] = f_c
            else:
                x, f = _shrink_simplex(x, f, func, sig)
                n_feval += n_dim

        idx = np.argsort(f)
        f = f[idx]
        x = x[idx, :]

        converged = (f[-1] - f[0] < tol)
        diverged = (n_feval > max_feval)

    return x[0, :], f[0], n_feval


def _shrink_simplex(x, f, func, sig):
    """
    向最优顶点压缩单纯形。
    """
    n_dim = x.shape[1]
    x_best = x[0, :]
    f[0] = func(x_best)
    for i in range(1, n_dim + 1):
        x[i, :] = sig * x[i, :] + (1.0 - sig) * x_best
        f[i] = func(x[i, :])
    return x, f


# ---------------------------------------------------------------------------
# TSP 下降启发式（源自 1364_tsp_descent）
# ---------------------------------------------------------------------------

def path_cost(n, distance, p):
    """
    计算 TSP 路径 p 的总成本。
    """
    cost = 0.0
    i1 = n - 1
    for i2 in range(n):
        cost += distance[p[i1], p[i2]]
        i1 = i2
    return cost


def tsp_descent(distance, variation_num=2000, seed=None):
    """
    通过 transposition 与 reversal 下降搜索 TSP 较优解。

    参数
    ----
    distance : ndarray (n, n)
        对称距离矩阵
    variation_num : int
        迭代次数
    seed : int or None
        随机种子

    返回
    ----
    best_path : ndarray (n,)
    best_cost : float
    """
    if seed is not None:
        np.random.seed(seed)
    n = distance.shape[0]
    if n < 4:
        raise ValueError("城市数 n >= 4")

    # 验证距离矩阵
    if not np.allclose(distance, distance.T):
        raise ValueError("距离矩阵必须对称")
    if np.any(np.diag(distance) != 0.0):
        raise ValueError("距离矩阵对角线必须为零")

    p = np.random.permutation(n)
    cost = path_cost(n, distance, p)

    for _ in range(variation_num):
        # Transposition：交换两个非邻接城市
        c = np.random.choice(n, 2, replace=False)
        c = np.sort(c)
        i1, i2 = c[0], c[1]
        if i1 + 1 < i2:
            p2 = p.copy()
            p2[i1 + 1:i2 + 1] = np.roll(p2[i1 + 1:i2 + 1], 1)
            p2[i1 + 1] = p[i2]
            cost2 = path_cost(n, distance, p2)
            if cost2 < cost:
                p = p2
                cost = cost2

        # Reversal：反转一段子路径
        c = np.random.choice(n, 2, replace=False)
        c = np.sort(c)
        i1, i2 = c[0], c[1]
        p2 = p.copy()
        p2[i1:i2 + 1] = p2[i1:i2 + 1][::-1]
        cost2 = path_cost(n, distance, p2)
        if cost2 < cost:
            p = p2
            cost = cost2

    return p, cost


def generate_sampling_stations(n_stations, Lx, Lz, depth_min=50.0):
    """
    生成 AUV 采样站位置（水平-垂向二维空间中的点集）。
    """
    x = np.random.uniform(0.0, Lx, n_stations)
    z = np.random.uniform(depth_min, Lz, n_stations)
    return np.column_stack([x, z])


def build_distance_matrix(stations):
    """
    构建采样站之间的欧氏距离矩阵。
    """
    n = stations.shape[0]
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(stations[i] - stations[j])
            dist[i, j] = d
            dist[j, i] = d
    return dist
