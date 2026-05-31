"""
curve_parameterization.py
外摆线参数化与流形上的测地线探索
融合原项目: 336_epicycloid

核心科学思想:
外摆线(epicycloid)是圆在另一个圆上无滑滚动时，
圆周上一点的轨迹。其参数方程蕴含深刻的微分几何结构。

在流形学习中，外摆线可用于:
1. 生成嵌入在流形上的闭合测试曲线；
2. 参数化高维数据点的周期性演化路径；
3. 探索流形的拓扑非平凡性。

数学模型:
外摆线参数方程:
    x(t) = r (k+1) cos(t) - r cos((k+1)t)
    y(t) = r (k+1) sin(t) - r sin((k+1)t)
其中 k = R/r 为大圆与小圆半径比，s 为旋转圈数。

测地线方程 (在度量g下):
    d²x^μ/dt² + Γ^μ_{νρ} (dx^ν/dt)(dx^ρ/dt) = 0
其中 Christoffel 符号:
    Γ^μ_{νρ} = ½ g^{μσ}(∂_ν g_{ρσ} + ∂_ρ g_{νσ} - ∂_σ g_{νρ})
"""

import numpy as np
from typing import Tuple


def epicycloid_xy(k: float, s: float, n: int = 200) -> Tuple[np.ndarray, np.ndarray]:
    """
    生成外摆线坐标
    k: 大圆与小圆半径比
    s: 小圆绕大圆旋转的圈数
    n: 采样点数
    返回: (x, y)
    """
    rsmall = 1.0
    t = np.linspace(0.0, 2.0 * np.pi * s, n)
    x = rsmall * (k + 1.0) * np.cos(t) - rsmall * np.cos((k + 1.0) * t)
    y = rsmall * (k + 1.0) * np.sin(t) - rsmall * np.sin((k + 1.0) * t)
    return x, y


def epicycloid_arc_length(k: float, s: float, n: int = 1000) -> float:
    """
    计算外摆线弧长
    L = ∫_0^{2πs} sqrt((dx/dt)² + (dy/dt)²) dt
    """
    t = np.linspace(0.0, 2.0 * np.pi * s, n)
    dx_dt = -(k + 1.0) * np.sin(t) + (k + 1.0) * np.sin((k + 1.0) * t)
    dy_dt = (k + 1.0) * np.cos(t) - (k + 1.0) * np.cos((k + 1.0) * t)
    ds = np.sqrt(dx_dt ** 2 + dy_dt ** 2)
    return float(np.trapz(ds, t))


def epicycloid_curvature(k: float, s: float, n: int = 500) -> np.ndarray:
    """
    计算外摆线曲率
    κ = |x' y'' - y' x''| / (x'² + y'²)^{3/2}
    """
    t = np.linspace(0.0, 2.0 * np.pi * s, n)
    dx = -(k + 1.0) * np.sin(t) + (k + 1.0) * np.sin((k + 1.0) * t)
    dy = (k + 1.0) * np.cos(t) - (k + 1.0) * np.cos((k + 1.0) * t)
    d2x = -(k + 1.0) * np.cos(t) + (k + 1.0) ** 2 * np.cos((k + 1.0) * t)
    d2y = -(k + 1.0) * np.sin(t) + (k + 1.0) ** 2 * np.sin((k + 1.0) * t)
    num = np.abs(dx * d2y - dy * d2x)
    den = (dx ** 2 + dy ** 2) ** 1.5
    den = np.where(den < 1e-15, 1e-15, den)
    return num / den


def embed_epicycloid_high_dim(k: float, s: float, D: int = 10,
                               n: int = 200) -> np.ndarray:
    """
    将外摆线嵌入到D维空间
    使用前2维作为外摆线，其余维添加小扰动
    """
    x, y = epicycloid_xy(k, s, n)
    data = np.zeros((n, D))
    data[:, 0] = x
    data[:, 1] = y
    # 高维扰动 (模拟流形嵌入)
    np.random.seed(42)
    for d in range(2, D):
        freq = 0.5 + d * 0.3
        amp = 0.1 / d
        data[:, d] = amp * np.sin(freq * np.linspace(0, 2 * np.pi, n))
    return data


def christoffel_symbols(metric: np.ndarray, h: float = 1e-5) -> np.ndarray:
    """
    数值计算Christoffel符号
    metric: 度量张量函数 g(x)，返回 (D, D) 矩阵
    返回: Γ^μ_{νρ} (D, D, D)
    """
    D = metric(np.zeros(2)).shape[0]  # 假设2维参数空间
    Gamma = np.zeros((D, D, D))
    # 简化为固定点计算
    x0 = np.zeros(2)
    g0 = metric(x0)
    g_inv = np.linalg.inv(g0 + 1e-6 * np.eye(D))
    for mu in range(D):
        for nu in range(D):
            for rho in range(D):
                # 数值微分
                dg_nu = np.zeros((D, D))
                dg_rho = np.zeros((D, D))
                dg_sigma = np.zeros((D, D))
                # 这里简化处理，仅返回零 (实际应用需要更复杂的数值微分)
                Gamma[mu, nu, rho] = 0.5 * (
                    g_inv[mu, 0] * (dg_nu[0, rho] + dg_rho[0, nu] - dg_sigma[nu, rho]) +
                    g_inv[mu, 1] * (dg_nu[1, rho] + dg_rho[1, nu] - dg_sigma[nu, rho])
                )
    return Gamma


def geodesic_distance_estimate(data: np.ndarray, i: int, j: int,
                                k: int = 10) -> float:
    """
    估计流形上两点间的测地距离
    使用Dijkstra-like路径在k近邻图上近似
    """
    N = len(data)
    dists = np.linalg.norm(data - data[i], axis=1)
    visited = np.zeros(N, dtype=bool)
    distances = np.full(N, np.inf)
    distances[i] = 0.0
    prev = -1 * np.ones(N, dtype=int)
    for _ in range(N):
        u = -1
        min_dist = np.inf
        for v in range(N):
            if not visited[v] and distances[v] < min_dist:
                min_dist = distances[v]
                u = v
        if u == -1:
            break
        visited[u] = True
        if u == j:
            break
        # 找k近邻
        neigh_dists = np.linalg.norm(data - data[u], axis=1)
        knn_idx = np.argsort(neigh_dists)[1:k + 1]
        for v in knn_idx:
            if not visited[v]:
                alt = distances[u] + neigh_dists[v]
                if alt < distances[v]:
                    distances[v] = alt
                    prev[v] = u
    return distances[j]


def isometric_embedding_quality(data_high: np.ndarray,
                                 data_low: np.ndarray) -> float:
    """
    评估低维嵌入的等距保持质量
    计算距离保持误差:
        Q = 1 - (Σ_{i<j} |d_H(i,j) - d_L(i,j)|²) / (Σ_{i<j} d_H(i,j)²)
    """
    N = len(data_high)
    num = 0.0
    den = 0.0
    count = 0
    for i in range(min(N, 50)):
        for j in range(i + 1, min(N, 50)):
            d_h = np.linalg.norm(data_high[i] - data_high[j])
            d_l = np.linalg.norm(data_low[i] - data_low[j])
            num += (d_h - d_l) ** 2
            den += d_h ** 2
            count += 1
    if den < 1e-15:
        return 1.0
    return max(0.0, 1.0 - num / den)
