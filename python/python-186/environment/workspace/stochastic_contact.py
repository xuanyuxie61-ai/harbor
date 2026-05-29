"""
stochastic_contact.py
随机接触模型与几何概率模块

基于种子项目:
- 182_circle_positive_distance: 单位圆上几何概率
- 709_magic4_matrix: 4阶幻方矩阵
"""

import numpy as np
from typing import Tuple, List


def circle_positive_distance_monte_carlo(n_samples: int = 10000,
                                         seed: int = 42) -> Tuple[float, float]:
    """
    蒙特卡洛估计单位圆正象限上两点间平均距离。

    采样方法:
        theta = 2*pi * U(0,1)
        x = |cos(theta)|, y = |sin(theta)|

    两点 p,q 的欧氏距离:
        d(p,q) = sqrt((xp-xq)^2 + (yp-yq)^2)

    解析解 (已知):
        E[d] = (4 / pi) * (2 * sqrt(2) - 1) / 3 ≈ 0.958...
    """
    rng = np.random.default_rng(seed)

    # 在正象限采样
    theta = rng.uniform(0.0, 2.0 * np.pi, n_samples)
    x = np.abs(np.cos(theta))
    y = np.abs(np.sin(theta))

    # 两两距离 (抽样估计)
    n_pairs = min(n_samples // 2, 5000)
    indices = rng.choice(n_samples, size=(n_pairs, 2), replace=False)

    dx = x[indices[:, 0]] - x[indices[:, 1]]
    dy = y[indices[:, 0]] - y[indices[:, 1]]
    distances = np.sqrt(dx**2 + dy**2)

    mean_dist = float(np.mean(distances))
    var_dist = float(np.var(distances, ddof=1))

    return mean_dist, var_dist


def geometric_contact_rate(network_size: int,
                           activity_distribution: str = 'uniform',
                           seed: int = 42) -> np.ndarray:
    """
    基于几何概率的随机接触率模型。

    假设个体在二维平面上活动，接触概率随距离衰减:
        P(contact | d) = exp(-d / d_0)

    活动范围建模为圆盘上的均匀分布。

    参数:
        network_size: 网络节点数
        activity_distribution: 活动度分布 ('uniform', 'power_law', 'exponential')

    返回:
        contact_rates: (network_size, network_size) 接触率矩阵
    """
    rng = np.random.default_rng(seed)

    # 节点位置 (单位圆内均匀分布)
    r = np.sqrt(rng.uniform(0.0, 1.0, network_size))
    theta = rng.uniform(0.0, 2.0 * np.pi, network_size)
    pos_x = r * np.cos(theta)
    pos_y = r * np.sin(theta)

    # 活动度
    if activity_distribution == 'uniform':
        activity = np.ones(network_size, dtype=np.float64)
    elif activity_distribution == 'power_law':
        # P(a) ~ a^{-gamma}, gamma = 2.5
        gamma = 2.5
        u = rng.uniform(0.0, 1.0, network_size)
        activity = (1.0 - u)**(-1.0 / (gamma - 1.0))
        activity = np.clip(activity, 0.1, 10.0)
    elif activity_distribution == 'exponential':
        activity = rng.exponential(1.0, network_size)
    else:
        activity = np.ones(network_size, dtype=np.float64)

    # 接触率
    d0 = 0.3  # 特征距离
    contact_rates = np.zeros((network_size, network_size), dtype=np.float64)

    for i in range(network_size):
        for j in range(i + 1, network_size):
            dx = pos_x[i] - pos_x[j]
            dy = pos_y[i] - pos_y[j]
            d = np.sqrt(dx**2 + dy**2)

            # 接触率 = sqrt(a_i * a_j) * exp(-d/d0)
            rate = np.sqrt(activity[i] * activity[j]) * np.exp(-d / d0)
            contact_rates[i, j] = rate
            contact_rates[j, i] = rate

    return contact_rates


def magic4_test_matrix(n: int = 8) -> np.ndarray:
    """
    构造n阶幻方矩阵 (n为4的倍数)，用于测试网络鲁棒性。

    幻方性质:
        每行、每列、两条对角线之和均为 magic_constant = n(n^2+1)/2

    算法 (互补对方法):
        对于单元格 (i,j):
            k = (i-1)*n + j
            m1 = mod(|i-j|, 4)
            m2 = mod(i+j-1, 4)
            若 m1==0 或 m2==0: A[i,j] = n^2 + 1 - k
            否则: A[i,j] = k
    """
    if n % 4 != 0:
        n = (n // 4) * 4
        if n < 4:
            n = 4

    A = np.zeros((n, n), dtype=np.int32)
    for i in range(n):
        for j in range(n):
            k = i * n + j + 1
            m1 = abs(i - j) % 4
            m2 = (i + j + 1) % 4
            if m1 == 0 or m2 == 0:
                A[i, j] = n * n + 1 - k
            else:
                A[i, j] = k

    return A


def percolation_threshold_estimate(network_size: int,
                                   n_realizations: int = 50,
                                   seed: int = 42) -> float:
    """
    通过渗流理论估计网络传播阈值。

    Erdos-Renyi随机图的渗流阈值:
        p_c ≈ 1 / n

    对于具有幂律度分布的网络，阈值更低。

    蒙特卡洛估计:
        随机移除边，找到最大连通分量首次小于 n/2 时的临界概率。
    """
    rng = np.random.default_rng(seed)

    # 构建基础网络
    p_base = 0.1
    adj = np.zeros((network_size, network_size), dtype=np.float64)
    for i in range(network_size):
        for j in range(i + 1, network_size):
            if rng.random() < p_base:
                adj[i, j] = 1.0
                adj[j, i] = 1.0

    thresholds = []

    for _ in range(n_realizations):
        # 逐渐移除边
        edges = []
        for i in range(network_size):
            for j in range(i + 1, network_size):
                if adj[i, j] > 0:
                    edges.append((i, j))

        rng.shuffle(edges)
        n_edges = len(edges)
        adj_temp = adj.copy()

        for step, (i, j) in enumerate(edges):
            adj_temp[i, j] = 0.0
            adj_temp[j, i] = 0.0

            max_comp = largest_component_size(adj_temp)
            if max_comp < network_size / 2.0:
                p_remaining = 1.0 - (step + 1.0) / n_edges
                thresholds.append(p_remaining)
                break

        if len(thresholds) <= _:
            thresholds.append(0.0)

    return float(np.mean(thresholds))


def largest_component_size(adj: np.ndarray) -> int:
    """计算最大连通分量大小"""
    n = adj.shape[0]
    visited = np.zeros(n, dtype=bool)
    max_size = 0

    for start in range(n):
        if not visited[start]:
            size = 0
            queue = [start]
            visited[start] = True
            while queue:
                u = queue.pop(0)
                size += 1
                neighbors = np.where(adj[u, :] > 0)[0]
                for v in neighbors:
                    if not visited[v]:
                        visited[v] = True
                        queue.append(v)
            max_size = max(max_size, size)

    return max_size
