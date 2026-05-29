"""
particle_tracker.py
================================================================================
粒子追踪与最近点搜索模块 —— 基于种子项目 192_closest_point_brute

在 LES 的拉格朗日框架中，需要频繁地将粒子位置映射到欧拉网格上的
最近节点或单元，以插值当地速度、温度等场量。本模块提供暴力最近点
搜索与网格归属判断。

核心物理公式
--------------------------------------------------------------------------------
粒子追踪方程：
    dX_p/dt = u(X_p, t)
    X_p(0) = X_0

在离散时间步上，通过 RK2/RK4 或前向欧拉积分：
    X_p^{n+1} = X_p^n + Δt u(X_p^n, t^n)

对于被动标量粒子，浓度统计满足：
    ⟨C(x)⟩ = (1/V) Σ_p m_p δ(x - X_p) / ρ

其中 m_p 为粒子质量，δ 通过网格插值或箱计数近似。
"""

import numpy as np


def closest_point_brute(points, target):
    """
    暴力搜索目标点的最近邻。

    参数
    ----------
    points : np.ndarray, shape (n, d)
        点集
    target : np.ndarray, shape (d,)
        目标点

    返回
    -------
    idx : int
        最近点索引
    dist : float
        欧氏距离
    """
    points = np.atleast_2d(points)
    target = np.atleast_1d(target)

    diffs = points - target
    dists_sq = np.sum(diffs**2, axis=1)
    idx = int(np.argmin(dists_sq))
    dist = np.sqrt(dists_sq[idx])

    return idx, dist


def find_containing_tetrahedron(p, nodes, element_nodes):
    """
    判断点 p 位于哪个四面体内（通过重心坐标）。

    参数
    ----------
    p : np.ndarray, shape (3,)
    nodes : np.ndarray, shape (n_node, 3)
    element_nodes : np.ndarray, shape (n_elem, 4)

    返回
    -------
    elem_id : int
        包含点的单元索引，-1 表示未找到
    bary : np.ndarray
        重心坐标
    """
    from fem_basis import tetrahedron_volume

    for e in range(element_nodes.shape[0]):
        en = element_nodes[e]
        elem_nodes = nodes[en]

        try:
            vol = tetrahedron_volume(elem_nodes)
        except ValueError:
            continue

        # 计算子四面体体积
        sub_vols = np.zeros(4)
        valid = True
        for i in range(4):
            sub_nodes = np.copy(elem_nodes)
            sub_nodes[i] = p
            try:
                sub_vols[i] = tetrahedron_volume(sub_nodes)
            except ValueError:
                valid = False
                break

        if not valid:
            continue

        bary = sub_vols / vol

        # 数值容差
        if np.all(bary >= -1e-8) and np.all(bary <= 1 + 1e-8):
            return e, np.clip(bary, 0.0, 1.0)

    return -1, None


def track_particles_rk2(particles, velocity_func, dt, n_steps):
    """
    使用二阶 Runge-Kutta 追踪粒子。

    参数
    ----------
    particles : np.ndarray, shape (n, 3)
        初始位置
    velocity_func : callable
        velocity_func(x) → (u, v, w)
    dt : float
        时间步长
    n_steps : int
        步数

    返回
    -------
    trajectories : np.ndarray, shape (n_steps+1, n, 3)
    """
    n_particle = particles.shape[0]
    trajectories = np.zeros((n_steps + 1, n_particle, 3), dtype=np.float64)
    trajectories[0] = particles

    for step in range(n_steps):
        pos = trajectories[step]

        # RK2
        k1 = np.zeros((n_particle, 3))
        for i in range(n_particle):
            k1[i] = velocity_func(pos[i])

        pos_mid = pos + 0.5 * dt * k1

        k2 = np.zeros((n_particle, 3))
        for i in range(n_particle):
            k2[i] = velocity_func(pos_mid[i])

        new_pos = pos + dt * k2

        # 边界处理
        for i in range(n_particle):
            if new_pos[i, 2] < 0:
                new_pos[i, 2] = abs(new_pos[i, 2])

        trajectories[step + 1] = new_pos

    return trajectories
