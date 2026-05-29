#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
inverse_optimizer.py
大地电磁非线性反演优化器

融合种子项目：
  - 287_dijkstra: 最短路径优先更新策略（用于模型节点的信息传播优先级）
  - 670_levy_dragon_chaos: 迭代函数系统（IFS）混沌扰动策略（用于全局搜索）

核心算法：
  1. Occam 平滑反演
     目标泛函：Φ(m) = Φ_d(m) + λ Φ_m(m)
     其中：
       Φ_d = ||W_d (d_obs - F(m))||²   (数据拟合项)
       Φ_m = ||R (m - m_ref)||²        (模型粗糙度项)

  2. Gauss-Newton 迭代
     m_{k+1} = m_k + (J^T W_d^T W_d J + λ R^T R)^{-1}
               * [J^T W_d^T W_d (d_obs - F(m_k)) - λ R^T R (m_k - m_ref)]

  3. Dijkstra 优先更新
     将模型离散化为图节点，按数据敏感度确定更新优先级。

  4. IFS 混沌扰动
     在多尺度空间中进行随机扰动，避免陷入局部极小。
"""

import numpy as np


def dijkstra_priority_map(n_nodes, adjacency_list, source_nodes, sensitivity_weights):
    """
    使用 Dijkstra 算法思想构建模型更新优先级图

    融合种子项目 287_dijkstra 的核心算法。

    将模型参数空间视为图，节点间的"距离"由敏感度和空间距离共同决定。
    从数据敏感度最高的源节点出发，计算到所有节点的最短"信息传播路径"，
    从而确定参数更新的优先顺序。

    Parameters
    ----------
    n_nodes : int
        节点数
    adjacency_list : list of list of (neighbor, weight)
        邻接表
    source_nodes : list of int
        源节点（高敏感度节点）
    sensitivity_weights : ndarray
        各节点的敏感度权重

    Returns
    -------
    priority : ndarray
        优先级值（越小越优先）
    """
    INF = 1e18
    dist = np.full(n_nodes, INF, dtype=np.float64)
    connected = np.zeros(n_nodes, dtype=bool)

    # 初始化源节点距离为 0，并更新其邻居
    for s in source_nodes:
        dist[s] = 0.0

    # Dijkstra 主循环
    for _ in range(n_nodes):
        # 找到未连接节点中距离最小的
        min_dist = INF
        mv = -1
        for i in range(n_nodes):
            if not connected[i] and dist[i] < min_dist:
                min_dist = dist[i]
                mv = i

        if mv == -1:
            break

        connected[mv] = True

        # 更新邻居距离
        for neighbor, edge_weight in adjacency_list[mv]:
            if not connected[neighbor]:
                # 综合边权重和敏感度
                effective_weight = edge_weight / (sensitivity_weights[neighbor] + 1e-10)
                if dist[mv] + effective_weight < dist[neighbor]:
                    dist[neighbor] = dist[mv] + effective_weight

    return dist


def ifs_chaos_perturbation(x, scale=0.1, n_maps=4):
    """
    迭代函数系统（IFS）混沌扰动

    融合种子项目 670_levy_dragon_chaos 的核心思想，
    设计多尺度仿射变换对模型参数进行扰动。

    定义 n_maps 个随机仿射变换：
        x' = A_i @ x + b_i

    随机选择一个映射应用于当前参数，产生扰动。

    Parameters
    ----------
    x : ndarray
        当前参数向量
    scale : float
        扰动尺度
    n_maps : int
        映射数量

    Returns
    -------
    x_perturbed : ndarray
        扰动后的参数
    """
    x = np.asarray(x, dtype=np.float64)
    dim = len(x)

    # 预定义 Levy 龙风格的收缩映射
    # 每个映射将空间压缩并平移
    i = np.random.randint(0, n_maps)

    # 生成随机收缩矩阵（特征值绝对值 < 1）
    angle = 2.0 * np.pi * i / n_maps
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    scale_factor = 0.5 + 0.3 * np.random.rand()

    A = scale_factor * np.array([[cos_a, -sin_a],
                                  [sin_a, cos_a]])
    if dim > 2:
        # 高维扩展：在随机二维子空间中旋转
        A_full = np.eye(dim) * 0.7
        idx = np.random.choice(dim, min(2, dim), replace=False)
        for k in range(len(idx)):
            for l in range(len(idx)):
                A_full[idx[k], idx[l]] = A[k % 2, l % 2] if k < 2 and l < 2 else A_full[idx[k], idx[l]]
        A = A_full

    b = scale * (np.random.rand(dim) - 0.5)

    x_perturbed = A @ x + b
    return x_perturbed


class OccamInversion:
    """
    Occam 平滑反演器

    最小化目标泛函：
        Φ(m) = Φ_d(m) + λ Φ_m(m)

    其中数据拟合项：
        Φ_d = (d_obs - d_pred)^T C_d^{-1} (d_obs - d_pred)

    模型粗糙度项（一阶差分）：
        Φ_m = (Dm)^T (Dm)

    D 是离散梯度算子。
    """

    def __init__(self, forward_func, n_model, data_errors=None,
                 m_ref=None, lambda_init=1.0, max_iter=30,
                 target_misfit=1.0, lambda_factor=2.0):
        """
        Parameters
        ----------
        forward_func : callable
            正演函数 F(m) -> d_pred
        n_model : int
            模型参数个数
        data_errors : ndarray or None
            数据误差标准差
        m_ref : ndarray or None
            参考模型
        lambda_init : float
            初始正则化参数
        max_iter : int
            最大迭代次数
        target_misfit : float
            目标拟合差（归一化）
        lambda_factor : float
            λ 调整因子
        """
        self.forward_func = forward_func
        self.n_model = n_model
        self.data_errors = data_errors
        self.m_ref = m_ref if m_ref is not None else np.zeros(n_model)
        self.lambda_param = lambda_init
        self.max_iter = max_iter
        self.target_misfit = target_misfit
        self.lambda_factor = lambda_factor
        self.history = []

    def _build_roughness_operator(self, n_model):
        """
        构建一阶差分粗糙度算子 R

        对于层状模型，R 是 (n-1) x n 矩阵：
            R_{i,i} = 1, R_{i,i+1} = -1
        """
        if n_model <= 1:
            return np.zeros((1, n_model))
        R = np.zeros((n_model - 1, n_model), dtype=np.float64)
        for i in range(n_model - 1):
            R[i, i] = 1.0
            R[i, i + 1] = -1.0
        return R

    def _compute_jacobian(self, m, dm=0.01):
        """
        用有限差分计算 Jacobian 矩阵 J = ∂d/∂m

        使用中心差分以提高精度：
            J_{,j} ≈ (F(m + δe_j) - F(m - δe_j)) / (2δ)
        """
        d0 = self.forward_func(m)
        n_data = len(d0)
        J = np.zeros((n_data, self.n_model), dtype=np.float64)

        for j in range(self.n_model):
            m_plus = m.copy()
            m_minus = m.copy()
            # 相对扰动
            delta = dm * max(abs(m[j]), 1.0)
            m_plus[j] += delta
            m_minus[j] -= delta

            d_plus = self.forward_func(m_plus)
            d_minus = self.forward_func(m_minus)
            J[:, j] = (d_plus - d_minus) / (2.0 * delta)

        return J

    def _solve_linear_system(self, J, Wd, R, lambda_param, rhs):
        """
        求解 Gauss-Newton 线性系统

        (J^T Wd^T Wd J + λ R^T R) δm = rhs
        """
        Wd2 = (Wd ** 2)[:, np.newaxis]
        lhs = J.T @ (Wd2 * J) + lambda_param * (R.T @ R)
        # 添加 Levenberg-Marquardt 阻尼项以提高稳定性
        lhs += 1e-6 * np.eye(self.n_model)

        try:
            delta_m = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            delta_m = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
        return delta_m

    def invert(self, d_obs, m_initial):
        """
        执行 Occam 反演

        Parameters
        ----------
        d_obs : ndarray
            观测数据
        m_initial : ndarray
            初始模型参数

        Returns
        -------
        m_best : ndarray
            最优模型
        lambda_best : float
            最优正则化参数
        """
        d_obs = np.asarray(d_obs, dtype=np.float64)
        m = np.asarray(m_initial, dtype=np.float64)
        n_data = len(d_obs)

        if self.data_errors is None:
            Wd = np.ones(n_data)
        else:
            Wd = 1.0 / np.asarray(self.data_errors)

        R = self._build_roughness_operator(self.n_model)

        m_best = m.copy()
        best_misfit = np.inf
        lambda_best = self.lambda_param

        # TODO [Hole 3]: 实现 Occam 平滑反演的核心迭代算法
        # 需要完成：
        # 1. 迭代循环：计算正演预测 d_pred = forward_func(m)
        # 2. 计算数据拟合差 misfit = ||W_d * (d_obs - d_pred)||^2
        # 3. 记录迭代历史（iter, misfit, lambda, model）
        # 4. 更新最优模型和最优正则化参数
        # 5. 检查收敛条件（misfit <= target_misfit）
        # 6. 用有限差分计算 Jacobian 矩阵 J = partial d / partial m
        # 7. 构建 Gauss-Newton 右端项:
        #    rhs = J^T W_d^2 (d_obs - d_pred) - lambda * R^T R (m - m_ref)
        # 8. 求解线性系统得到模型更新量 delta_m
        # 9. 线搜索确定步长 alpha，保证物理约束（m_new >= 0.1）
        # 10. 根据 misfit 变化动态调整正则化参数 lambda
        # 11. 每5次迭代引入 IFS 混沌扰动以避免局部极小
        # 关键耦合：本方法的 misfit 计算必须与 forward_func 的输出格式兼容
        # 关键科学知识：Gauss-Newton + Tikhonov 正则化的非线性反演理论
        raise NotImplementedError("Hole 3: Occam 反演核心迭代算法待实现")


class MultiObjectiveOptimizer:
    """
    多目标优化器（数据拟合 + 模型粗糙度 + 边界距离）

    使用 Dijkstra 优先级指导参数更新顺序。
    """

    def __init__(self, forward_func, n_model, n_data,
                 dijkstra_sources=None, adjacency=None):
        self.forward_func = forward_func
        self.n_model = n_model
        self.n_data = n_data
        self.dijkstra_sources = dijkstra_sources or [0]
        self.adjacency = adjacency

    def compute_priorities(self, sensitivity):
        """计算模型参数更新优先级"""
        if self.adjacency is None:
            # 默认链式邻接
            adj = [[] for _ in range(self.n_model)]
            for i in range(self.n_model - 1):
                adj[i].append((i + 1, 1.0))
                adj[i + 1].append((i, 1.0))
            self.adjacency = adj

        priorities = dijkstra_priority_map(
            self.n_model, self.adjacency,
            self.dijkstra_sources, sensitivity
        )
        return priorities

    def weighted_update(self, d_obs, m_current, sensitivity, learning_rate=0.1):
        """
        基于优先级的加权参数更新

        敏感度高的节点优先更新，更新幅度与优先级成反比。
        """
        priorities = self.compute_priorities(sensitivity)
        # 优先级归一化到 [0, 1]
        p_max = np.max(priorities)
        if p_max > 0:
            weights = 1.0 - priorities / p_max
        else:
            weights = np.ones(self.n_model)

        d_pred = self.forward_func(m_current)
        residual = d_obs - d_pred

        # 简单的梯度下降（用于演示）
        # 实际应用中应使用完整的 Jacobian
        J = np.zeros((self.n_data, self.n_model))
        dm = 0.01
        for j in range(self.n_model):
            m_plus = m_current.copy()
            m_plus[j] += dm
            d_plus = self.forward_func(m_plus)
            J[:, j] = (d_plus - d_pred) / dm

        gradient = -2.0 * J.T @ residual
        gradient = gradient * weights

        m_new = m_current - learning_rate * gradient
        m_new = np.maximum(m_new, 0.1)
        return m_new


if __name__ == "__main__":
    # 自检：简单线性反演
    def linear_forward(m):
        return 2.0 * m + 1.0

    inv = OccamInversion(linear_forward, n_model=3, max_iter=20)
    d_obs = np.array([5.0, 7.0, 9.0])
    m_init = np.ones(3)
    m_best, lam = inv.invert(d_obs, m_init)
    print(f"Occam 反演结果: m = {m_best}, λ = {lam:.4f}")
    print(f"预测: {linear_forward(m_best)}")

    # IFS 扰动测试
    x = np.array([1.0, 2.0, 3.0])
    for _ in range(3):
        x = ifs_chaos_perturbation(x, scale=0.5)
        print(f"IFS 扰动后: {x}")

    # Dijkstra 优先级测试
    adj = [[(1, 1.0), (2, 2.0)], [(0, 1.0), (2, 1.0), (3, 1.0)],
           [(0, 2.0), (1, 1.0), (3, 1.0)], [(1, 1.0), (2, 1.0)]]
    sens = np.array([1.0, 0.5, 0.8, 0.3])
    prio = dijkstra_priority_map(4, adj, [0], sens)
    print(f"Dijkstra 优先级: {prio}")
