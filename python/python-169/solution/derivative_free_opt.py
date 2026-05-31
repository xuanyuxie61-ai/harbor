"""
无导数优化模块（PRAXIS主方向法）
================================
基于种子项目:
  - 907_praxis : Brent的PRAXIS算法，基于主方向搜索与SVD更新

核心数学模型:
  1. 目标: 最小化 F(x)，x∈ℝ^n，不计算或近似梯度。

  2. 二次模型:
     Q(x') = F(x) + ½ (x'-x)^T A (x'-x)
     其中 A = V^{-T} D V^{-1}，V 为搜索方向矩阵，D 存储二阶差分估计。

  3. 主方向更新（SVD）:
     定期对方向矩阵 V 做SVD分解:
       V = U Σ W^T
     按奇异值降序排列后，新的搜索方向为 V 的列（即主方向）。
     这消除了病态条件，使算法沿曲率最大的方向搜索。

  4. 一维线搜索（minny）:
     沿方向 d 进行二次插值/估计二阶导数预测最小值:
       d2 ≈ [x2(f1-f0) - x1(f2-f0)] / [x1 x2 (x1-x2)]
     若二阶导数为正，预测最小值位置:
       x_min ≈ -f'(0) / d2

  5. 病态处理:
     illc = true 时，沿当前最优方向进行随机步进以逃离平坦山谷。
"""

import numpy as np
from typing import Callable, Tuple, Optional


def _minfit(n: int, tol: float, A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    r"""
    对对称矩阵 A 进行SVD分解（这里直接使用numpy的SVD，
    但保留EISPACK风格的接口）。
    返回 (d, V)，其中 d 为奇异值，V 为右奇异向量矩阵（列向量）。
    """
    try:
        U, d, Vt = np.linalg.svd(A, full_matrices=False)
    except np.linalg.LinAlgError:
        d = np.ones(n, dtype=float)
        Vt = np.eye(n, dtype=float)
    # 按奇异值降序排列
    idx = np.argsort(-d)
    d = d[idx]
    V = Vt.T[:, idx]
    return d, V


def _line_search(f: Callable, x: np.ndarray, d: np.ndarray,
                 f0: float, h0: float = 1.0, tol: float = 1e-8) -> Tuple[float, np.ndarray]:
    r"""
    沿方向 d 的一维线搜索。
    使用三点二次模型估计最小值。
    """
    d = np.asarray(d, dtype=float)
    d_norm = np.linalg.norm(d)
    if d_norm < 1e-14:
        return f0, x.copy()
    d = d / d_norm
    # 三点采样
    x1 = x + h0 * d
    x2 = x + 2.0 * h0 * d
    f1 = f(x1)
    f2 = f(x2)
    # 二次插值估计最小值
    # 拟合 φ(s) = a s^2 + b s + c，已知 φ(0)=f0, φ(h0)=f1, φ(2h0)=f2
    # a = (f2 - 2f1 + f0) / (2 h0^2)
    # b = (4f1 - f2 - 3f0) / (2 h0)
    # s* = -b / (2a)
    denom = 2.0 * h0 * h0
    a = (f2 - 2.0 * f1 + f0) / denom
    b = (4.0 * f1 - f2 - 3.0 * f0) / (2.0 * h0)
    if a > 1e-14:
        s_star = -b / (2.0 * a)
        s_star = np.clip(s_star, -h0 * 3.0, h0 * 3.0)
    elif b < 0:
        s_star = h0 * 3.0
    else:
        s_star = 0.0
    x_best = x + s_star * d
    f_best = f(x_best)
    if f_best > f0:
        return f0, x.copy()
    return f_best, x_best


class PraxisOptimizer:
    r"""
    PRAXIS主方向无导数优化器。
    """

    def __init__(self, tol: float = 1e-6, max_iter: int = 500,
                 h0: float = 0.1, scbd: float = 10.0):
        self.tol = tol
        self.max_iter = max_iter
        self.h0 = h0
        self.scbd = scbd  # 轴向缩放因子

    def minimize(self, f: Callable, x0: np.ndarray) -> Tuple[np.ndarray, float]:
        r"""
        最小化 f(x)，返回 (x_opt, f_opt)。
        """
        x = np.asarray(x0, dtype=float).copy()
        n = x.size
        fval = float(f(x))
        # 初始搜索方向为单位矩阵的列
        V = np.eye(n, dtype=float)
        D_est = np.ones(n, dtype=float)  # 二阶差分估计

        for iteration in range(self.max_iter):
            x_old = x.copy()
            f_old = fval
            # 沿每个主方向搜索
            for i in range(n):
                d = V[:, i]
                f_new, x_new = _line_search(f, x, d, fval, h0=self.h0, tol=self.tol)
                if f_new < fval:
                    D_est[i] = max(D_est[i] * 0.5, 1e-8)
                    fval = f_new
                    x = x_new
                else:
                    D_est[i] = min(D_est[i] * 2.0, 1e6)
            # 检查是否收敛
            if np.linalg.norm(x - x_old) < self.tol and abs(fval - f_old) < self.tol:
                break
            # 每n次迭代或方向矩阵退化时，更新主方向（SVD）
            if (iteration + 1) % max(n, 3) == 0:
                # 使用当前点与历史点构造方向矩阵
                # 简化：使用对角二阶差分构造近似Hessian的主方向
                H_approx = V @ np.diag(D_est) @ V.T
                d_new, V_new = _minfit(n, self.tol, H_approx)
                # 按奇异值缩放方向
                for i in range(n):
                    if d_new[i] > 1e-14:
                        V[:, i] = V_new[:, i] * (1.0 / np.sqrt(d_new[i]))
                    else:
                        V[:, i] = V_new[:, i]
                # 重新归一化
                for i in range(n):
                    norm = np.linalg.norm(V[:, i])
                    if norm > 1e-14:
                        V[:, i] = V[:, i] / norm
            # 病态检测：随机步进
            if abs(fval - f_old) < self.tol * 0.01:
                perturb = self.h0 * (np.random.default_rng().random(n) - 0.5) * self.scbd
                x_try = x + perturb
                f_try = f(x_try)
                if f_try < fval:
                    fval = f_try
                    x = x_try
            # 边界保护
            if not np.isfinite(fval):
                x = x_old
                fval = f_old
                break
        return x, fval


def trajectory_cost_function(q_nodes_flat: np.ndarray,
                              kin_func,
                              obstacle_checker,
                              q_start: np.ndarray,
                              q_goal: np.ndarray,
                              dt: float = 0.1) -> float:
    r"""
    轨迹优化目标函数（用于PRAXIS无导数优化）。
    将Bézier控制点展平为向量，评估综合代价:
      J = w1·路径长度 + w2·避障惩罚 + w3·末端偏离目标惩罚
          + w4·关节速度惩罚 + w5·奇异位形惩罚
    """
    n_dof = q_start.size
    n_nodes = q_nodes_flat.size // n_dof
    if n_nodes * n_dof != q_nodes_flat.size:
        return 1e10
    q_nodes = q_nodes_flat.reshape(n_nodes, n_dof)
    # 固定起点和终点
    q_nodes[0] = q_start
    q_nodes[-1] = q_goal

    cost = 0.0
    w = [1.0, 500.0, 50.0, 10.0, 100.0]
    # 路径长度
    for i in range(n_nodes - 1):
        cost += w[0] * np.linalg.norm(q_nodes[i + 1] - q_nodes[i])
    # 碰撞惩罚与目标跟踪
    for i in range(n_nodes):
        q = q_nodes[i]
        # 避障惩罚：计算末端位置到障碍物的距离
        try:
            T_ee = kin_func(q)
            p_ee = T_ee[:3, 3]
        except Exception:
            cost += 1e6
            continue
        dist_penalty = obstacle_checker(p_ee)
        if dist_penalty < 0:
            cost += w[1] * abs(dist_penalty)  # 碰撞内部，大惩罚
        elif dist_penalty < 0.2:
            cost += w[2] * (0.2 - dist_penalty)  # 接近障碍物
        # 奇异位形惩罚
        # 这里使用一个简化度量：避免关节角接近边界
        for j in range(n_dof):
            margin = min(abs(q[j] + np.pi), abs(q[j] - np.pi))
            if margin < 0.1:
                cost += w[4] / (margin + 1e-3)
    # 速度平滑性
    for i in range(1, n_nodes - 1):
        ddq = q_nodes[i + 1] - 2 * q_nodes[i] + q_nodes[i - 1]
        cost += w[3] * np.linalg.norm(ddq)
    # 边界保护
    if not np.isfinite(cost):
        cost = 1e10
    return float(cost)
