"""
stability_optimizer.py
多足机器人静态/动态稳定性优化与约束处理模块。
融入种子项目：
  - 845_pagerank2（稀疏邻接矩阵 → 支撑图的中心性分析）
  - 1358_trinity（三角网格区域的线性规划约束 → 支撑多边形稳定性 LP）

科学背景：
多足机器人稳定性判据：
1. 静态稳定性：投影重心（COM）必须落在支撑多边形（support polygon）内部。
2. 动态稳定性：零力矩点（ZMP）必须落在支撑多边形内部。

支撑多边形定义为所有支撑足的凸包。对 n 个支撑足位置 p_i ∈ R^2，
凸包可通过线性规划或 Graham 扫描求得。

ZMP 计算公式（简化平面模型）：
    x_zmp = (m·g·x_com + Σ τ_y) / (m·g + Σ f_z)
    y_zmp = (m·g·y_com - Σ τ_x) / (m·g + Σ f_z)
其中 τ 为惯性力矩，f_z 为垂直接触力。
"""

import numpy as np
from typing import Tuple, List, Optional
from utils import clip_to_bounds


class SupportPolygon:
    """
    支撑多边形计算与包含性测试。
    源自 trinity 项目的三角网格覆盖思想，映射为二维凸包问题。
    """

    def __init__(self, foot_positions: np.ndarray):
        """
        foot_positions: (n, 2) 或 (n, 3) 的支撑足端位置（仅取 xy）
        """
        pts = np.asarray(foot_positions, dtype=float)
        if pts.ndim == 1:
            pts = pts.reshape(-1, 2)
        elif pts.shape[1] >= 3:
            pts = pts[:, :2]
        self.points = pts
        self.hull = self._convex_hull_graham(pts)

    def _convex_hull_graham(self, pts: np.ndarray) -> np.ndarray:
        """
        Graham 扫描法求凸包，O(n log n)。

        算法步骤：
        1. 选取 y 坐标最小的点作为极点 P0。
        2. 对其余点按相对于 P0 的极角排序。
        3. 用栈维护凸包顶点，对每个新点检查是否形成左转（叉积 > 0）。
        """
        if len(pts) <= 1:
            return pts.copy()
        # 找最低点
        start_idx = np.argmin(pts[:, 1])
        p0 = pts[start_idx]
        # 按极角排序
        others = np.delete(pts, start_idx, axis=0)
        angles = np.arctan2(others[:, 1] - p0[1], others[:, 0] - p0[0])
        order = np.argsort(angles)
        sorted_pts = others[order]

        hull = [p0, sorted_pts[0]]
        for i in range(1, len(sorted_pts)):
            while len(hull) > 1:
                cross = self._cross(hull[-2], hull[-1], sorted_pts[i])
                if cross <= 1e-12:
                    hull.pop()
                else:
                    break
            hull.append(sorted_pts[i])
        return np.array(hull)

    @staticmethod
    def _cross(o: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
        """二维叉积 (OA × OB)_z。"""
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    def contains_point(self, p: np.ndarray) -> bool:
        """
        判断点 p 是否在凸包内部（含边界）。
        方法：将凸包分解为以第一个顶点为公共顶点的三角形扇，
        检查 p 是否在某三角形内且所有叉积同号。
        """
        p = np.asarray(p, dtype=float)[:2]
        hull = self.hull
        n = len(hull)
        if n == 0:
            return False
        if n == 1:
            return np.allclose(p, hull[0])
        if n == 2:
            # 在线段上
            cross = self._cross(hull[0], hull[1], p)
            if abs(cross) > 1e-9:
                return False
            dot = np.dot(p - hull[0], p - hull[1])
            return dot <= 1e-9

        sign = None
        for i in range(n):
            a = hull[i]
            b = hull[(i + 1) % n]
            c = self._cross(a, b, p)
            if abs(c) < 1e-9:
                continue
            curr_sign = c > 0
            if sign is None:
                sign = curr_sign
            elif sign != curr_sign:
                return False
        return True

    def distance_to_boundary(self, p: np.ndarray) -> float:
        """
        计算点 p 到支撑多边形边界的带符号距离。
        正值表示在内部，负值表示在外部。
        """
        p = np.asarray(p, dtype=float)[:2]
        hull = self.hull
        n = len(hull)
        if n < 3:
            return -np.inf
        min_dist = float('inf')
        for i in range(n):
            a = hull[i]
            b = hull[(i + 1) % n]
            # 点到线段距离
            ab = b - a
            t = np.dot(p - a, ab) / (np.dot(ab, ab) + 1e-14)
            t = clip_to_bounds(np.array([t]), np.array([0.0]), np.array([1.0]))[0]
            closest = a + t * ab
            dist = np.linalg.norm(p - closest)
            # 符号：叉积判断左右
            cross = self._cross(a, b, p)
            if cross < 0:
                dist = -dist
            min_dist = min(min_dist, dist)
        return min_dist


class StabilityMargin:
    """
    静态与动态稳定性裕度计算。
    """

    def __init__(self, robot_mass: float = 5.0, gravity: float = 9.81):
        self.m = robot_mass
        self.g = gravity

    def static_margin(self, com: np.ndarray, support: SupportPolygon) -> float:
        """
        静态稳定性裕度：COM 到支撑多边形边界的最短距离。
        单位：米。正值表示稳定。
        """
        return support.distance_to_boundary(com)

    def zmp_position(self, com: np.ndarray, com_acceleration: np.ndarray,
                     angular_momentum_rate: np.ndarray,
                     foot_forces: np.ndarray, foot_positions: np.ndarray) -> np.ndarray:
        """
        计算零力矩点（ZMP）位置。

        简化公式（平面地面 z=0）：
            x_zmp = x_com - (z_com · a_x) / g
            y_zmp = y_com - (z_com · a_y) / g
        更一般形式：
            p_zmp = p_com - ( (p_com - p_i) × f_i + τ_i )_z / (f_i)_z
        这里采用简化版本以保证数值鲁棒性。
        """
        com = np.asarray(com, dtype=float)
        a = np.asarray(com_acceleration, dtype=float)
        z_com = com[2] if len(com) > 2 else 0.3
        x_zmp = com[0] - (z_com * a[0]) / self.g
        y_zmp = com[1] - (z_com * a[1]) / self.g
        return np.array([x_zmp, y_zmp])


class SupportGraphCentrality:
    """
    源自 pagerank2 的稀疏图分析思想。

    将支撑状态建模为有向图 G = (V, E)，其中：
    - 顶点 V：机器人的每条腿
    - 边 E(i,j)：腿 i 的支撑/摆动相位转移对腿 j 的影响权重

    采用 PageRank 式特征向量中心性评估每条腿在整体稳定性中的重要性：
        r = α·M·r + (1-α)·v
    其中 M 为列随机转移矩阵，α 为阻尼因子（通常 0.85），
    v 为个性化向量（可设均匀分布）。
    """

    def __init__(self, n_legs: int = 6, alpha: float = 0.85):
        self.n = n_legs
        self.alpha = alpha

    def build_transition_matrix(self, stance_state: np.ndarray,
                                coupling_matrix: np.ndarray) -> np.ndarray:
        """
        根据当前支撑状态与腿间耦合矩阵构建转移矩阵 M。
        stance_state: (n,) 0/1 向量，1 表示支撑
        coupling_matrix: (n, n) 非负权重矩阵
        """
        M = np.zeros((self.n, self.n))
        for j in range(self.n):
            col_sum = 0.0
            for i in range(self.n):
                if i == j:
                    # 自环：支撑腿保持支撑的趋势更强
                    w = 2.0 if stance_state[i] else 0.5
                else:
                    w = coupling_matrix[i, j]
                M[i, j] = w
                col_sum += w
            if col_sum > 1e-12:
                M[:, j] /= col_sum
            else:
                M[:, j] = 1.0 / self.n
        return M

    def pagerank(self, M: np.ndarray, tol: float = 1e-8, max_iter: int = 100) -> np.ndarray:
        """
        幂迭代求解 PageRank 向量。
        """
        v = np.ones(self.n) / self.n
        r = v.copy()
        for _ in range(max_iter):
            r_new = self.alpha * M @ r + (1.0 - self.alpha) * v
            if np.linalg.norm(r_new - r, 1) < tol:
                break
            r = r_new
        return r


class LinearStabilityConstraint:
    """
    源自 trinity 的线性规划约束思想。

    将支撑多边形包含性约束线性化为 LP 不等式：
    对凸包顶点按逆时针排列的边 (v_i, v_{i+1})，内部点 p 满足
        n_i · p ≤ n_i · v_i    （或 ≥，取决于法向方向）
    其中 n_i 为边的外法向量。

    在步态优化中，这些不等式作为硬约束加入二次规划（QP）：
        min_τ   τ^T·W·τ + c^T·τ
        s.t.    A·τ ≤ b
                τ_min ≤ τ ≤ τ_max
    """

    def __init__(self):
        pass

    def polygon_constraints(self, hull: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        将凸多边形转化为半空间表示 A·p ≤ b。
        返回 (A, b)，A 为 (m, 2)，b 为 (m,)。
        """
        n = len(hull)
        A = np.zeros((n, 2))
        b = np.zeros(n)
        for i in range(n):
            v_i = hull[i]
            v_j = hull[(i + 1) % n]
            edge = v_j - v_i
            # 外法向量（逆时针凸包的外法指向右侧）
            normal = np.array([edge[1], -edge[0]])
            norm_len = np.linalg.norm(normal)
            if norm_len > 1e-12:
                normal /= norm_len
            A[i] = normal
            b[i] = np.dot(normal, v_i)
        return A, b

    def com_feasible_region(self, support: SupportPolygon,
                            margin: float = 0.01) -> Tuple[np.ndarray, np.ndarray]:
        """
        生成带安全裕度的 COM 可行域约束。
        将每条边的半空间向内收缩 margin 距离。
        """
        A, b = self.polygon_constraints(support.hull)
        b_shrunk = b - margin
        return A, b_shrunk
