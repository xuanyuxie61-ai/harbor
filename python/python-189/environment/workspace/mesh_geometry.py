"""
mesh_geometry.py

状态空间网格划分与三角剖分工具

基于种子项目:
  - 823_obj_to_tri_surface: 3D 网格面片转三角面

科学应用:
  在连续状态空间中, 三角剖分 (Triangulation) 提供了一种局部线性逼近策略的方法:

  1. 状态空间采样得到点云 {s_i}.
  2. Delaunay 三角剖分将凸包分解为单纯形 {Δ_j}.
  3. 在每个单纯形内, 策略近似为重心坐标插值:
         π(s) = Σ_{k=1}^{d+1} λ_k(s) · π(s_{j_k})
         其中 λ_k 为 s 关于顶点 s_{j_k} 的重心坐标.

  物理意义:
      在等离子体约束控制中, 磁面 (flux surface) 的三角剖分
      用于局部 PDE 离散化与控制器插值.
"""

import numpy as np
from math import factorial
from scipy.spatial import Delaunay
from typing import List, Tuple


class StateSpaceTriangulation:
    """
    基于 Delaunay 三角剖分的状态空间划分器.
    """

    def __init__(self, states: np.ndarray):
        """
        参数:
            states: N×d 状态点云
        """
        self.states = np.asarray(states, dtype=float)
        if self.states.ndim == 1:
            self.states = self.states.reshape(-1, 1)
        self.N, self.d = self.states.shape
        if self.N < self.d + 1:
            raise ValueError("StateSpaceTriangulation: not enough points")
        self.tri = Delaunay(self.states)
        self.simplices = self.tri.simplices

    def find_simplex(self, point: np.ndarray) -> int:
        """查询点所在的单纯形索引 (-1 表示在外部)."""
        return int(self.tri.find_simplex(point))

    def barycentric_coordinates(self, point: np.ndarray, simplex_idx: int) -> np.ndarray:
        """
        计算点在单纯形内的重心坐标.

        数学推导:
            设单纯形顶点为 v_1, ..., v_{d+1}.
            解 V · λ = p, 其中 V = [v_2-v_1, ..., v_{d+1}-v_1],
            然后 λ_1 = 1 - Σ_{k=2}^{d+1} λ_k.
        """
        if simplex_idx < 0 or simplex_idx >= len(self.simplices):
            return None
        vertices = self.states[self.simplices[simplex_idx]]
        # 齐次坐标法
        M = np.vstack([vertices.T, np.ones(self.d + 1)])
        rhs = np.append(point, 1.0)
        try:
            lam = np.linalg.solve(M, rhs)
        except np.linalg.LinAlgError:
            lam = np.linalg.lstsq(M, rhs, rcond=None)[0]
        return lam

    def interpolate(self, point: np.ndarray, values: np.ndarray) -> float:
        """
        在点处对顶点值进行重心坐标插值.

        参数:
            point:  d 维查询点
            values: N 维顶点值数组

        返回:
            插值结果
        """
        idx = self.find_simplex(point)
        if idx < 0:
            # 外部: 返回最近邻值
            dists = np.linalg.norm(self.states - point, axis=1)
            return float(values[np.argmin(dists)])
        lam = self.barycentric_coordinates(point, idx)
        if lam is None:
            dists = np.linalg.norm(self.states - point, axis=1)
            return float(values[np.argmin(dists)])
        verts = self.simplices[idx]
        return float(np.dot(lam, values[verts]))

    def simplex_volumes(self) -> np.ndarray:
        """计算所有单纯形的体积."""
        vols = []
        for simp in self.simplices:
            verts = self.states[simp]
            # d 维单纯形体积 = |det(v_2-v_1, ..., v_{d+1}-v_1)| / d!
            M = np.zeros((self.d, self.d))
            for i in range(self.d):
                M[:, i] = verts[i + 1] - verts[0]
            vol = abs(np.linalg.det(M)) / factorial(self.d)
            vols.append(vol)
        return np.array(vols)


def adaptive_mesh_refinement(states: np.ndarray, values: np.ndarray,
                              threshold: float = 0.1, max_points: int = 500) -> np.ndarray:
    """
    自适应网格加密:
    在值函数变化剧烈的区域插入新点.

    策略:
        1. 计算每个单纯形内的值函数梯度估计;
        2. 若梯度 > threshold, 在形心插入新点;
        3. 重复直到点数达到上限.
    """
    points = np.asarray(states, dtype=float).copy()
    vals = np.asarray(values, dtype=float).copy()
    if points.ndim == 1:
        points = points.reshape(-1, 1)

    for _ in range(20):  # 最大加密轮数
        if len(points) >= max_points:
            break
        tri = StateSpaceTriangulation(points)
        new_points = []
        for simp in tri.simplices:
            verts = points[simp]
            vvals = vals[simp]
            # 估计梯度 = 值变化范围 / 顶点间距
            max_diff = np.max(vvals) - np.min(vvals)
            diam = np.max([np.linalg.norm(verts[i] - verts[j])
                           for i in range(len(verts)) for j in range(i + 1, len(verts))])
            if diam < 1.0e-10:
                continue
            grad_est = max_diff / diam
            if grad_est > threshold:
                centroid = np.mean(verts, axis=0)
                new_points.append(centroid)
        if len(new_points) == 0:
            break
        # 去重并合并
        new_points = np.array(new_points)
        # 简单去重
        kept = []
        for p in new_points:
            if len(points) == 0 or np.min(np.linalg.norm(points - p, axis=1)) > 1.0e-6:
                kept.append(p)
                if len(kept) + len(points) >= max_points:
                    break
        if len(kept) == 0:
            break
        kept = np.array(kept)
        points = np.vstack([points, kept])
        # 新点的值用最近邻插值
        for i in range(len(kept)):
            dists = np.linalg.norm(points[:len(points) - len(kept) + i] - kept[i], axis=1)
            vals = np.append(vals, vals[np.argmin(dists)])
    return points
