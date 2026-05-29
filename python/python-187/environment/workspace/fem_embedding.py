#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fem_embedding.py
================

基于种子项目 419_fem3d_sample 的有限元潜空间插值器。

科学背景
--------
用户-物品潜空间可视为一个低维流形 M ⊂ ℝ^d。当评分数据稀疏时，
我们需要在流形上进行函数插值以预测缺失评分。有限元方法（FEM）
将流形离散化为四面体单元，并在每个单元上使用局部基函数进行插值。

对于四面体单元 T = {v1, v2, v3, v4}，体积坐标（barycentric coordinates）:
    φ_i(x) = |det([x, v_j, v_k, v_l])| / |det([v1, v2, v3, v4])|
    
其中 {i,j,k,l} = {1,2,3,4}。

插值公式:
    ũ(x) = Σ_{i=1}^4 u_i φ_i(x)
    
该公式满足:
    - ũ(v_i) = u_i  （插值精确性）
    - Σ φ_i(x) = 1  （单位分解）
    - φ_i(x) ≥ 0    （凸组合，保极值）

对于潜空间中的缺失评分点 x，找到包含它的四面体 T，
计算体积坐标，然后加权平均四个顶点的已知评分。
"""

import numpy as np
from scipy.spatial import Delaunay


class FemEmbeddingInterpolator:
    """
    基于 3D 有限元四面体的潜空间插值器。
    """
    
    def __init__(self, latent_dim=6):
        self.latent_dim = max(3, latent_dim)
    
    def _tetrahedron_volume(self, tetra):
        """
        计算四面体的有向体积。
        
        V = det([[x2-x1, x3-x1, x4-x1],
                 [y2-y1, y3-y1, y4-y1],
                 [z2-z1, z3-z1, z4-z1]]) / 6
                 
        边界保护:
            - 退化四面体（体积接近 0）返回极小正数
        """
        v1, v2, v3, v4 = tetra
        mat = np.array([
            v2 - v1,
            v3 - v1,
            v4 - v1
        ])
        vol = np.linalg.det(mat) / 6.0
        if abs(vol) < 1e-14:
            return 1e-14 if vol >= 0 else -1e-14
        return vol
    
    def _barycentric_coords(self, p, tetra):
        """
        计算点 p 相对于四面体 tetra 的体积坐标。
        
        公式:
            λ_i = V_i / V
            V_i 是将第 i 个顶点替换为 p 后的四面体体积
            
        边界条件:
            - 若 p 恰好在边上或面上，对应 λ_i 可能为 0
            - 若 p 在外部，某些 λ_i < 0
        """
        v0, v1, v2, v3 = tetra
        p = np.asarray(p, dtype=float)
        
        # 构建增广矩阵以使用行列式计算
        def signed_volume(a, b, c, d):
            mat = np.column_stack([b - a, c - a, d - a])
            return np.linalg.det(mat) / 6.0
        
        vol_total = signed_volume(v0, v1, v2, v3)
        if abs(vol_total) < 1e-14:
            return np.array([0.25, 0.25, 0.25, 0.25])
        
        # TODO(Hole 2): 计算体积坐标（barycentric coordinates）
        # 科学背景: 对于四面体 T = {v0, v1, v2, v3}，点 p 的体积坐标为
        #   λ_i = V_i / V_total
        # 其中 V_total 是四面体总体积（signed_volume(v0, v1, v2, v3)）
        # V_i 是将第 i 个顶点替换为 p 后的子四面体有向体积
        # 例如: V_0 = signed_volume(p, v1, v2, v3)
        # 注意: λ_3 可用 1 - λ_0 - λ_1 - λ_2 计算（利用归一化条件）
        lam = np.array([0.25, 0.25, 0.25, 0.25])  # 占位，需要替换为正确实现
        return lam
    
    def _find_containing_tetrahedron(self, p, tri, points):
        """
        找到包含点 p 的四面体。
        
        使用 SciPy Delaunay 的 find_simplex 方法，时间复杂度 O(log N)。
        边界处理:
            - 若 p 在凸包外，返回最近四面体
        """
        p = np.asarray(p, dtype=float)
        simplex = tri.find_simplex(p)
        if simplex < 0:
            # 在凸包外，寻找最近的四面体中心
            centers = tri.transform[:, :3, :3].sum(axis=1) / 3.0 + tri.transform[:, 3, :]
            dists = np.linalg.norm(centers - p, axis=1)
            simplex = np.argmin(dists)
        return simplex
    
    def interpolate(self, R_matrix):
        """
        对评分矩阵进行有限元插值。
        
        策略:
            1. 对每对用户/物品，取潜向量前三维构造 3D 点云
            2. Delaunay 三角剖分得到四面体网格
            3. 对每个缺失评分，在网格中插值
            
        边界保护:
            - 若某行/列全缺失，使用全局均值填充
            - 插值结果截断在 [1, 5]
        """
        R = np.array(R_matrix, dtype=float)
        n_users, n_items = R.shape
        R_interp = np.copy(R)
        
        # 全局均值（用于退化情况）
        global_mean = np.nanmean(R)
        if np.isnan(global_mean):
            global_mean = 3.0
        
        # 构建 3D 嵌入: 使用 SVD 降维到 3 维
        # 先填充 NaN 为行均值
        R_filled = np.copy(R)
        row_means = np.nanmean(R_filled, axis=1)
        for i in range(n_users):
            if not np.isnan(row_means[i]):
                R_filled[i, np.isnan(R_filled[i, :])] = row_means[i]
            else:
                R_filled[i, :] = global_mean
        
        # SVD 降维
        try:
            U, s, Vt = np.linalg.svd(R_filled - global_mean, full_matrices=False)
            coords = U[:, :3] * s[:3]
        except np.linalg.LinAlgError:
            coords = np.random.randn(n_users, 3)
        
        # Delaunay 三角剖分（需要至少 4 个点）
        if n_users < 4:
            # 点数不足，直接返回行均值填充
            for i in range(n_users):
                R_interp[i, np.isnan(R_interp[i, :])] = row_means[i] if not np.isnan(row_means[i]) else global_mean
            return np.clip(R_interp, 1.0, 5.0)
        
        try:
            tri = Delaunay(coords)
        except Exception:
            # 退化情况：所有点共面或共线
            for i in range(n_users):
                R_interp[i, np.isnan(R_interp[i, :])] = row_means[i] if not np.isnan(row_means[i]) else global_mean
            return np.clip(R_interp, 1.0, 5.0)
        
        # 对每个物品，在四面体网格上插值缺失评分
        for j in range(n_items):
            col = R[:, j]
            known = ~np.isnan(col)
            if not np.any(known):
                R_interp[:, j] = global_mean
                continue
            
            # 已知评分的用户坐标和值
            known_coords = coords[known]
            known_vals = col[known]
            
            if known_coords.shape[0] < 4:
                # 已知点太少，使用最近已知值
                for i in range(n_users):
                    if np.isnan(R_interp[i, j]):
                        dists = np.linalg.norm(known_coords - coords[i], axis=1)
                        nearest = np.argmin(dists)
                        R_interp[i, j] = known_vals[nearest]
                continue
            
            # 为已知点构建局部 Delaunay（在 3D 坐标中）
            try:
                local_tri = Delaunay(known_coords)
            except Exception:
                # 退化为最近邻
                for i in range(n_users):
                    if np.isnan(R_interp[i, j]):
                        dists = np.linalg.norm(known_coords - coords[i], axis=1)
                        nearest = np.argmin(dists)
                        R_interp[i, j] = known_vals[nearest]
                continue
            
            # 对缺失点插值
            missing = np.isnan(col)
            for i in np.where(missing)[0]:
                simplex = local_tri.find_simplex(coords[i])
                if simplex >= 0:
                    # 点在四面体内，使用体积坐标插值
                    tetra = known_coords[local_tri.simplices[simplex]]
                    lam = self._barycentric_coords(coords[i], tetra)
                    # 确保非负且归一化
                    lam = np.maximum(lam, 0.0)
                    lam /= lam.sum() + 1e-12
                    val = np.dot(lam, known_vals[local_tri.simplices[simplex]])
                else:
                    # 点在凸包外，使用最近已知点
                    dists = np.linalg.norm(known_coords - coords[i], axis=1)
                    nearest = np.argmin(dists)
                    val = known_vals[nearest]
                R_interp[i, j] = val
        
        # 数值鲁棒性截断
        R_interp = np.clip(R_interp, 1.0, 5.0)
        return R_interp
