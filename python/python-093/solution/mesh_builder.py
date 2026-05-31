#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mesh_builder.py
水声传播抛物方程模型 — 计算网格生成与域管理

本模块构建抛物方程求解所需的计算网格，来源于：
- 382_fem_to_xml（FEM 网格数据结构：节点、单元、连通性）
- 1265_toms112（点是否在多边形内 — 用于域掩码）

核心数学内容：
1. 深度方向非均匀网格生成（基于声速剖面自适应加密）：
   在声道轴附近和海底边界处加密，开海域处稀疏。
   网格点 z_j 满足映射：
   z_j = z_max · ξ_j^p，其中 p 为拉伸指数（通常 1.5−3.0），
   ξ_j 为 [0,1] 上的均匀节点。

2. 水平步进网格：
   r_m = m · Δr，m = 0,1,...,M。
   Δr 满足稳定性条件：Δr ≤ λ₀/10 = c₀/(10·f)。

3. 节点与单元数据结构：
   - nodes: (N,2) 数组，每行为 (r, z)
   - elements: (E,3) 数组，三角形单元，每行为三个节点索引
   - node_mask: bool 数组，标记有效水域节点（排除陆地）

4. 点是否在多边形内（射线交叉算法，TOMS 112）：
   对水平射线从点 (x0,y0) 向右发射，统计与多边形边界的交叉次数。
   奇数次交叉 → 点在内部。
   算法伪代码：
     inside = False
     for each edge (x_i,y_i) → (x_{i+1},y_{i+1}):
       if (y_i > y0) != (y_{i+1} > y0):
          t = (x_{i+1}−x_i)·(y0−y_i) / (y_{i+1}−y_i) + x_i
          if t > x0: inside = not inside

5. 网格质量指标：
   - 纵横比 AR = L_max / L_min（三角形最大边与最小边之比）
   - 面积变化率：ΔA/A（相邻单元面积相对差异）
"""

import numpy as np


def generate_depth_grid(z_max, nz, stretch_power=2.0, z_axis=None):
    """
    生成深度方向非均匀网格，在 z=0（海面）和 z=z_max（海底）处加密。
    若提供 z_axis，则在声道轴附近额外加密。

    映射公式：
      ξ_j = j/(nz−1),  j=0,...,nz−1
      z_j = z_max · [ξ_j^p / (ξ_j^p + (1−ξ_j)^p)]
    该映射保证两端加密，中间相对均匀。
    """
    j = np.arange(nz, dtype=np.float64)
    xi = j / (nz - 1)
    # 使用双曲正切型映射获得边界层加密
    eps = 1e-10
    xi = np.clip(xi, eps, 1.0 - eps)
    z = z_max * (xi ** stretch_power) / (xi ** stretch_power + (1.0 - xi) ** stretch_power)
    # 若提供声道轴，进行局部扰动加密
    if z_axis is not None and 0 < z_axis < z_max:
        # 在 z_axis 附近添加一个局部压缩因子
        sigma = z_max * 0.15
        compression = 1.0 - 0.3 * np.exp(-0.5 * ((z - z_axis) / sigma) ** 2)
        z = np.cumsum(np.diff(np.concatenate([[0], z])) * compression)
        z = z_max * z / z[-1]
    z[0] = 0.0
    z[-1] = z_max
    return z


def generate_range_grid(r_max, dr):
    """生成均匀水平网格。"""
    n_r = int(np.ceil(r_max / dr)) + 1
    return np.linspace(0.0, r_max, n_r)


def point_in_polygon(x_poly, y_poly, x0, y0):
    """
    判断点 (x0,y0) 是否在简单多边形内部（射线交叉算法，TOMS 112）。
    x_poly, y_poly 为多边形顶点坐标（首尾自动闭合）。
    返回 bool。
    """
    x_poly = np.asarray(x_poly, dtype=np.float64)
    y_poly = np.asarray(y_poly, dtype=np.float64)
    n = len(x_poly)
    if n < 3:
        return False
    inside = False
    x1 = x_poly[-1]
    y1 = y_poly[-1]
    for i in range(n):
        x2 = x_poly[i]
        y2 = y_poly[i]
        # 检查边 (x1,y1) → (x2,y2) 是否与水平射线相交
        if (y1 > y0) != (y2 > y0):
            t = (x2 - x1) * (y0 - y1) / (y2 - y1 + 1e-15) + x1
            if t > x0:
                inside = not inside
        x1, y1 = x2, y2
    return inside


class PEMesh:
    """
    抛物方程计算网格管理器。
    管理 range-depth 平面上的结构化网格，并支持地形自适应裁剪。
    """

    def __init__(self, r_grid, z_grid, env):
        """
        参数:
            r_grid: 水平距离网格 (m)
            z_grid: 深度网格 (m)，向下为正
            env: OceanEnvironment 实例
        """
        self.r_grid = np.asarray(r_grid, dtype=np.float64)
        self.z_grid = np.asarray(z_grid, dtype=np.float64)
        self.env = env
        self.nr = len(r_grid)
        self.nz = len(z_grid)
        self.dr = r_grid[1] - r_grid[0] if self.nr > 1 else 1.0
        self.dz = np.diff(z_grid)
        self.dz = np.concatenate([self.dz, [self.dz[-1]]])  # 补齐末端
        # 构建二维节点坐标 (nr, nz)
        self.R, self.Z = np.meshgrid(self.r_grid, self.z_grid, indexing='ij')
        # 计算每个 range 步的海底深度
        self.seafloor_depth = self.env.bathymetry(self.r_grid)
        # 节点掩码：z <= seafloor_depth(r) 的为有效水域节点
        self.node_mask = np.zeros((self.nr, self.nz), dtype=bool)
        for m in range(self.nr):
            h_b = self.seafloor_depth[m]
            self.node_mask[m, :] = self.z_grid <= h_b + 1e-6
        # 海面层始终有效
        self.node_mask[:, 0] = True
        # 计算有效节点数
        self.num_valid_nodes = np.sum(self.node_mask)
        # 构建三角形单元列表（仅用于后处理/积分，PE 求解使用结构化差分）
        self.elements = self._build_triangular_elements()

    def _build_triangular_elements(self):
        """
        从结构化网格中提取三角形单元（每个矩形划分为 2 个三角形）。
        仅包含全部顶点均在有效域内的单元。
        返回 elements: (E,3) 的节点全局索引数组。
        """
        elements = []
        for m in range(self.nr - 1):
            for n in range(self.nz - 1):
                # 矩形四个顶点的全局索引
                i1 = m * self.nz + n
                i2 = (m + 1) * self.nz + n
                i3 = (m + 1) * self.nz + (n + 1)
                i4 = m * self.nz + (n + 1)
                # 检查掩码
                mask1 = self.node_mask[m, n]
                mask2 = self.node_mask[m + 1, n]
                mask3 = self.node_mask[m + 1, n + 1]
                mask4 = self.node_mask[m, n + 1]
                if mask1 and mask2 and mask4:
                    elements.append([i1, i2, i4])
                if mask2 and mask3 and mask4:
                    elements.append([i2, i3, i4])
        return np.asarray(elements, dtype=np.int64)

    def get_1d_slice(self, m):
        """获取第 m 个 range 步的深度方向一维数据。"""
        return self.z_grid.copy(), self.node_mask[m, :].copy()

    def global_index(self, m, n):
        """将二维索引 (m,n) 映射到全局一维索引。"""
        return m * self.nz + n

    def local_index(self, idx):
        """全局一维索引 → (m,n)。"""
        m = idx // self.nz
        n = idx % self.nz
        return m, n

    def adaptive_range_step(self, m, safety_factor=0.5):
        """
        基于局部 Courant 条件的自适应水平步长：
        Δr_adaptive = safety_factor · min_j [ 2 / (k₀·|n²(z_j)−1|) ]
        防止折射项过大导致数值不稳定。
        """
        z_valid = self.z_grid[self.node_mask[m, :]]
        if len(z_valid) == 0:
            return self.dr
        n2_dev = self.env.refractive_index_squared_deviation(z_valid)
        denom = self.env.k0 * np.abs(n2_dev)
        denom = np.maximum(denom, 1e-12)
        dr_adaptive = safety_factor * 2.0 / np.max(denom)
        return min(dr_adaptive, self.dr * 2.0)

    def mesh_quality_stats(self):
        """计算网格质量统计信息。"""
        if len(self.elements) == 0:
            return {}
        # 纵横比统计（简化：基于深度网格比）
        ar_list = []
        for elem in self.elements:
            nodes = []
            for idx in elem:
                m, n = self.local_index(idx)
                nodes.append((self.R[m, n], self.Z[m, n]))
            nodes = np.asarray(nodes)
            # 三边长
            d = [np.linalg.norm(nodes[i] - nodes[(i + 1) % 3]) for i in range(3)]
            if min(d) > 1e-9:
                ar_list.append(max(d) / min(d))
        return {
            'num_elements': len(self.elements),
            'num_valid_nodes': int(self.num_valid_nodes),
            'aspect_ratio_mean': float(np.mean(ar_list)) if ar_list else 0.0,
            'aspect_ratio_max': float(np.max(ar_list)) if ar_list else 0.0,
        }
