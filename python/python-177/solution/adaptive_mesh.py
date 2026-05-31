# -*- coding: utf-8 -*-
"""
adaptive_mesh.py
================
界面附近的自适应网格生成与节点优化模块。

融合原始项目:
  - 757_mesh2d: 2D 非结构网格生成（边界离散化、Delaunay 三角化思想）
  - 238_cvt: Centroidal Voronoi Tessellation（节点优化、能量最小化）

核心数学公式
------------
1. 自适应网格尺寸函数:
   h(x) = h_min + (h_max - h_min) · tanh( |φ(x)| / h_band )
   在界面附近（|φ|<h_band）网格加密，远离界面处粗化。

2. CVT 能量泛函:
   E({z_i}_{i=1}^n) = Σ_i ∫_{V_i} ρ(x) ||x - z_i||² dx
   其中 V_i 为 Voronoi 单元，z_i 为生成点（即网格节点），
   ρ(x) 为密度函数（与界面距离相关）。

3. Lloyd 迭代（CVT 计算）:
   z_i^{k+1} = centroid(V_i^k) = ∫_{V_i} x ρ(x) dx / ∫_{V_i} ρ(x) dx
   迭代直至 ||z^{k+1} - z^k|| < tol。

4. 三角形面积（源自 mesh2d/triarea）:
   A = 0.5 · | (p2-p1) × (p3-p1) |
     = 0.5 · | (x2-x1)(y3-y1) - (y2-y1)(x3-x1) |

5. 网格质量度量:
   Q = 4√3 · A / (L1² + L2² + L3²)
   其中 A 为面积，L_i 为边长。等边三角形 Q=1。
"""

import numpy as np


class AdaptiveMesh:
    """
    基于水平集函数的自适应笛卡尔网格细化，
    并可选使用 CVT 进行节点优化。
    """

    def __init__(self, levelset, h_min=0.01, h_max=0.1, h_band=0.2):
        """
        参数:
            levelset : LevelSetFunction 实例
            h_min    : 界面附近最小网格尺寸
            h_max    : 远处最大网格尺寸
            h_band   : 加密过渡带宽度
        """
        self.ls = levelset
        self.h_min = h_min
        self.h_max = h_max
        self.h_band = h_band

    def compute_size_function(self):
        """
        计算尺寸函数 h(x,y)，基于水平集函数的距离值。
        公式:
            h(x) = h_min + (h_max - h_min) · tanh( |φ(x)| / h_band )
        当 φ≈0 时 h≈h_min；当 |φ|→∞ 时 h→h_max。
        """
        phi = np.abs(self.ls.phi)
        h = self.h_min + (self.h_max - self.h_min) * np.tanh(phi / self.h_band)
        return h

    def refine_grid_uniform(self, factor=2):
        """
        全局均匀细化（演示用，非真正自适应）。
        将网格在每个方向加密 factor 倍。
        """
        nx, ny = self.ls.nx, self.ls.ny
        xlim, ylim = self.ls.xlim, self.ls.ylim
        new_nx = nx * factor
        new_ny = ny * factor

        x_new = np.linspace(xlim[0], xlim[1], new_nx)
        y_new = np.linspace(ylim[0], ylim[1], new_ny)
        X_new, Y_new = np.meshgrid(x_new, y_new, indexing='ij')

        # 双线性插值到新网格
        x_old = self.ls.x
        y_old = self.ls.y
        phi_new = np.zeros((new_nx, new_ny), dtype=np.float64)

        for i in range(new_nx):
            for j in range(new_ny):
                xi = X_new[i, j]
                yi = Y_new[i, j]
                # 找到旧网格中的包围矩形
                ii = np.searchsorted(x_old, xi) - 1
                jj = np.searchsorted(y_old, yi) - 1
                ii = np.clip(ii, 0, nx - 2)
                jj = np.clip(jj, 0, ny - 2)

                tx = (xi - x_old[ii]) / (x_old[ii + 1] - x_old[ii])
                ty = (yi - y_old[jj]) / (y_old[jj + 1] - y_old[jj])

                phi00 = self.ls.phi[ii, jj]
                phi10 = self.ls.phi[ii + 1, jj]
                phi01 = self.ls.phi[ii, jj + 1]
                phi11 = self.ls.phi[ii + 1, jj + 1]

                phi_new[i, j] = (1 - tx) * (1 - ty) * phi00 \
                                + tx * (1 - ty) * phi10 \
                                + (1 - tx) * ty * phi01 \
                                + tx * ty * phi11

        self.ls.nx = new_nx
        self.ls.ny = new_ny
        self.ls.x = x_new
        self.ls.y = y_new
        self.ls.dx = (xlim[1] - xlim[0]) / (new_nx - 1)
        self.ls.dy = (ylim[1] - ylim[0]) / (new_ny - 1)
        self.ls.phi = phi_new
        return self

    def cvt_optimize_nodes_2d(self, num_points=100, max_iter=50, tol=1e-4):
        """
        基于 CVT 的二维节点优化（融入 238_cvt 思想）。
        在计算域内放置 num_points 个生成点，通过 Lloyd 迭代
        使 Voronoi 单元的质心与生成点重合。

        密度函数:
            ρ(x,y) = 1 / (1 + α |φ(x,y)|²)
        使界面附近节点更密集。

        返回优化后的节点坐标。
        """
        xlim, ylim = self.ls.xlim, self.ls.ylim
        # 初始节点：均匀网格 + 小扰动
        nx_pts = int(np.sqrt(num_points))
        ny_pts = int(np.ceil(num_points / nx_pts))
        x_pts = np.linspace(xlim[0] + 0.05, xlim[1] - 0.05, nx_pts)
        y_pts = np.linspace(ylim[0] + 0.05, ylim[1] - 0.05, ny_pts)
        X, Y = np.meshgrid(x_pts, y_pts, indexing='ij')
        points = np.column_stack([X.ravel(), Y.ravel()])[:num_points, :]

        # 添加随机扰动
        points += 0.01 * (np.random.rand(*points.shape) - 0.5)

        # 密度函数参数
        alpha = 50.0

        # 预计算密度场
        phi = self.ls.phi
        x_grid = self.ls.x
        y_grid = self.ls.y
        rho_grid = 1.0 / (1.0 + alpha * phi ** 2)

        # Lloyd 迭代（简化版，在笛卡尔网格上近似）
        for it in range(max_iter):
            points_new = np.zeros_like(points)
            # 对每个生成点，计算其 Voronoi 单元的近似质心
            # 这里使用网格采样近似
            dxg = x_grid[1] - x_grid[0]
            dyg = y_grid[1] - y_grid[0]

            for pidx in range(len(points)):
                px, py = points[pidx]
                # 找到最近的网格区域
                ix = np.argmin(np.abs(x_grid - px))
                iy = np.argmin(np.abs(y_grid - py))
                # 局部窗口
                wx = max(1, int(0.2 / dxg))
                wy = max(1, int(0.2 / dyg))
                i0 = max(0, ix - wx)
                i1 = min(len(x_grid), ix + wx + 1)
                j0 = max(0, iy - wy)
                j1 = min(len(y_grid), iy + wy + 1)

                Xg, Yg = np.meshgrid(x_grid[i0:i1], y_grid[j0:j1], indexing='ij')
                Rg = rho_grid[i0:i1, j0:j1]

                # 计算到当前点的距离，近似 Voronoi 单元
                dists = np.sqrt((Xg - px) ** 2 + (Yg - py) ** 2)
                # 到所有生成点的最小距离
                min_dists = dists.copy()
                for qidx in range(len(points)):
                    if qidx == pidx:
                        continue
                    qx, qy = points[qidx]
                    d2 = np.sqrt((Xg - qx) ** 2 + (Yg - qy) ** 2)
                    min_dists = np.minimum(min_dists, d2)

                mask = dists <= min_dists + 1e-8
                if np.sum(mask) == 0:
                    points_new[pidx] = points[pidx]
                else:
                    wsum = np.sum(Rg[mask])
                    if wsum < 1e-14:
                        points_new[pidx] = points[pidx]
                    else:
                        cx = np.sum(Xg[mask] * Rg[mask]) / wsum
                        cy = np.sum(Yg[mask] * Rg[mask]) / wsum
                        points_new[pidx] = [cx, cy]

            diff = np.max(np.linalg.norm(points_new - points, axis=1))
            points = points_new.copy()
            if diff < tol:
                break

        return points

    @staticmethod
    def triangle_area(p1, p2, p3):
        """
        计算三角形有向面积（融入 mesh2d/triarea 思想）。
        A = 0.5 · ((p2-p1) × (p3-p1))
        参数为 ndarray shape (2,) 或 (N,2)。
        """
        p1 = np.asarray(p1)
        p2 = np.asarray(p2)
        p3 = np.asarray(p3)
        if p1.ndim == 1:
            return 0.5 * ((p2[0] - p1[0]) * (p3[1] - p1[1])
                          - (p2[1] - p1[1]) * (p3[0] - p1[0]))
        else:
            return 0.5 * ((p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1])
                          - (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0]))

    @staticmethod
    def triangle_quality(p1, p2, p3):
        """
        计算三角形质量因子 Q ∈ (0,1]。
        Q = 4√3 · A / (L1² + L2² + L3²)
        Q=1 为等边三角形。
        """
        A = abs(AdaptiveMesh.triangle_area(p1, p2, p3))
        L1 = np.linalg.norm(p2 - p1)
        L2 = np.linalg.norm(p3 - p2)
        L3 = np.linalg.norm(p1 - p3)
        s2 = L1 ** 2 + L2 ** 2 + L3 ** 2
        if s2 < 1e-14:
            return 0.0
        Q = 4.0 * np.sqrt(3.0) * A / s2
        return Q

    def estimate_interface_mesh_quality(self):
        """
        估计界面附近网格的质量（使用提取的零等值线分段近似）。
        将零等值线点连线形成多边形，计算各段质量指标。
        """
        points = self.ls.get_zero_levelset_points()
        if len(points) < 3:
            return 0.0

        # 简单近似：按角度排序后计算相邻三点形成的三角形质量
        cx = np.mean(points[:, 0])
        cy = np.mean(points[:, 1])
        angles = np.arctan2(points[:, 1] - cy, points[:, 0] - cx)
        idx = np.argsort(angles)
        points_sorted = points[idx]

        qualities = []
        n = len(points_sorted)
        for i in range(n):
            p1 = points_sorted[i]
            p2 = points_sorted[(i + 1) % n]
            p3 = points_sorted[(i + 2) % n]
            q = self.triangle_quality(p1, p2, p3)
            qualities.append(q)

        return np.mean(qualities) if qualities else 0.0
