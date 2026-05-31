#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
geometric_sampler.py
====================

基于种子项目 181_circle_monte_carlo, 182_circle_positive_distance,
1254_tetrahedron_slice_display 的几何采样与蒙特卡洛积分模块。

科学背景
--------
在高维潜空间中，直接计算积分往往不可行。蒙特卡洛方法通过随机采样
提供了一致的估计量。本模块实现了多种几何采样策略：

1. 单位圆/球面采样:
   用户潜向量可视为球面 S^{d-1} 上的点。均匀采样使用:
       x = r · cos(θ),  y = r · sin(θ),  θ ~ U[0, 2π)
   
2. 圆上单项式积分（解析公式）:
   I(e1, e2) = ∮_{S¹} x^{e1} y^{e2} ds
   
   若 e1 或 e2 为奇数: I = 0
   否则:
       I = 2 · Γ((e1+1)/2) · Γ((e2+1)/2) / Γ((e1+e2+2)/2)
       
   其中 Γ(z) 是 Gamma 函数，满足 Γ(n+1) = n! 对整数 n。

3. 正象限圆距离统计:
   用于度量推荐列表的多样性。两个推荐向量的距离分布:
   D = ||p - q||,  p, q ~ U(S¹_+)
   
4. 四面体-平面相交:
   在 3D 潜空间中计算置信区域的截面面积:
   
   平面法形式: n·(x - p₀) = 0
   四面体顶点: t₁, t₂, t₃, t₄
   
   对每个棱边 (t_i, t_j)，若 d_i·d_j < 0（异侧），则交点为:
       x_int = (d_i·t_j - d_j·t_i) / (d_i - d_j)
"""

import numpy as np
from scipy.special import gamma


class GeometricSampler:
    """
    几何蒙特卡洛采样器。
    """
    
    def __init__(self, n_samples=5000):
        self.n_samples = max(n_samples, 10)
        self.rng = np.random.RandomState(42)
    
    def sample_unit_circle(self, n):
        """
        在单位圆 S¹ 上均匀随机采样 n 个点。
        
        参数化:
            x(θ) = [cos(θ), sin(θ)],  θ ~ U[0, 2π)
        """
        n = max(n, 1)
        theta = self.rng.rand(n) * 2.0 * np.pi
        x = np.column_stack([np.cos(theta), np.sin(theta)])
        return x
    
    def sample_positive_circle(self, n):
        """
        在单位圆的正象限（x>0, y>0）上均匀采样。
        
        方法:
            θ ~ U[0, 2π), 然后取绝对值:
            x = |cos(θ)|, y = |sin(θ)|
        """
        n = max(n, 1)
        theta = self.rng.rand(n) * 2.0 * np.pi
        x = np.abs(np.cos(theta))
        y = np.abs(np.sin(theta))
        return np.column_stack([x, y])
    
    def circle_monomial_integral(self, e):
        """
        计算单位圆上单项式 x^{e1} y^{e2} 的精确积分。
        
        公式:
            I = ∮ x^{e1} y^{e2} ds
            
            若 e1 或 e2 为奇数: I = 0（对称性）
            否则:
                I = 2 · Γ((e1+1)/2) · Γ((e2+1)/2) / Γ((e1+e2+2)/2)
                
        边界保护:
            - e 元素为负时返回 0
        """
        e = np.asarray(e, dtype=int)
        if np.any(e < 0):
            return 0.0
        
        if np.any(e % 2 == 1):
            return 0.0
        
        integral = 2.0
        for i in range(2):
            integral *= gamma(0.5 * (e[i] + 1))
        integral /= gamma(0.5 * (e[0] + e[1] + 2))
        
        return float(integral)
    
    def monte_carlo_circle_integral(self, func, n):
        """
        蒙特卡洛估计圆上函数的积分。
        
        估计量:
            Î = (2π/n) Σ_{i=1}^n f(x_i)
            
        其中 2π 是单位圆的周长（弧长元 ds 的积分）。
        """
        n = max(n, 10)
        samples = self.sample_unit_circle(n)
        vals = np.array([func(p[0], p[1]) for p in samples])
        return (2.0 * np.pi / n) * np.sum(vals)
    
    def positive_circle_distance_stats(self, n):
        """
        估计正象限圆上随机点对距离的统计量。
        
        距离:
            D = ||p - q|| = √((x_p - x_q)² + (y_p - y_q)²)
            
        返回均值 μ 和无偏方差 σ²。
        """
        n = max(n, 10)
        p = self.sample_positive_circle(n)
        q = self.sample_positive_circle(n)
        dists = np.linalg.norm(p - q, axis=1)
        
        mu = np.mean(dists)
        if n > 1:
            var = np.sum((dists - mu) ** 2) / (n - 1)
        else:
            var = 0.0
        
        return float(mu), float(var)
    
    def parallelogram_area_3d(self, p):
        """
        计算 3D 平行四边形的面积。
        
        公式:
            A = ||(p3 - p1) × (p2 - p1)||
            
        其中 × 是叉积。
        """
        p = np.asarray(p, dtype=float)
        if p.shape[0] < 3:
            return 0.0
        v1 = p[1] - p[0]
        v2 = p[2] - p[0]
        cross = np.cross(v1, v2)
        return float(np.linalg.norm(cross))
    
    def quadrilateral_area_3d(self, q):
        """
        计算 3D 四边形的面积。
        
        方法:
            通过 Varignon 平行四边形:
            1. 取相邻顶点中点构成平行四边形
            2. 四边形面积 = 2 × 平行四边形面积
            
        边界保护:
            - 少于 3 个点返回 0
            - 3 个点按三角形计算
        """
        q = np.asarray(q, dtype=float)
        m = q.shape[0]
        if m < 3:
            return 0.0
        if m == 3:
            # 三角形: A = 0.5 ||(v1 × v2)||
            v1 = q[1] - q[0]
            v2 = q[2] - q[0]
            return 0.5 * np.linalg.norm(np.cross(v1, v2))
        
        # Varignon 平行四边形
        p = np.zeros((4, 3))
        for i in range(3):
            p[i] = (q[i] + q[i+1]) / 2.0
        p[3] = (q[3] + q[0]) / 2.0
        
        para_area = self.parallelogram_area_3d(p)
        return 2.0 * para_area
    
    def plane_tetrahedron_intersect(self, pp, normal, t):
        """
        计算平面与四面体的交线/交面。
        
        平面法形式:
            n · (x - p₀) = 0
            
        算法:
            1. 计算每个顶点到平面的有符号距离:
               d_i = n·(t_i - p₀) / ||n||
            2. 若所有 d_i 同号，无交
            3. 对每对异侧棱边，计算线性插值交点:
               x = (d_i·t_j - d_j·t_i) / (d_i - d_j)
               
        返回:
            int_num : 交点数量（0,1,2,3,4）
            pint    : 交点坐标数组
        """
        pp = np.asarray(pp, dtype=float).flatten()
        normal = np.asarray(normal, dtype=float).flatten()
        t = np.asarray(t, dtype=float)
        
        if t.shape[0] != 4 or t.shape[1] != 3:
            return 0, np.zeros((4, 3))
        
        dn = np.linalg.norm(normal)
        if dn < 1e-12:
            return 0, np.zeros((4, 3))
        
        n_unit = normal / dn
        
        # 有符号距离
        d = np.array([np.dot(n_unit, t[i] - pp) for i in range(4)])
        
        # 检查是否全同号
        if np.all(d < -1e-12) or np.all(d > 1e-12):
            return 0, np.zeros((4, 3))
        
        pint = np.zeros((4, 3))
        int_num = 0
        
        for j1 in range(4):
            if abs(d[j1]) < 1e-12:
                # 顶点在平面上
                pint[int_num] = t[j1]
                int_num += 1
            else:
                for j2 in range(j1 + 1, 4):
                    # 严格异侧
                    if d[j1] * d[j2] < -1e-12:
                        pint[int_num] = (
                            d[j1] * t[j2] - d[j2] * t[j1]
                        ) / (d[j1] - d[j2])
                        int_num += 1
        
        # 限制为最多 4 个交点
        int_num = min(int_num, 4)
        
        # 若 4 个交点，调整顺序使四边形面积最大
        if int_num == 4:
            area1 = self.quadrilateral_area_3d(pint)
            pint2 = np.copy(pint)
            pint2[2], pint2[3] = pint[3].copy(), pint[2].copy()
            area2 = self.quadrilateral_area_3d(pint2)
            if area2 > area1:
                pint = pint2
        
        return int_num, pint
