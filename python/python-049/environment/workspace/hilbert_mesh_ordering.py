#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
 hilbert_mesh_ordering.py
 
 融合种子项目：
   - 536_hilbert_curve_3d：3D Hilbert 曲线空间填充
 
 科学功能：
   Hilbert 曲线空间排序优化。
   
   在海啸模拟的大规模网格计算中，内存访问局部性对性能至关重要。
   Hilbert 曲线是一种空间填充曲线，能够保持二维/三维空间中
   相邻点的局部性。本模块对计算网格进行 Hilbert 曲线排序，
   优化缓存利用率。
 
 核心数学公式：
 
   1) Hilbert 曲线定义：
      Hilbert 曲线是一种分形空间填充曲线，通过递归构造：
      - 阶 1：一个 U 形曲线覆盖 2×2 网格
      - 阶 n：将四个阶 n-1 的曲线通过旋转和反射组合，
              覆盖 2^n × 2^n 网格
   
   2) Hilbert 曲线坐标转换（3D 版改造为 2D 版）：
      给定一维坐标 h ∈ [0, 2^{2n}-1]，计算二维坐标 (x, y)。
      
      递归算法：
      - 提取 h 的最低两位：o = h mod 4
      - 根据 o 确定初始坐标
      - 右移 h >>= 2，递归处理更高阶
      - 每阶通过旋转和反射变换坐标
      
   3) 局部性度量：
      Hilbert 排序的局部性指数定义为相邻点之间的平均欧氏距离：
      L = (1/N) Σ_{i=0}^{N-1} ||p_{i+1} - p_i||
      
      对于相同点集，Hilbert 排序的 L 值通常远小于行优先排序。
   
   4) 海啸模拟中的应用：
      将二维/三维网格点按 Hilbert 曲线顺序排列，
      使得空间相邻的计算单元在内存中也相邻，
      从而提升有限差分计算的缓存命中率。
"""

import numpy as np


class HilbertMeshOrderer:
    """
    Hilbert 曲线网格排序器。
    """
    
    def __init__(self, order=6):
        """
        Parameters
        ----------
        order : int
            Hilbert 曲线阶数，覆盖 2^order × 2^order 网格
        """
        self.order = order
        self.N = 2 ** order
        
        if order < 1:
            raise ValueError("阶数必须 ≥ 1")
    
    def h_to_xy(self, h):
        """
        将一维 Hilbert 坐标 h 转换为二维坐标 (x, y)。
        
        来源于 h_to_xyz.m 的 2D 改造版本。
        
        Parameters
        ----------
        h : int
            Hilbert 坐标
            
        Returns
        -------
        x, y : int
            二维网格坐标
        """
        # 初始坐标由最低两位决定
        o = h & 3
        
        if o == 0:
            x, y = 0, 0
        elif o == 1:
            x, y = 1, 0
        elif o == 2:
            x, y = 1, 1
        else:  # o == 3
            x, y = 0, 1
        
        h >>= 2
        w = 2
        
        while h > 0:
            o = h & 3
            xold, yold = x, y
            
            if o == 0:
                x = yold
                y = xold
            elif o == 1:
                x = xold + w
                y = yold
            elif o == 2:
                x = xold + w
                y = yold + w
            else:  # o == 3
                x = w - 1 - yold
                y = w - 1 - xold
            
            h >>= 2
            w *= 2
        
        return x, y
    
    def xy_to_h(self, x, y):
        """
        将二维坐标 (x, y) 转换为一维 Hilbert 坐标 h。
        
        来源于 xyz_to_h.m 的 2D 改造版本。
        """
        h = 0
        s = self.N // 2
        
        # 旋转和反射状态
        rx, ry = 0, 0
        
        while s > 0:
            rx = 1 if (x & s) > 0 else 0
            ry = 1 if (y & s) > 0 else 0
            
            h += s * s * ((3 * rx) ^ ry)
            
            # 旋转和反射
            if ry == 0:
                if rx == 1:
                    x = self.N - 1 - x
                    y = self.N - 1 - y
                x, y = y, x
            
            s >>= 1
        
        return h
    
    def order_2d_grid(self, nx, ny):
        """
        对二维网格进行 Hilbert 排序。
        
        Parameters
        ----------
        nx, ny : int
            网格尺寸
            
        Returns
        -------
        ordered_indices : ndarray
            Hilbert 排序后的 (j, i) 索引列表，形状 (N, 2)
        """
        # 确定覆盖网格所需的最小 Hilbert 阶数
        max_dim = max(nx, ny)
        needed_order = int(np.ceil(np.log2(max_dim)))
        needed_order = max(needed_order, 1)
        
        N_hilbert = 2 ** needed_order
        
        # 生成 Hilbert 排序的索引
        indices = []
        for h in range(N_hilbert * N_hilbert):
            x, y = self._h_to_xy_with_order(h, needed_order)
            if x < nx and y < ny:
                indices.append((y, x))
        
        return np.array(indices)
    
    def _h_to_xy_with_order(self, h, order):
        """
        带指定阶数的 h_to_xy。
        """
        o = h & 3
        
        if o == 0:
            x, y = 0, 0
        elif o == 1:
            x, y = 1, 0
        elif o == 2:
            x, y = 1, 1
        else:
            x, y = 0, 1
        
        h >>= 2
        w = 2
        
        while h > 0:
            o = h & 3
            xold, yold = x, y
            
            if o == 0:
                x = yold
                y = xold
            elif o == 1:
                x = xold + w
                y = yold
            elif o == 2:
                x = xold + w
                y = yold + w
            else:
                x = w - 1 - yold
                y = w - 1 - xold
            
            h >>= 2
            w *= 2
        
        return x, y
    
    def compute_locality_index(self, ordered_indices):
        """
        计算排序的局部性指数。
        
        L = (1/N) Σ ||p_{i+1} - p_i||
        
        指数越小，局部性越好。
        """
        if len(ordered_indices) < 2:
            return 0.0
        
        total_distance = 0.0
        for i in range(len(ordered_indices) - 1):
            p1 = ordered_indices[i]
            p2 = ordered_indices[i + 1]
            dist = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
            total_distance += dist
        
        return total_distance / (len(ordered_indices) - 1)
    
    def compare_orderings(self, nx, ny):
        """
        比较 Hilbert 排序与行优先排序的局部性。
        
        Returns
        -------
        locality_hilbert : float
            Hilbert 排序局部性指数
        locality_row_major : float
            行优先排序局部性指数
        """
        # Hilbert 排序
        hilbert_indices = self.order_2d_grid(nx, ny)
        locality_hilbert = self.compute_locality_index(hilbert_indices)
        
        # 行优先排序
        row_major = []
        for j in range(ny):
            for i in range(nx):
                row_major.append((j, i))
        row_major = np.array(row_major)
        locality_row_major = self.compute_locality_index(row_major)
        
        return locality_hilbert, locality_row_major
    
    def reorder_field(self, field, ordered_indices):
        """
        按照 Hilbert 顺序重排场数据。
        
        Parameters
        ----------
        field : ndarray
            原始场数据，形状 (ny, nx)
        ordered_indices : ndarray
            Hilbert 排序索引
            
        Returns
        -------
        reordered : ndarray
            重排后的数据
        """
        reordered = np.zeros(len(ordered_indices))
        for idx, (j, i) in enumerate(ordered_indices):
            reordered[idx] = field[j, i]
        return reordered
    
    def inverse_reorder_field(self, reordered, ordered_indices, ny, nx):
        """
        将 Hilbert 排序的数据恢复为原始网格布局。
        """
        field = np.zeros((ny, nx))
        for idx, (j, i) in enumerate(ordered_indices):
            field[j, i] = reordered[idx]
        return field
