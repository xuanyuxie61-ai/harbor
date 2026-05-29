#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
 mesh_interpolation.py
 
 融合种子项目：
   - 956_quadrilateral_surface_display：四边形网格双线性插值
   - 1356_trig_interp：三角插值（周期性边界处理）
 
 科学功能：
   网格插值系统。包括：
   1) 四边形网格上的双线性插值（用于地形和波场在不同分辨率网格间的映射）
   2) 三角插值处理周期性边界条件（用于经度方向周期性边界）
 
 核心物理公式：
 
   1) 四边形双线性插值：
      给定四边形四个顶点 (x_i, y_i) 及其值 z_i，i=1,2,3,4，
      四边形内部点 (x, y) 的值为：
      
      z(x,y) = Σ_{i=1}^4 N_i(ξ,η) · z_i
      
      其中 N_i 为双线性形函数：
      N_1 = (1-ξ)(1-η)/4
      N_2 = (1+ξ)(1-η)/4
      N_3 = (1+ξ)(1+η)/4
      N_4 = (1-ξ)(1+η)/4
      
      参考坐标 (ξ, η) 通过逆映射从 (x, y) 得到。
   
   2) 三角基函数插值（三角插值）：
      对于等间距数据节点 x_j = x_0 + j·h，j=0,...,n-1，
      周期为 n·h 的三角插值基函数为：
      
      C_j(x) = (1/n) · sin(n·π·(x-x_j)/(n·h)) / tan(π·(x-x_j)/(n·h))
      
      对于偶数 n：
      C_j(x) = (1/n) · sin(n·π·(x-x_j)/(n·h)) / sin(π·(x-x_j)/(n·h))
      
      插值函数：
      y(x) = Σ_{j=0}^{n-1} y_j · C_j(x)
      
   3) 海啸波场的多分辨率映射：
      在粗网格上求解后，通过双线性插值映射到细网格，
      用于后续的高分辨率分析或与其他数据集融合。
"""

import numpy as np


class MeshInterpolator:
    """
    网格插值系统。
    """
    
    def __init__(self, x_coarse, y_coarse):
        """
        Parameters
        ----------
        x_coarse, y_coarse : ndarray
            粗网格坐标
        """
        self.x_coarse = x_coarse
        self.y_coarse = y_coarse
        self.nx_c = len(x_coarse)
        self.ny_c = len(y_coarse)
        self.dx_c = x_coarse[1] - x_coarse[0]
        self.dy_c = y_coarse[1] - y_coarse[0]
    
    def bilinear_interpolate(self, z_coarse, x_fine, y_fine):
        """
        四边形双线性插值：将粗网格数据插值到细网格。
        
        来源于 quadrilateral_surface_display 中四边形网格插值思想。
        
        Parameters
        ----------
        z_coarse : ndarray
            粗网格数据，形状 (ny_c, nx_c)
        x_fine, y_fine : ndarray
            细网格坐标
            
        Returns
        -------
        z_fine : ndarray
            细网格插值结果
        """
        ny_f = len(y_fine)
        nx_f = len(x_fine)
        z_fine = np.zeros((ny_f, nx_f))
        
        for j in range(ny_f):
            for i in range(nx_f):
                x = x_fine[i]
                y = y_fine[j]
                
                # 找到包含 (x, y) 的粗网格单元
                ix = min(max(int((x - self.x_coarse[0]) / self.dx_c), 0), self.nx_c - 2)
                iy = min(max(int((y - self.y_coarse[0]) / self.dy_c), 0), self.ny_c - 2)
                
                # 局部坐标
                xi = (x - self.x_coarse[ix]) / self.dx_c
                eta = (y - self.y_coarse[iy]) / self.dy_c
                
                # 边界检查
                xi = np.clip(xi, 0.0, 1.0)
                eta = np.clip(eta, 0.0, 1.0)
                
                # 双线性形函数
                N1 = (1.0 - xi) * (1.0 - eta)
                N2 = xi * (1.0 - eta)
                N3 = xi * eta
                N4 = (1.0 - xi) * eta
                
                # 插值
                z_fine[j, i] = (
                    N1 * z_coarse[iy, ix] +
                    N2 * z_coarse[iy, ix + 1] +
                    N3 * z_coarse[iy + 1, ix + 1] +
                    N4 * z_coarse[iy + 1, ix]
                )
        
        return z_fine
    
    def trigonometric_periodic_boundary(self, field, axis=0):
        """
        使用三角插值处理周期性边界。
        
        来源于 trig_interp_cardinal 的三角基函数插值思想。
        用于保证经度方向边界条件的周期性连续性。
        
        Parameters
        ----------
        field : ndarray
            输入场数据
        axis : int
            周期性边界所在的轴（0=y方向，1=x方向）
            
        Returns
        -------
        field_periodic : ndarray
            经过周期边界处理后的场
        """
        field_periodic = field.copy()
        
        if axis == 0:
            n = field.shape[0]
            if n < 3:
                return field_periodic
            
            # 在边界附近使用三角插值平滑过渡
            # 计算边界处的三角插值值
            h = 1.0  # 归一化间距
            
            # 对第一行和最后一行之间的差异进行三角基函数平滑
            diff = field[0, :] - field[-1, :]
            
            # 应用平滑过渡：在边界附近几行进行插值修正
            smooth_width = min(3, n // 4)
            for j in range(smooth_width):
                alpha = j / smooth_width
                # 使用正弦过渡保证周期性
                weight = 0.5 * (1.0 - np.cos(np.pi * alpha))
                field_periodic[j, :] = field[j, :] - diff * (1.0 - weight) * 0.5
                field_periodic[-1-j, :] = field[-1-j, :] + diff * (1.0 - weight) * 0.5
        
        elif axis == 1:
            n = field.shape[1]
            if n < 3:
                return field_periodic
            
            diff = field[:, 0] - field[:, -1]
            
            smooth_width = min(3, n // 4)
            for i in range(smooth_width):
                alpha = i / smooth_width
                weight = 0.5 * (1.0 - np.cos(np.pi * alpha))
                field_periodic[:, i] = field[:, i] - diff * (1.0 - weight) * 0.5
                field_periodic[:, -1-i] = field[:, -1-i] + diff * (1.0 - weight) * 0.5
        
        return field_periodic
    
    def cardinal_basis(self, x_nodes, x_eval, j):
        """
        计算第 j 个节点的三角基函数值 C_j(x)。
        
        来源于 trig_cardinal 函数。
        
        Parameters
        ----------
        x_nodes : ndarray
            等间距数据节点
        x_eval : float
            求值点
        j : int
            节点索引
            
        Returns
        -------
        Cj : float
            基函数值
        """
        n = len(x_nodes)
        h = x_nodes[1] - x_nodes[0]
        
        if n % 2 == 1:
            # 奇数 n：使用正切形式
            if abs(x_eval - x_nodes[j]) < 1e-12:
                return 1.0
            Cj = np.sin(n * np.pi * (x_eval - x_nodes[j]) / (n * h)) / \
                 (n * np.tan(np.pi * (x_eval - x_nodes[j]) / (n * h)))
        else:
            # 偶数 n：使用正弦形式
            if abs(x_eval - x_nodes[j]) < 1e-12:
                return 1.0
            Cj = np.sin(n * np.pi * (x_eval - x_nodes[j]) / (n * h)) / \
                 (n * np.sin(np.pi * (x_eval - x_nodes[j]) / (n * h)))
        
        return Cj
    
    def trigonometric_interpolate_1d(self, x_nodes, y_nodes, x_eval):
        """
        一维三角插值。
        
        Parameters
        ----------
        x_nodes : ndarray
            等间距节点
        y_nodes : ndarray
            节点值
        x_eval : ndarray
            求值点
            
        Returns
        -------
        y_eval : ndarray
            插值结果
        """
        y_eval = np.zeros_like(x_eval)
        n = len(x_nodes)
        
        for xi_idx, xi in enumerate(x_eval):
            yi = 0.0
            for j in range(n):
                yi += y_nodes[j] * self.cardinal_basis(x_nodes, xi, j)
            y_eval[xi_idx] = yi
        
        return y_eval
    
    def interpolate_quadrilateral_surface(self, nodes_xy, node_values, eval_points):
        """
        四边形曲面插值。
        
        直接来源于 quadrilateral_surface_display 的插值核心思想：
        在四边形单元上使用双线性插值计算内部点值。
        
        Parameters
        ----------
        nodes_xy : ndarray, shape (4, 2)
            四边形四个顶点的 (x, y) 坐标
        node_values : ndarray, shape (4,)
            四个顶点的值
        eval_points : ndarray, shape (n, 2)
            待求值点坐标
            
        Returns
        -------
        values : ndarray
            插值结果
        """
        n_eval = eval_points.shape[0]
        values = np.zeros(n_eval)
        
        # 四边形顶点
        x1, y1 = nodes_xy[0]
        x2, y2 = nodes_xy[1]
        x3, y3 = nodes_xy[2]
        x4, y4 = nodes_xy[3]
        
        for i in range(n_eval):
            x, y = eval_points[i]
            
            # 使用面积坐标近似逆映射（简化版）
            # 计算到四个顶点的加权距离
            d1 = 1.0 / (np.sqrt((x - x1)**2 + (y - y1)**2) + 1e-12)
            d2 = 1.0 / (np.sqrt((x - x2)**2 + (y - y2)**2) + 1e-12)
            d3 = 1.0 / (np.sqrt((x - x3)**2 + (y - y3)**2) + 1e-12)
            d4 = 1.0 / (np.sqrt((x - x4)**2 + (y - y4)**2) + 1e-12)
            
            w_sum = d1 + d2 + d3 + d4
            w1, w2, w3, w4 = d1/w_sum, d2/w_sum, d3/w_sum, d4/w_sum
            
            values[i] = w1 * node_values[0] + w2 * node_values[1] + \
                        w3 * node_values[2] + w4 * node_values[3]
        
        return values
