#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
 variomino_adaptive_cover.py
 
 融合种子项目：
   - 1389_variomino (variomino_matrix)：变体多米诺平铺矩阵系统
 
 科学功能：
   自适应多分辨率网格覆盖。
   
   在海啸模拟中，近场（震中附近）需要高分辨率，
   远场可以使用低分辨率。本模块借鉴 variomino 平铺思想，
   将不同大小的"瓦片"（对应不同分辨率）自适应地覆盖计算域，
   实现计算资源的最优分配。
 
 核心数学公式：
 
   1) 变体多米诺（Variomino）平铺：
      给定一个区域 R 和一组变体瓦片 P，寻找一种平铺方式，
      使得 R 被 P 完全覆盖且不重叠。
      
      数学表述为整数线性规划：
        A · x = b
        x_i ∈ {0, 1}
      
      其中 A 为覆盖矩阵，x 为瓦片放置指示向量，b 为区域覆盖约束。
   
   2) 自适应覆盖准则：
      对于海啸波高场 η(x,y)，定义局部梯度：
        |∇η| = √((∂η/∂x)² + (∂η/∂y)²)
      
      细化准则：
        若 |∇η| > η_threshold，则该区域使用细瓦片（高分辨率）
        若 |∇η| ≤ η_threshold，则该区域使用粗瓦片（低分辨率）
   
   3) 分辨率层次：
      第 0 层（最粗）：瓦片大小 L × L
      第 k 层：瓦片大小 L/2^k × L/2^k
      
      总瓦片数约束：Σ_k N_k · (L/2^k)² = |R|
      
   4) 覆盖效率指标：
      计算资源节省率 = 1 - (实际瓦片数 / 均匀细网格单元数)
"""

import numpy as np


class AdaptiveMeshCover:
    """
    自适应多分辨率网格覆盖器。
    """
    
    def __init__(self):
        pass
    
    def generate_adaptive_cover(self, field, threshold=0.1, max_level=3):
        """
        基于场梯度生成自适应覆盖。
        
        Parameters
        ----------
        field : ndarray
            输入场数据（如初始波高）
        threshold : float
            细化阈值
        max_level : int
            最大细化层数
            
        Returns
        -------
        cover_mask : ndarray
            细化区域掩码（True 表示需要细化）
        """
        ny, nx = field.shape
        
        # 计算梯度场
        grad_x = np.zeros_like(field)
        grad_y = np.zeros_like(field)
        
        grad_x[:, :-1] = field[:, 1:] - field[:, :-1]
        grad_y[:-1, :] = field[1:, :] - field[:-1, :]
        
        gradient = np.sqrt(grad_x**2 + grad_y**2)
        
        # 初始掩码：梯度大于阈值的区域
        cover_mask = gradient > threshold
        
        # 多层级细化（类似 variomino 的多尺度平铺）
        for level in range(1, max_level + 1):
            # 对当前掩码进行膨胀操作（包含相邻区域）
            cover_mask = self._dilate_mask(cover_mask)
        
        return cover_mask
    
    def _dilate_mask(self, mask):
        """
        掩码膨胀：将 True 区域扩展一层邻居。
        """
        dilated = mask.copy()
        
        # 上
        dilated[:-1, :] |= mask[1:, :]
        # 下
        dilated[1:, :] |= mask[:-1, :]
        # 左
        dilated[:, :-1] |= mask[:, 1:]
        # 右
        dilated[:, 1:] |= mask[:, :-1]
        
        return dilated
    
    def build_cover_matrix(self, field, max_level=3):
        """
        构建覆盖矩阵（借鉴 variomino_matrix 思想）。
        
        将自适应覆盖问题建模为线性系统 A·x = b，
        其中每个变量表示在某个位置放置某个尺寸的瓦片。
        
        Parameters
        ----------
        field : ndarray
            输入场
        max_level : int
            最大细化层数
            
        Returns
        -------
        A : ndarray
            覆盖矩阵
        b : ndarray
            右端项（每个网格单元必须被覆盖一次）
        cover_info : list
            每个变量对应的瓦片信息
        """
        ny, nx = field.shape
        
        # 收集所有可能的瓦片放置
        cover_info = []
        
        for level in range(max_level + 1):
            tile_size = 2 ** (max_level - level)
            
            for j in range(0, ny - tile_size + 1, tile_size):
                for i in range(0, nx - tile_size + 1, tile_size):
                    cover_info.append({
                        'level': level,
                        'size': tile_size,
                        'j': j,
                        'i': i
                    })
        
        n_vars = len(cover_info)
        n_cells = ny * nx
        
        # 构建覆盖矩阵
        A = np.zeros((n_cells, n_vars), dtype=int)
        b = np.ones(n_cells, dtype=int)
        
        for var_idx, info in enumerate(cover_info):
            tile_size = info['size']
            j0, i0 = info['j'], info['i']
            
            for dj in range(tile_size):
                for di in range(tile_size):
                    cell_idx = (j0 + dj) * nx + (i0 + di)
                    A[cell_idx, var_idx] = 1
        
        return A, b, cover_info
    
    def compute_cover_efficiency(self, cover_mask, max_level=3):
        """
        计算自适应覆盖的计算资源节省率。
        
        Returns
        -------
        savings : float
            节省比例 [0, 1]
        """
        ny, nx = cover_mask.shape
        total_cells = ny * nx
        
        # 均匀细网格的单元数
        fine_cells = total_cells
        
        # 自适应覆盖的等效单元数
        # 粗区域用 2^max_level × 2^max_level 的瓦片
        coarse_cells = np.sum(~cover_mask)
        fine_cover_cells = np.sum(cover_mask)
        
        # 粗瓦片等效为一个大单元
        coarse_equiv = coarse_cells / (4 ** max_level)
        adaptive_equiv = coarse_equiv + fine_cover_cells
        
        savings = 1.0 - (adaptive_equiv / fine_cells)
        
        return savings
    
    def apply_cover_to_field(self, field, cover_mask, coarse_value_fn=np.mean):
        """
        将自适应覆盖应用于场数据。
        
        粗区域使用聚合值，细区域保留原始值。
        """
        result = field.copy()
        
        # 对非细化区域进行粗化
        ny, nx = field.shape
        block_size = 4  # 粗化块大小
        
        for j in range(0, ny - block_size + 1, block_size):
            for i in range(0, nx - block_size + 1, block_size):
                block_mask = cover_mask[j:j+block_size, i:i+block_size]
                
                if not np.any(block_mask):  # 完全粗区域
                    block = field[j:j+block_size, i:i+block_size]
                    coarse_val = coarse_value_fn(block)
                    result[j:j+block_size, i:i+block_size] = coarse_val
        
        return result
    
    def variomino_transformations(self, tile):
        """
        计算瓦片的所有旋转变换（借鉴 variomino_transform）。
        
        Parameters
        ----------
        tile : ndarray
            二维二进制瓦片
            
        Returns
        -------
        variants : list
            所有唯一的旋转变体
        """
        variants = [tile.copy()]
        
        current = tile.copy()
        for _ in range(3):
            # 旋转 90 度
            current = np.rot90(current)
            # 检查是否已存在
            is_new = True
            for v in variants:
                if np.array_equal(current, v):
                    is_new = False
                    break
            if is_new:
                variants.append(current.copy())
        
        return variants
