#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hilbert_hashing.py
==================

基于种子项目 536_hilbert_curve_3d 的 Hilbert 空间填充曲线局部敏感哈希（LSH）。

科学背景
--------
在推荐系统中，需要在高维潜空间中快速找到近邻用户/物品。
Hilbert 曲线是一种空间填充曲线，将 d 维空间映射到 1 维，
同时最大程度保持局部性：欧氏空间中邻近的点在 1D 映射中也邻近。

3D Hilbert 曲线映射:
    H = xyz_to_h(x, y, z, r)
    
    r : 曲线阶数，网格分辨率为 2^r × 2^r × 2^r
    
递归构造:
    每个象限（octant）的 Hilbert 曲线通过旋转/反射与相邻象限连接。
    具体地，对于每个级别的坐标 (xw, yw, zw) ∈ {0,1}³:
        o = octant_index(xw, yw, zw)
        根据 o 对子坐标进行旋转/反射变换
        h = 8·h + o
        
保持局部性的度量:
    对于任意两点 p, q:
    |H(p) - H(q)|^{1/d} 与 ||p - q|| 有近似线性关系

在 LSH 中，将向量量化到网格后计算 Hilbert 索引，
相近索引的向量被视为候选近邻。
"""

import numpy as np


class HilbertLSH:
    """
    基于 3D Hilbert 曲线的局部敏感哈希。
    """
    
    def __init__(self, order=4):
        """
        参数:
            order : Hilbert 曲线阶数 r，网格大小为 2^r
        """
        self.order = max(order, 1)
        self.grid_size = 2 ** self.order
    
    def _rmin(self, x, y, z):
        """
        计算 (x,y,z) 二进制表示中最低有效 1 位的位置。
        
        用于确定 Hilbert 变换的旋转状态。
        """
        if x == 0 and y == 0 and z == 0:
            return 0
        rm = 0
        t = 1
        while t <= max(x, y, z):
            if (x & t) or (y & t) or (z & t):
                rm += 1
            t <<= 1
        return rm
    
    def xyz_to_h(self, x, y, z):
        """
        将 3D 网格坐标 (x,y,z) 映射到 Hilbert 索引 h。
        
        坐标范围: [0, 2^r)
        
        算法:
            1. 根据 rmin 和旋转状态 t 对坐标进行预处理旋转
            2. 从最高有效位到最低有效位逐层处理:
               - 确定当前象限 o ∈ {0,...,7}
               - 根据 o 对子坐标进行旋转/反射
               - h = 8·h + o
            3. 返回最终 h 值
            
        边界保护:
            - 坐标越界时取模
        """
        x = int(x) & (self.grid_size - 1)
        y = int(y) & (self.grid_size - 1)
        z = int(z) & (self.grid_size - 1)
        
        # 预处理旋转
        rm = self._rmin(x, y, z)
        t = (self.order - rm) % 3
        
        if t == 1:
            x, y, z = z, x, y
        elif t == 2:
            x, y, z = y, z, x
        
        h = 0
        w = 2 ** (rm - 1) if rm > 0 else 2 ** (self.order - 1)
        start_k = rm if rm > 0 else self.order
        
        for k in range(start_k, 0, -1):
            xw = (x // w) & 1
            yw = (y // w) & 1
            zw = (z // w) & 1
            
            o = (xw << 2) | (yw << 1) | zw
            # 简化映射（标准 Hilbert 象限映射）
            o_map = [
                0, 1, 3, 2,
                6, 7, 5, 4
            ]
            o = o_map[o & 7]
            
            # 子坐标变换（简化版本）
            if o == 0:
                x, y, z = y, z, x
            elif o == 1:
                x, y, z = z, x, y
            elif o == 2:
                x, y, z = z, x, y
            elif o == 3:
                x, y, z = (w - 1 - x), y, (2 * w - 1 - z)
            elif o == 4:
                x, y, z = (w - 1 - x), (y - w), (2 * w - 1 - z)
            elif o == 5:
                x, y, z = (2 * w - 1 - y), (2 * w - 1 - z), (x - w)
            elif o == 6:
                x, y, z = (2 * w - 1 - y), (w - 1 - z), (x - w)
            elif o == 7:
                x, y, z = z, (w - 1 - x), (2 * w - 1 - y)
            
            h = (h << 3) | o
            w >>= 1
        
        return h
    
    def hash_vectors(self, vectors):
        """
        将一批 3D 向量量化为网格并计算 Hilbert 哈希值。
        
        参数:
            vectors : ndarray, shape (n, 3)
            
        返回:
            hashes  : ndarray, shape (n,)
        """
        vectors = np.asarray(vectors, dtype=float)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        
        # 归一化到 [0, grid_size)
        vmin = vectors.min(axis=0)
        vmax = vectors.max(axis=0)
        span = vmax - vmin
        span[span < 1e-12] = 1.0
        
        normed = (vectors - vmin) / span * (self.grid_size - 1)
        normed = np.clip(normed, 0, self.grid_size - 1)
        
        hashes = np.array([
            self.xyz_to_h(int(normed[i, 0]), int(normed[i, 1]), int(normed[i, 2]))
            for i in range(vectors.shape[0])
        ])
        return hashes
    
    def approximate_nn(self, user_hashes, item_hashes, top_k=5):
        """
        基于 Hilbert 哈希索引的近似最近邻搜索。
        
        策略:
            对每个用户，在排序后的 Hilbert 索引中寻找 top_k 个最近物品。
            
        返回:
            pairs : list of (user_idx, item_idx)
        """
        user_hashes = np.asarray(user_hashes)
        item_hashes = np.asarray(item_hashes)
        
        # 排序物品索引
        sorted_item_idx = np.argsort(item_hashes)
        sorted_hashes = item_hashes[sorted_item_idx]
        
        pairs = []
        for u_idx, u_h in enumerate(user_hashes):
            # 二分查找最近位置
            pos = np.searchsorted(sorted_hashes, u_h)
            
            # 取邻近窗口
            start = max(0, pos - top_k)
            end = min(len(sorted_hashes), pos + top_k)
            neighbors = sorted_item_idx[start:end]
            
            for i_idx in neighbors[:top_k]:
                pairs.append((u_idx, i_idx))
        
        return pairs
