#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
triangulation_boundary.py
=========================

基于种子项目 1331_triangulation_boundary 的三角剖分边界检测器。

科学背景
--------
在推荐系统中，用户群体并非均匀分布，而是形成多个"社区"（community）。
通过 Delaunay 三角剖分将用户嵌入空间离散化后，边界边（boundary edge）
仅出现一次，而内部边出现两次（正/反向各一次）。

边界边检测算法:
    1. 提取所有三角形的边:
       对每个三角形 (n1, n2, n3)，生成三条有向边:
           (n1, n2), (n2, n3), (n3, n1)
    2. 对每条边，将节点按小大排序，记录原始方向
    3. 按排序后的边排序
    4. 仅出现一次的边即为边界边
    
边界路径构造:
    从第一条边界边开始，寻找下一条以当前终点为起点的边，
    直到回到起点，形成闭合路径。

物理意义:
    边界节点代表"利基用户"（niche users），他们的偏好与主流群体不同，
    可能需要特殊的推荐策略。
"""

import numpy as np
from scipy.spatial import Delaunay


class TriangulationBoundaryDetector:
    """
    三角剖分边界检测器，用于识别用户社区的边缘。
    """
    
    def __init__(self):
        pass
    
    def detect_boundary_edges(self, triangles):
        """
        从三角形列表中检测边界边。
        
        参数:
            triangles : ndarray, shape (n_tri, 3)，每行是一个三角形的三个节点索引
            
        返回:
            boundary_edge : ndarray, shape (n_be, 2)
            
        算法:
            内部边出现两次（一次正序，一次逆序），边界边只出现一次。
            对排序后的边进行计数即可区分。
        """
        triangles = np.asarray(triangles, dtype=int)
        n_tri = triangles.shape[0]
        
        if n_tri == 0:
            return np.array([]).reshape(0, 2)
        
        # 生成所有有向边
        edge_num = 3 * n_tri
        edges = np.zeros((edge_num, 3), dtype=int)
        
        edges[0:n_tri, 0:2] = triangles[:, 0:2]
        edges[n_tri:2*n_tri, 0] = triangles[:, 1]
        edges[n_tri:2*n_tri, 1] = triangles[:, 2]
        edges[2*n_tri:3*n_tri, 0] = triangles[:, 2]
        edges[2*n_tri:3*n_tri, 1] = triangles[:, 0]
        
        # 记录原始方向: 1 表示 (min, max)，0 表示需要交换
        edges[:, 2] = (edges[:, 0] < edges[:, 1]).astype(int)
        
        # 排序每条边
        e1 = np.minimum(edges[:, 0], edges[:, 1])
        e2 = np.maximum(edges[:, 0], edges[:, 1])
        edges[:, 0] = e1
        edges[:, 1] = e2
        
        # 按边排序
        sort_idx = np.lexsort((edges[:, 1], edges[:, 0]))
        edges = edges[sort_idx]
        
        # 提取仅出现一次的边
        boundary_edges = []
        e = 0
        while e < edge_num:
            if e == edge_num - 1:
                # 最后一条边，只出现一次
                be = edges[e]
                e += 1
            else:
                if edges[e, 0] == edges[e+1, 0] and edges[e, 1] == edges[e+1, 1]:
                    # 出现两次，是内部边
                    e += 2
                    continue
                else:
                    be = edges[e]
                    e += 1
            
            # 恢复原始方向
            if be[2] == 1:
                boundary_edges.append([be[0], be[1]])
            else:
                boundary_edges.append([be[1], be[0]])
        
        return np.array(boundary_edges, dtype=int)
    
    def construct_boundary_path(self, boundary_edges):
        """
        将边界边列表构造成一条或多条闭合路径。
        
        算法:
            贪心法：从第一条边开始，依次寻找以下一条边的终点为起点的边。
            
        边界保护:
            - 边不连通时返回多条路径
            - 空输入返回空列表
        """
        if boundary_edges.size == 0:
            return []
        
        n_be = boundary_edges.shape[0]
        used = np.zeros(n_be, dtype=bool)
        paths = []
        
        while not np.all(used):
            # 找到第一个未使用的边
            start_idx = np.where(~used)[0][0]
            path = [boundary_edges[start_idx, 0]]
            current = boundary_edges[start_idx, 1]
            used[start_idx] = True
            path.append(current)
            
            # 寻找下一条边
            while True:
                found = False
                for i in range(n_be):
                    if used[i]:
                        continue
                    if boundary_edges[i, 0] == current:
                        current = boundary_edges[i, 1]
                        used[i] = True
                        path.append(current)
                        found = True
                        break
                if not found:
                    break
                if current == path[0]:
                    break
            
            paths.append(path)
        
        return paths
    
    def detect_boundary(self, points_2d):
        """
        对 2D 点集进行 Delaunay 三角剖分并检测边界节点。
        
        参数:
            points_2d : ndarray, shape (n, 2)
            
        返回:
            boundary_nodes : set of int
            
        边界保护:
            - 点数 < 3 时返回所有点
            - 退化共线时返回极值点
        """
        points = np.asarray(points_2d, dtype=float)
        n = points.shape[0]
        
        if n < 3:
            return set(range(n))
        
        try:
            tri = Delaunay(points)
            triangles = tri.simplices
        except Exception:
            # 退化情况：返回坐标极值点
            boundary_nodes = set()
            for dim in range(2):
                boundary_nodes.add(int(np.argmin(points[:, dim])))
                boundary_nodes.add(int(np.argmax(points[:, dim])))
            return boundary_nodes
        
        boundary_edges = self.detect_boundary_edges(triangles)
        if boundary_edges.size == 0:
            return set()
        
        boundary_nodes = set(boundary_edges.flatten())
        return boundary_nodes
