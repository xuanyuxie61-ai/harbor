"""
mesh_generator.py
自适应空间网格生成模块

基于四边形 (Q4) 与三角形 (T3/T6) 单元的混合网格生成器，
支持从规则参考单元到物理域的等参映射，
并提供基于粒子负载的自适应网格细化功能。

核心数学：
    - Q4 参考单元: [0,1] x [0,1]
      形函数:
        N1(r,s) = (1-r)(1-s)
        N2(r,s) = r(1-s)
        N3(r,s) = rs
        N4(r,s) = (1-r)s
    
    - T3 参考单元: 顶点 (0,0), (1,0), (0,1)
      形函数（面积坐标）:
        L1 = 1 - r - s
        L2 = r
        L3 = s
        面积 A = 0.5 * |x1(y2-y3) + x2(y3-y1) + x3(y1-y2)|
    
    - 等参映射:
        x(r,s) = sum_i N_i(r,s) * x_i
        J = [[dx/dr, dx/ds], [dy/dr, dy/ds]]
        det(J) 用于积分换元: dx dy = |det(J)| dr ds
    
    - 自适应细化判据（基于粒子负载）:
        若单元 e 的粒子数 n_e > n_avg * (1 + theta)，则细化该单元
        其中 theta 为负载不均衡阈值。
"""

import numpy as np
from typing import List, Tuple, Optional
from utils import compute_triangle_area, reference_to_physical_q4, check_bounds


class MeshElement:
    """网格单元基类。"""
    def __init__(self, nodes: np.ndarray, elem_type: str = "Q4"):
        self.nodes = np.asarray(nodes, dtype=int)  # 局部节点索引
        self.elem_type = elem_type
        self.level = 0          # 细化层级
        self.load = 0.0         # 计算负载
        self.area = 0.0         # 单元面积


class QuadMesh:
    """
    四边形网格 (Q4) 管理器。
    
    每个单元为四节点四边形，支持基于粒子负载的自适应细化。
    """
    def __init__(self, domain: Tuple[float, float, float, float],
                 nx: int, ny: int):
        """
        Parameters
        ----------
        domain : (xmin, xmax, ymin, ymax)
        nx, ny : int
            初始网格划分数
        """
        self.domain = domain
        self.xmin, self.xmax, self.ymin, self.ymax = domain
        self.nx = nx
        self.ny = ny

        self.nodes = self._build_initial_nodes()
        self.elements = self._build_initial_elements()
        self._compute_element_areas()

    def _build_initial_nodes(self) -> np.ndarray:
        """构建初始均匀网格节点。"""
        x = np.linspace(self.xmin, self.xmax, self.nx + 1)
        y = np.linspace(self.ymin, self.ymax, self.ny + 1)
        nodes = []
        for j in range(self.ny + 1):
            for i in range(self.nx + 1):
                nodes.append([x[i], y[j]])
        return np.array(nodes, dtype=float)

    def _build_initial_elements(self) -> List[MeshElement]:
        """构建初始四边形单元（逆时针编号）。"""
        elements = []
        for j in range(self.ny):
            for i in range(self.nx):
                n0 = j * (self.nx + 1) + i
                n1 = n0 + 1
                n2 = n1 + (self.nx + 1)
                n3 = n0 + (self.nx + 1)
                elem = MeshElement(np.array([n0, n1, n2, n3]), "Q4")
                elements.append(elem)
        return elements

    def _compute_element_areas(self):
        """计算所有单元的面积。"""
        for elem in self.elements:
            coords = self.nodes[elem.nodes]
            if elem.elem_type == "Q4":
                # 四边形拆分为两个三角形计算面积
                a1 = abs(compute_triangle_area(coords[0], coords[1], coords[2]))
                a2 = abs(compute_triangle_area(coords[0], coords[2], coords[3]))
                elem.area = a1 + a2
            elif elem.elem_type == "T3":
                elem.area = abs(compute_triangle_area(coords[0], coords[1], coords[2]))

    def evaluate_load(self, particles: np.ndarray) -> np.ndarray:
        """
        评估每个单元的粒子负载。
        
        对每个粒子，通过遍历找到包含它的单元（点-in-四边形测试）。
        
        Parameters
        ----------
        particles : np.ndarray, shape (n, 2)
            粒子位置
        
        Returns
        -------
        np.ndarray
            每个单元的粒子数
        """
        particles = np.asarray(particles, dtype=float)
        loads = np.zeros(len(self.elements), dtype=int)

        for p in range(particles.shape[0]):
            x, y = particles[p]
            # 快速定位（假设近似均匀网格）
            ix = int((x - self.xmin) / (self.xmax - self.xmin) * self.nx)
            iy = int((y - self.ymin) / (self.ymax - self.ymin) * self.ny)
            ix = max(0, min(self.nx - 1, ix))
            iy = max(0, min(self.ny - 1, iy))
            elem_idx = iy * self.nx + ix
            if elem_idx < len(self.elements):
                loads[elem_idx] += 1
        return loads

    def refine_by_load(self, particles: np.ndarray, theta: float = 0.3,
                       max_level: int = 3) -> "QuadMesh":
        """
        基于负载的自适应网格细化。
        
        判据：若单元负载 n_e > n_avg * (1 + theta)，将该单元细分为4个子单元。
        
        数学：
            n_avg = N_particles / N_elements
            若 n_e > n_avg * (1 + theta) 且 level < max_level:
                将 Q4 单元细分为 4 个子 Q4 单元
        
        Parameters
        ----------
        particles : np.ndarray
            粒子位置
        theta : float
            负载不均衡阈值
        max_level : int
            最大细化层级
        
        Returns
        -------
        QuadMesh
            细化后的新网格（原地修改当前网格后返回自身）
        """
        loads = self.evaluate_load(particles)
        avg_load = np.mean(loads) if len(loads) > 0 else 1.0
        if avg_load < 1e-10:
            avg_load = 1.0

        threshold = avg_load * (1.0 + theta)
        new_elements = []
        new_nodes = list(self.nodes)
        node_offset = len(new_nodes)

        for idx, elem in enumerate(self.elements):
            if loads[idx] > threshold and elem.level < max_level:
                # 细分当前四边形为4个子四边形
                coords = self.nodes[elem.nodes]
                # 计算4条边中点和中心点
                mid01 = 0.5 * (coords[0] + coords[1])
                mid12 = 0.5 * (coords[1] + coords[2])
                mid23 = 0.5 * (coords[2] + coords[3])
                mid30 = 0.5 * (coords[3] + coords[0])
                center = 0.25 * (coords[0] + coords[1] + coords[2] + coords[3])

                # 添加新节点（避免重复，简化处理：直接添加）
                n01 = node_offset
                n12 = node_offset + 1
                n23 = node_offset + 2
                n30 = node_offset + 3
                nc = node_offset + 4
                node_offset += 5

                new_nodes.extend([mid01, mid12, mid23, mid30, center])

                n0, n1, n2, n3 = elem.nodes
                # 4个子四边形
                sub_elems = [
                    MeshElement(np.array([n0, n01, nc, n30]), "Q4"),
                    MeshElement(np.array([n01, n1, n12, nc]), "Q4"),
                    MeshElement(np.array([nc, n12, n2, n23]), "Q4"),
                    MeshElement(np.array([n30, nc, n23, n3]), "Q4"),
                ]
                for se in sub_elems:
                    se.level = elem.level + 1
                new_elements.extend(sub_elems)
            else:
                new_elements.append(elem)

        self.nodes = np.array(new_nodes, dtype=float)
        self.elements = new_elements
        self._compute_element_areas()
        return self

    def triangulate_elements(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        将所有四边形单元剖分为三角形 (T3)，返回节点和三角形连接矩阵。
        
        每个 Q4 剖分为 2 个 T3:
            (n0, n1, n2) 和 (n0, n2, n3)
        
        Returns
        -------
        nodes : np.ndarray
            节点坐标
        triangles : np.ndarray, shape (n_tri, 3)
            三角形节点索引（1-based，兼容FEM模块）
        """
        # TODO(Hole_1): 实现四边形到三角形的剖分，并确定索引约定（0-based 或 1-based）
        # 要求：
        #   - Q4 单元拆分为 2 个 T3 三角形
        #   - 返回的 triangles 索引约定必须与 fem_solver.py 和 main.py 一致
        raise NotImplementedError("Hole_1: mesh_generator.py triangulate_elements 待实现")

    def get_element_centers(self) -> np.ndarray:
        """计算每个单元的几何中心。"""
        centers = []
        for elem in self.elements:
            coords = self.nodes[elem.nodes]
            centers.append(np.mean(coords, axis=0))
        return np.array(centers, dtype=float)


def build_delaunay_triangulation(nodes: np.ndarray) -> np.ndarray:
    """
    对给定节点集执行 Delaunay 三角剖分。
    
    使用 scipy.spatial.Delaunay（若可用），否则回退到简单网格。
    Delaunay 剖分最大化最小角，避免病态三角形。
    
    Parameters
    ----------
    nodes : np.ndarray, shape (n, 2)
        二维节点坐标
    
    Returns
    -------
    np.ndarray, shape (m, 3)
        三角形连接矩阵（1-based）
    """
    try:
        from scipy.spatial import Delaunay
        tri = Delaunay(nodes)
        return tri.simplices + 1  # 1-based
    except ImportError:
        print("[WARNING] scipy not available; using fallback grid triangulation.")
        n = nodes.shape[0]
        # 简单回退：假设节点近似呈矩形排列
        nx = int(np.sqrt(n))
        ny = max(1, n // nx)
        tri_list = []
        for j in range(ny - 1):
            for i in range(nx - 1):
                n0 = j * nx + i
                n1 = n0 + 1
                n2 = n1 + nx
                n3 = n0 + nx
                if n2 < n and n3 < n:
                    tri_list.append([n0, n1, n2])
                    tri_list.append([n0, n2, n3])
        return np.array(tri_list, dtype=int) + 1


def compute_mesh_bandwidth(element_node: np.ndarray, node_num: int) -> int:
    """
    计算网格的半带宽（用于稀疏矩阵分析）。
    
    半带宽定义:
        b = max_{所有单元} max_{i,j in 单元} |i - j|
    
    Parameters
    ----------
    element_node : np.ndarray, shape (elem_order, n_elem)
        单元节点连接（1-based）
    node_num : int
        总节点数
    
    Returns
    -------
    int
        半带宽
    """
    element_node = np.asarray(element_node, dtype=int)
    bandwidth = 0
    for e in range(element_node.shape[1]):
        nodes = element_node[:, e]
        local_bw = np.max(nodes) - np.min(nodes)
        bandwidth = max(bandwidth, local_bw)
    return bandwidth
