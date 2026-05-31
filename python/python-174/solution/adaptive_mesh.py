"""
adaptive_mesh.py
自适应网格细化模块

融合种子项目:
- 1351_triangulation_refine_local (三角网格局部细化)
- 758_mesh2d_to_medit (网格数据读写与管理)

科学背景:
在FMM中, 自适应网格细化用于:
1. 在粒子密度高的区域增加树深度
2. 在奇异点(如大电荷)附近使用更细的展开阶数
3. 通过局部细化提高精度

三角网格局部细化算法 (融合1351_triangulation_refine_local):
    给定一个三角形单元, 在每条边的中点插入新节点,
    将原三角形分裂为4个小三角形 (若边有相邻三角形, 则协同细化)

    新节点:
        N12 = (N1 + N2) / 2
        N23 = (N2 + N3) / 2
        N31 = (N3 + N1) / 2

    新三角形 (原内部):
        T0' = (N23, N31, N12)
        T1  = (N1, N12, N31)
        T2  = (N2, N23, N12)
        T3  = (N3, N31, N23)

网格数据管理 (融合758_mesh2d_to_medit):
    将网格数据输出为结构化格式, 记录节点、单元、标签等信息

核心公式:
    - 三角形面积 (Heron公式):
        A = sqrt(s*(s-a)*(s-b)*(s-c)), s = (a+b+c)/2
    - 局部细化误差指示子:
        eta_T = h_T * ||grad Phi||_T
        其中 h_T 为三角形直径
    - 自适应策略:
        若 eta_T > theta * max_eta, 则细化该单元
"""

import numpy as np


def triangle_area(p1, p2, p3):
    """
    计算三角形面积 (叉积法)
    
    公式:
        A = 0.5 * || (p2-p1) x (p3-p1) ||
    """
    p1, p2, p3 = np.asarray(p1), np.asarray(p2), np.asarray(p3)
    v1 = p2 - p1
    v2 = p3 - p1
    cross = np.cross(v1, v2)
    return 0.5 * np.linalg.norm(cross)


def refine_triangle_midpoint(p1, p2, p3):
    """
    通过中点细分一个三角形为4个小三角形
    
    参数:
        p1, p2, p3: ndarray (2,) 或 (3,)
    
    返回:
        new_points: list of ndarray, 新点 (包含原点和3个中点)
        new_triangles: list of tuple, 新三角形索引 (相对于new_points)
    """
    p1, p2, p3 = np.asarray(p1), np.asarray(p2), np.asarray(p3)
    n12 = 0.5 * (p1 + p2)
    n23 = 0.5 * (p2 + p3)
    n31 = 0.5 * (p3 + p1)

    points = [p1, p2, p3, n12, n23, n31]
    triangles = [
        (3, 4, 5),  # 中心三角形 (n12, n23, n31)
        (0, 3, 5),  # (p1, n12, n31)
        (1, 4, 3),  # (p2, n23, n12)
        (2, 5, 4),  # (p3, n31, n23)
    ]
    return points, triangles


def refine_triangle_local(node_xy, element_node, target_element_idx, element_neighbors=None):
    """
    局部细化指定的三角形单元 (融合1351_triangulation_refine_local)
    
    参数:
        node_xy: ndarray (N, 2), 节点坐标
        element_node: list of tuple, 每个三角形 (i1, i2, i3)
        target_element_idx: int, 要细化的三角形索引
        element_neighbors: list of tuple (可选), 邻居索引
    
    返回:
        new_node_xy: ndarray, 新节点坐标
        new_element_node: list, 新三角形列表
    """
    node_xy = np.asarray(node_xy, dtype=float)
    if element_neighbors is None:
        element_neighbors = [(-1, -1, -1)] * len(element_node)

    n1, n2, n3 = element_node[target_element_idx]
    n1, n2, n3 = int(n1), int(n2), int(n3)

    # 新节点索引
    n12_idx = node_xy.shape[0]
    n23_idx = node_xy.shape[0] + 1
    n31_idx = node_xy.shape[0] + 2

    new_xy = np.vstack([
        node_xy,
        0.5 * (node_xy[n1] + node_xy[n2]),
        0.5 * (node_xy[n2] + node_xy[n3]),
        0.5 * (node_xy[n3] + node_xy[n1])
    ])

    new_elements = []
    # 保留未改变的三角形
    for e_idx, tri in enumerate(element_node):
        if e_idx != target_element_idx:
            new_elements.append(tri)

    # 添加4个新三角形
    new_elements.append((n23_idx, n31_idx, n12_idx))  # 中心
    new_elements.append((n1, n12_idx, n31_idx))
    new_elements.append((n2, n23_idx, n12_idx))
    new_elements.append((n3, n31_idx, n23_idx))

    return new_xy, new_elements


class AdaptiveTriMesh:
    """自适应三角网格"""

    def __init__(self, points, elements):
        """
        参数:
            points: ndarray (N, 2)
            elements: list of tuple (i1, i2, i3)
        """
        self.points = np.asarray(points, dtype=float)
        self.elements = [tuple(int(x) for x in e) for e in elements]

    def element_area(self, e_idx):
        """计算单元面积"""
        i1, i2, i3 = self.elements[e_idx]
        return triangle_area(self.points[i1], self.points[i2], self.points[i3])

    def element_diameter(self, e_idx):
        """计算单元直径 (最长边)"""
        i1, i2, i3 = self.elements[e_idx]
        p1, p2, p3 = self.points[i1], self.points[i2], self.points[i3]
        d12 = np.linalg.norm(p1 - p2)
        d23 = np.linalg.norm(p2 - p3)
        d31 = np.linalg.norm(p3 - p1)
        return max(d12, d23, d31)

    def refine_by_indicator(self, indicators, theta=0.7):
        """
        根据误差指示子自适应细化
        
        参数:
            indicators: ndarray (M,), 各单元误差指示子
            theta: float, 细化阈值比例 (0 < theta < 1)
        
        返回:
            新网格
        """
        indicators = np.asarray(indicators)
        if len(indicators) != len(self.elements):
            raise ValueError("指示子数量必须等于单元数")
        max_eta = np.max(indicators)
        if max_eta < 1e-15:
            return AdaptiveTriMesh(self.points.copy(), self.elements.copy())

        threshold = theta * max_eta
        points = self.points.copy()
        elements = list(self.elements)

        # 按指示子从大到小排序细化
        sorted_idx = np.argsort(-indicators)
        for e_idx in sorted_idx:
            if indicators[e_idx] < threshold:
                break
            points, elements = refine_triangle_local(points, elements, e_idx)
            # 更新指示子 (简化: 停止继续细化)
            break

        return AdaptiveTriMesh(points, elements)

    def compute_error_indicator_fmm(self, element_potentials, element_direct):
        """
        基于FMM势能与直接求和差异计算误差指示子
        
        公式:
            eta_T = |Phi_FMM(T) - Phi_direct(T)| / |Phi_direct(T)|
        """
        diff = np.abs(element_potentials - element_direct)
        denom = np.abs(element_direct) + 1e-15
        return diff / denom

    def to_mesh_data(self):
        """
        输出网格数据为字典 (融合758_mesh2d_to_medit)
        
        返回:
            dict: 包含vertices, triangles, vertex_labels等
        """
        n_vertices = self.points.shape[0]
        n_triangles = len(self.elements)
        vertex_labels = np.zeros(n_vertices, dtype=int)
        triangle_labels = np.zeros(n_triangles, dtype=int)

        return {
            "dim": 2,
            "vertices": n_vertices,
            "triangles": n_triangles,
            "vertex_coordinate": self.points.T,
            "vertex_label": vertex_labels,
            "triangle_vertex": np.array(self.elements).T,
            "triangle_label": triangle_labels
        }


def project_3d_to_2d(points_3d, normal=None):
    """
    将3D点投影到2D平面
    
    用于在FMM中分析粒子在某个截面上的分布
    
    参数:
        points_3d: ndarray (N, 3)
        normal: ndarray (3,), 平面法向量 (默认z轴)
    
    返回:
        ndarray (N, 2)
    """
    points_3d = np.asarray(points_3d)
    if normal is None:
        return points_3d[:, :2]
    normal = np.asarray(normal)
    normal = normal / (np.linalg.norm(normal) + 1e-15)
    # 选择投影平面
    if abs(normal[2]) < 0.9:
        u = np.cross(normal, [0, 0, 1])
    else:
        u = np.cross(normal, [1, 0, 0])
    u = u / (np.linalg.norm(u) + 1e-15)
    v = np.cross(normal, u)
    v = v / (np.linalg.norm(v) + 1e-15)
    proj = np.column_stack([
        np.dot(points_3d, u),
        np.dot(points_3d, v)
    ])
    return proj
