"""
tet_mesh.py
博士级大变形非线性有限元分析 — 四面体网格生成与细化模块

融合原项目:
  - 1247_tetrahedron_grid: 四面体参数化网格生成
  - 1238_tet_mesh_refine: 四面体网格细化（8-子四面体细分）

核心数学:
  1. 四面体参数化网格:
     给定四面体顶点 T = [v1, v2, v3, v4] (3x4),
     将每条边分为 n 段，内部点通过重心坐标生成:
       x = (i*v1 + j*v2 + k*v3 + l*v4) / n
     其中 i+j+k+l = n, i,j,k,l >= 0
     总点数: N = (n+1)(n+2)(n+3)/6

  2. 网格细化（8-细分）:
     每个四面体通过边中点细分为 8 个子四面体:
       - 原节点保留
       - 每条边新增一个中点节点
       - 新节点数: N2 = N1 + E_unique
       - 新单元数: T2 = 8 * T1

  3. 雅可比行列式检查:
     对线性四面体单元，体积为:
       V = |det(J)| / 6
     其中 J = [x2-x1, x3-x1, x4-x1] (3x3)
     要求 V > 0 以保证单元正向
"""

import numpy as np


class TetMesh:
    def __init__(self, nodes, elements):
        """
        nodes: (N, 3) ndarray, 节点坐标
        elements: (T, 4) ndarray, 四面体单元节点索引 (0-based)
        """
        self.nodes = np.array(nodes, dtype=float)
        self.elements = np.array(elements, dtype=int)
        self.n_nodes = self.nodes.shape[0]
        self.n_elements = self.elements.shape[0]

    def compute_volumes(self):
        """
        计算所有四面体单元的有向体积
        V = det([x2-x1, x3-x1, x4-x1]) / 6
        """
        vols = np.zeros(self.n_elements)
        for e in range(self.n_elements):
            idx = self.elements[e]
            x1, x2, x3, x4 = self.nodes[idx]
            J = np.column_stack([x2 - x1, x3 - x1, x4 - x1])
            vols[e] = np.linalg.det(J) / 6.0
        return vols

    def check_jacobian_positive(self, tol=1e-12):
        """
        检查所有单元的雅可比行列式是否为正
        """
        vols = self.compute_volumes()
        min_vol = np.min(vols)
        neg_count = np.sum(vols < tol)
        return min_vol, neg_count, vols


def tetrahedron_grid_count(n):
    """
    计算将四面体每条边分为 n 段后的总节点数
    N = (n+1)(n+2)(n+3) / 6
    源自原项目 1247_tetrahedron_grid (tetrahedron_grid_count)
    """
    return ((n + 1) * (n + 2) * (n + 3)) // 6


def generate_tetrahedron_grid(n, tet_vertices):
    """
    在单个四面体内生成参数化网格点

    输入:
        n: 每条边的分段数
        tet_vertices: (4, 3) 四面体顶点坐标
    输出:
        points: (N, 3) 网格点坐标

    源自原项目 1247_tetrahedron_grid (tetrahedron_grid)
    """
    tet_vertices = np.array(tet_vertices, dtype=float)
    ng = tetrahedron_grid_count(n)
    points = np.zeros((ng, 3))
    p = 0
    for i in range(n + 1):
        for j in range(n + 1 - i):
            for k in range(n + 1 - i - j):
                l = n - i - j - k
                points[p] = (i * tet_vertices[0] +
                             j * tet_vertices[1] +
                             k * tet_vertices[2] +
                             l * tet_vertices[3]) / n
                p += 1
    return points


def generate_cube_tet_mesh(nx, ny, nz, xlim=(0.0, 1.0), ylim=(0.0, 1.0), zlim=(0.0, 1.0)):
    """
    生成长方体区域的四面体网格（通过将六面体拆分为6个四面体）

    数学:
      节点编号: idx = i + j*(nx+1) + k*(nx+1)*(ny+1)
      每个六面体 (i,j,k) 拆分为6个四面体，保持一致的定向
    """
    dx = (xlim[1] - xlim[0]) / nx
    dy = (ylim[1] - ylim[0]) / ny
    dz = (zlim[1] - zlim[0]) / nz

    npx, npy, npz = nx + 1, ny + 1, nz + 1
    n_nodes = npx * npy * npz
    nodes = np.zeros((n_nodes, 3))

    for k in range(npz):
        for j in range(npy):
            for i in range(npx):
                idx = i + j * npx + k * npx * npy
                nodes[idx] = [xlim[0] + i * dx, ylim[0] + j * dy, zlim[0] + k * dz]

    elements = []
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                # 六面体8个角点索引
                n000 = i + j * npx + k * npx * npy
                n100 = (i + 1) + j * npx + k * npx * npy
                n110 = (i + 1) + (j + 1) * npx + k * npx * npy
                n010 = i + (j + 1) * npx + k * npx * npy
                n001 = i + j * npx + (k + 1) * npx * npy
                n101 = (i + 1) + j * npx + (k + 1) * npx * npy
                n111 = (i + 1) + (j + 1) * npx + (k + 1) * npx * npy
                n011 = i + (j + 1) * npx + (k + 1) * npx * npy

                # 将六面体拆分为6个四面体（确保定向一致，全部使用8个顶点）
                # 方案: 以空间对角线 n000-n111 为基础，全部体积为正
                tets = [
                    [n000, n101, n100, n111],
                    [n000, n001, n101, n111],
                    [n000, n011, n001, n111],
                    [n000, n010, n011, n111],
                    [n000, n110, n010, n111],
                    [n000, n100, n110, n111],
                ]
                elements.extend(tets)

    return TetMesh(nodes, elements)


def refine_tet_mesh(mesh):
    """
    对四面体网格进行一次8-子四面体细化

    源自原项目 1238_tet_mesh_refine (tet_mesh_order4_refine_compute / tet_mesh_order4_refine_size)

    数学:
      对原四面体的6条边各取中点:
        e12 = (n1+n2)/2, e13 = (n1+n3)/2, e14 = (n1+n4)/2,
        e23 = (n2+n3)/2, e24 = (n2+n4)/2, e34 = (n3+n4)/2

      8个子四面体:
        T1: [n1, e12, e13, e14]
        T2: [n2, e12, e23, e24]
        T3: [n3, e13, e23, e34]
        T4: [n4, e14, e24, e34]
        T5: [e12, e13, e14, e24]
        T6: [e12, e13, e23, e24]
        T7: [e13, e14, e24, e34]
        T8: [e13, e23, e24, e34]
    """
    old_nodes = mesh.nodes.copy()
    old_elements = mesh.elements.copy()
    n_old = old_nodes.shape[0]

    # 收集所有边并去重
    edge_to_mid = {}
    new_elements = []

    # 先遍历所有单元确定中点
    for elem in old_elements:
        edges = [(elem[0], elem[1]), (elem[0], elem[2]), (elem[0], elem[3]),
                 (elem[1], elem[2]), (elem[1], elem[3]), (elem[2], elem[3])]
        for e in edges:
            key = tuple(sorted(e))
            if key not in edge_to_mid:
                edge_to_mid[key] = None  # 占位

    # 分配中点节点索引和坐标
    mid_indices = {}
    new_node_list = list(old_nodes)
    for key in edge_to_mid:
        i, j = key
        mid_idx = len(new_node_list)
        mid_indices[key] = mid_idx
        new_node_list.append((old_nodes[i] + old_nodes[j]) * 0.5)

    new_nodes = np.array(new_node_list)

    # 生成8个子单元
    for elem in old_elements:
        n1, n2, n3, n4 = elem
        e12 = mid_indices[tuple(sorted((n1, n2)))]
        e13 = mid_indices[tuple(sorted((n1, n3)))]
        e14 = mid_indices[tuple(sorted((n1, n4)))]
        e23 = mid_indices[tuple(sorted((n2, n3)))]
        e24 = mid_indices[tuple(sorted((n2, n4)))]
        e34 = mid_indices[tuple(sorted((n3, n4)))]

        sub_tets = [
            [n1, e12, e13, e14],
            [n2, e12, e23, e24],
            [n3, e13, e23, e34],
            [n4, e14, e24, e34],
            [e12, e13, e14, e24],
            [e12, e13, e23, e24],
            [e13, e14, e24, e34],
            [e13, e23, e24, e34],
        ]
        new_elements.extend(sub_tets)

    return TetMesh(new_nodes, np.array(new_elements, dtype=int))
