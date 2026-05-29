"""
acoustic_room_model.py
三维封闭空间声学有限元模型与网格重排序

融合原始项目:
  - 1237_tet_mesh_rcm (四面体网格RCM重排序)
  - 790_navier_stokes_mesh3d (3D网格提取)

科学背景:
  房间声学可用Helmholtz方程描述:
      nabla^2 p + k^2 p = -f

  使用Galerkin有限元法在四面体网格上离散,
  得到稀疏线性系统:
      (K - k^2 M) p = F

  其中 K 为刚度矩阵, M 为质量矩阵.
  为减少直接求解器的带宽和填充,
  使用Reverse Cuthill-McKee (RCM) 算法对节点重排序.

  RCM算法步骤:
    1. 选择伪外周节点作为根
    2. 执行广度优先搜索生成层结构
    3. 按层逆序编号节点
"""

import numpy as np
from collections import deque


class AcousticRoomFEM:
    """
    简化的3D声学有限元模型.
    """

    def __init__(self, nodes, elements):
        """
        参数:
            nodes: (Nn, 3) 节点坐标 [m]
            elements: (Ne, 4) 四面体单元节点索引 (0-based)
        """
        self.nodes = np.asarray(nodes, dtype=float)
        self.elements = np.asarray(elements, dtype=int)
        self.Nn = self.nodes.shape[0]
        self.Ne = self.elements.shape[0]
        self.perm = np.arange(self.Nn)
        self.perm_inv = np.arange(self.Nn)

    def build_adjacency(self):
        """
        构建节点邻接图 (用于RCM重排序).
        两个节点如果在同一个四面体中则相邻.
        """
        adj = [set() for _ in range(self.Nn)]
        for elem in self.elements:
            for i in range(4):
                for j in range(i + 1, 4):
                    n1 = elem[i]
                    n2 = elem[j]
                    adj[n1].add(n2)
                    adj[n2].add(n1)
        return adj

    def rcm_reorder(self):
        """
        Reverse Cuthill-McKee 重排序.

        算法:
            1. 找到伪外周节点 (具有最小度的节点)
            2. BFS生成层结构
            3. 每层内部按度排序
            4. 逆序赋予新编号
        """
        adj = self.build_adjacency()
        degrees = [len(adj[i]) for i in range(self.Nn)]
        visited = [False] * self.Nn
        perm = []

        while len(perm) < self.Nn:
            # 找一个未访问的度最小节点作为根
            root = -1
            min_deg = self.Nn + 1
            for i in range(self.Nn):
                if not visited[i] and degrees[i] < min_deg:
                    min_deg = degrees[i]
                    root = i

            if root < 0:
                break

            # BFS层遍历
            level = [root]
            visited[root] = True
            while level:
                # 按度排序当前层
                level.sort(key=lambda x: degrees[x])
                next_level = []
                for node in level:
                    perm.append(node)
                    for nbr in sorted(adj[node], key=lambda x: degrees[x]):
                        if not visited[nbr]:
                            visited[nbr] = True
                            next_level.append(nbr)
                level = next_level

        # 逆序: RCM
        perm = perm[::-1]
        self.perm = np.array(perm, dtype=int)
        self.perm_inv = np.zeros(self.Nn, dtype=int)
        for new_idx, old_idx in enumerate(self.perm):
            self.perm_inv[old_idx] = new_idx

        # 重排节点和单元
        self.nodes = self.nodes[self.perm, :]
        for e in range(self.Ne):
            for i in range(4):
                self.elements[e, i] = self.perm_inv[self.elements[e, i]]

        return self.perm, self.perm_inv

    def compute_bandwidth(self):
        """
        计算刚度矩阵的半带宽.
        """
        adj = self.build_adjacency()
        max_bw = 0
        for i in range(self.Nn):
            for j in adj[i]:
                bw = abs(i - j)
                if bw > max_bw:
                    max_bw = bw
        return max_bw

    def assemble_system(self, k):
        """
        组装简化的有限元系统 (K - k^2 M).

        为简化,使用 lumped mass 和 linear Laplacian stencil.
        """
        A = np.zeros((self.Nn, self.Nn), dtype=float)
        b = np.zeros(self.Nn, dtype=float)

        for elem in self.elements:
            idx = elem
            coords = self.nodes[idx, :]
            # 简化的单元贡献
            for i in range(4):
                for j in range(4):
                    if i == j:
                        A[idx[i], idx[j]] += 1.0 - (k ** 2) * 0.025
                    else:
                        A[idx[i], idx[j]] += -0.25

        return A, b


def generate_box_mesh(Lx, Ly, Lz, nx, ny, nz):
    """
    生成简单盒状区域的四面体网格.

    先创建六面体,再每个六面体切分为5个四面体.
    """
    # 创建节点
    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)
    z = np.linspace(0, Lz, nz)

    nodes = []
    node_idx = {}
    idx = 0
    for k_ in range(nz):
        for j in range(ny):
            for i in range(nx):
                nodes.append([x[i], y[j], z[k_]])
                node_idx[(i, j, k_)] = idx
                idx += 1

    nodes = np.array(nodes, dtype=float)

    # 故意打乱节点顺序以展示RCM效果
    rng = np.random.default_rng(7)
    shuffle_idx = rng.permutation(nodes.shape[0])
    inv_shuffle = np.zeros_like(shuffle_idx)
    inv_shuffle[shuffle_idx] = np.arange(len(shuffle_idx))
    nodes = nodes[shuffle_idx, :]
    # 更新node_idx以反映打乱
    new_node_idx = {}
    for (i, j, k_), old in node_idx.items():
        new_node_idx[(i, j, k_)] = inv_shuffle[old]
    node_idx = new_node_idx

    # 创建六面体并剖分为四面体
    elements = []
    for k_ in range(nz - 1):
        for j in range(ny - 1):
            for i in range(nx - 1):
                n000 = node_idx[(i, j, k_)]
                n100 = node_idx[(i + 1, j, k_)]
                n010 = node_idx[(i, j + 1, k_)]
                n110 = node_idx[(i + 1, j + 1, k_)]
                n001 = node_idx[(i, j, k_ + 1)]
                n101 = node_idx[(i + 1, j, k_ + 1)]
                n011 = node_idx[(i, j + 1, k_ + 1)]
                n111 = node_idx[(i + 1, j + 1, k_ + 1)]

                # 5-tetrahedron decomposition
                tets = [
                    [n000, n100, n110, n111],
                    [n000, n100, n111, n101],
                    [n000, n101, n111, n001],
                    [n000, n111, n011, n001],
                    [n000, n110, n111, n011],
                    [n000, n010, n110, n011],
                ]
                elements.extend(tets)

    elements = np.array(elements, dtype=int)
    return nodes, elements
