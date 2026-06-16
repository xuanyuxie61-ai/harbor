"""
mesh_topology.py
================
网格拓扑与 FEM 基函数模块。

整合:
  1. mesh_etoe (源自 755): 单元到单元连接关系
  2. fem_basis (源自 371): 有限元基函数
  3. 材料微观结构网格专用处理

数学背景:
  单元-边-面拓扑 (DG 方法必需):
    ETOV: Element TO Vertices
    ETOE: Element TO Element
    ETOF: Element TO Face (边)

  FEM 基函数 (1D P2 - 二次 Lagrange):
    N_1(xi) = 2*xi^2 - 3*xi + 1 = (1-xi)(1-2xi)
    N_2(xi) = 4*xi*(1-xi)
    N_3(xi) = 2*xi^2 - xi = xi(2xi-1)

  2D P1 三角形 (线性):
    N_1 = L1 (重心坐标)
    N_2 = L2
    N_3 = L3 = 1 - L1 - L2

  2D P2 三角形 (6节点, 二次):
    N_1 = L1*(2*L1-1)
    N_2 = L2*(2*L2-1)
    N_3 = L3*(2*L3-1)
    N_4 = 4*L1*L2
    N_5 = 4*L2*L3
    N_6 = 4*L3*L1

  3D P1 四面体:
    N_i = L_i, i = 1,2,3,4 (体积坐标)

  Jacobian 变换:
    dx/dxi = sum_j x_j * dN_j/dxi
    J = [dx/dxi, dx/deta; dy/dxi, dy/deta]
    det(J) = 2 * Area (三角形)
"""

import numpy as np
from material_constants import SMALL_NUMBER


# ============================================================================
# 单元到单元连接 (源自 755_mesh_etoe)
# ============================================================================
def mesh_etoe(e_order, e_num, etov):
    """
    构建单元到单元连接表 (源自 755)。

    算法:
      1. 提取所有面 (边), 每面附带单元号和局部面号
      2. 对每个面的节点排序形成键
      3. 排序所有面
      4. 相邻的相同键对应的两个单元互为邻接

    参数:
        e_order: 每个单元的节点数 (三角形为 3)
        e_num: 单元总数
        etov: [e_order, e_num] 单元-节点表 (1-indexed)

    返回:
        etoe: [e_order, e_num] 邻接单元表
              etoe(f, e) = 与单元 e 的第 f 个面相邻的单元号
              = 0 表示边界
    """
    etov = np.asarray(etov, dtype=np.int64)
    if etov.shape[0] != e_order:
        etov = etov.T

    # 面的定义 (三角形): 边 f 对面为节点 f
    # 边 1: 节点 2-3
    # 边 2: 节点 1-3
    # 边 3: 节点 1-2
    face_nodes = {
        3: [[2, 3], [1, 3], [1, 2]],
        4: [[2, 3, 4], [1, 3, 4], [1, 2, 4], [1, 2, 3]],  # 四面体
    }
    if e_order not in face_nodes:
        face_nodes[e_order] = [[(i % e_order) + 1 for i in range(j, j + e_order - 1)] for j in range(e_order)]

    fnodes = face_nodes[e_order]
    n_faces = len(fnodes)

    # 构建面列表
    face_list = []
    for e in range(e_num):
        for f_idx, fnode in enumerate(fnodes):
            nodes = tuple(sorted(etov[n - 1, e] for n in fnode))
            face_list.append((nodes, e + 1, f_idx + 1))

    # 排序
    face_list.sort(key=lambda x: x[0])

    etoe = np.zeros((n_faces, e_num), dtype=np.int64)
    i = 0
    n_total = len(face_list)
    while i < n_total:
        j = i + 1
        while j < n_total and face_list[j][0] == face_list[i][0]:
            j += 1
        if j - i == 2:
            _, e_a, f_a = face_list[i]
            _, e_b, f_b = face_list[j - 1]
            etoe[f_a - 1, e_a - 1] = e_b
            etoe[f_b - 1, e_b - 1] = e_a
        i = j

    return etoe


# ============================================================================
# FEM 基函数 (源自 371)
# ============================================================================
def fem_basis_1d_p2(xi):
    """
    1D P2 二次 Lagrange 基函数 (参考单元 [0, 1]):

    N_1(xi) = (1-xi)(1-2xi) = 2xi^2 - 3xi + 1
    N_2(xi) = 4xi(1-xi)
    N_3(xi) = xi(2xi-1) = 2xi^2 - xi

    返回: (N, dN_dxi) 各 [3]
    """
    N = np.array([
        (1.0 - xi) * (1.0 - 2.0*xi),
        4.0 * xi * (1.0 - xi),
        xi * (2.0 * xi - 1.0),
    ])
    dN = np.array([
        4.0 * xi - 3.0,
        4.0 - 8.0 * xi,
        4.0 * xi - 1.0,
    ])
    return N, dN


def fem_basis_triangle_p1(L1, L2):
    """
    2D P1 三角形线性基函数 (重心坐标):

    N_1 = L1, N_2 = L2, N_3 = 1 - L1 - L2

    在物理坐标 (x, y) 下的梯度:
    grad N_i = (1/(2*A)) * [y_j - y_k, x_k - x_j]

    返回: (N, dN_dL) 其中 N = [3], dN_dL = [[dN1/dL1, dN1/dL2], ...]
    """
    L3 = 1.0 - L1 - L2
    N = np.array([L1, L2, L3])
    dN_dL = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [-1.0, -1.0],
    ])
    return N, dN_dL


def fem_basis_triangle_p2(L1, L2):
    """
    2D P2 三角形二次基函数 (6 节点):

    N_1 = L1*(2*L1 - 1)
    N_2 = L2*(2*L2 - 1)
    N_3 = L3*(2*L3 - 1), L3 = 1 - L1 - L2
    N_4 = 4*L1*L2
    N_5 = 4*L2*L3
    N_6 = 4*L3*L1
    """
    L3 = 1.0 - L1 - L2
    N = np.array([
        L1 * (2*L1 - 1),
        L2 * (2*L2 - 1),
        L3 * (2*L3 - 1),
        4*L1*L2,
        4*L2*L3,
        4*L3*L1,
    ])
    # dN/dL1, dN/dL2
    dN = np.array([
        [4*L1 - 1, 0.0],
        [0.0, 4*L2 - 1],
        [- (4*L3 - 1), - (4*L3 - 1)],
        [4*L2, 4*L1],
        [-4*L2, 4*(L3 - L2)],
        [4*(L3 - L1), -4*L1],
    ])
    return N, dN


def fem_basis_tet_p1(L1, L2, L3):
    """
    3D P1 四面体基函数 (4 节点):

    N_i = L_i, i = 1,2,3
    N_4 = 1 - L1 - L2 - L3
    """
    L4 = 1.0 - L1 - L2 - L3
    N = np.array([L1, L2, L3, L4])
    dN = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [-1.0, -1.0, -1.0],
    ])
    return N, dN


# ============================================================================
# Jacobian 与梯度变换
# ============================================================================
def triangle_jacobian(node_xy, tri_nodes):
    """
    三角形 Jacobian 矩阵:

    J = [[x2-x1, x3-x1],
         [y2-y1, y3-y1]]

    det(J) = 2 * Area (有符号)
    J^{-1} 用于将 dN/d(L) 转换为 dN/d(x,y)
    """
    v1 = node_xy[:, tri_nodes[0] - 1]
    v2 = node_xy[:, tri_nodes[1] - 1]
    v3 = node_xy[:, tri_nodes[2] - 1]
    J = np.array([
        [v2[0] - v1[0], v3[0] - v1[0]],
        [v2[1] - v1[1], v3[1] - v1[1]],
    ])
    detJ = J[0, 0]*J[1, 1] - J[0, 1]*J[1, 0]
    return J, detJ


def physical_gradient(dN_dL, J_inv):
    """
    物理坐标下的形函数梯度:

    [dN/dx, dN/dy]^T = J^{-T} * [dN/dL1, dN/dL2]^T
    """
    return dN_dL @ J_inv.T


# ============================================================================
# 材料微观结构专用
# ============================================================================
def grain_boundary_mesh(n_grains, grain_size, Lx, Ly, seed=42):
    """
    简单的多晶材料微观结构网格 (Voronoi 型)。

    算法:
      1. 随机生成 n_grains 个晶核
      2. 每个晶核作为种子点构建 Voronoi 镶嵌
      3. 简化为三角形网格 (Delaunay 对偶)

    这里使用简化版本: 规则网格上扰动
    """
    rng = np.random.RandomState(seed)
    nx = int(np.sqrt(n_grains)) + 1
    ny = nx
    x_base = np.linspace(0, Lx, nx)
    y_base = np.linspace(0, Ly, ny)
    X, Y = np.meshgrid(x_base, y_base)
    # 扰动
    dx = Lx / (nx - 1) * 0.3
    dy = Ly / (ny - 1) * 0.3
    X += rng.uniform(-dx, dx, X.shape)
    Y += rng.uniform(-dy, dy, Y.shape)
    # 边界固定
    X[0, :] = 0; X[-1, :] = Lx
    Y[:, 0] = 0; Y[:, -1] = Ly

    node_xy = np.vstack([X.flatten(), Y.flatten()])

    # 三角形化
    triangles = []
    for i in range(nx - 1):
        for j in range(ny - 1):
            n1 = i * ny + j + 1
            n2 = (i+1) * ny + j + 1
            n3 = (i+1) * ny + (j+1) + 1
            n4 = i * ny + (j+1) + 1
            triangles.append([n1, n2, n3])
            triangles.append([n1, n3, n4])
    triangle_node = np.array(triangles, dtype=np.int64).T
    return node_xy, triangle_node


def assign_grain_ids(node_xy, n_grains, Lx, Ly, seed=42):
    """
    为每个节点分配晶粒 ID (基于最近晶核)。
    """
    rng = np.random.RandomState(seed)
    grain_centers = rng.uniform(0, 1, (n_grains, 2))
    grain_centers[:, 0] *= Lx
    grain_centers[:, 1] *= Ly

    n_node = node_xy.shape[1]
    grain_ids = np.zeros(n_node, dtype=np.int64)
    for i in range(n_node):
        dists = np.sum((grain_centers - node_xy[:, i:i+1].T)**2, axis=1)
        grain_ids[i] = np.argmin(dists)
    return grain_ids
