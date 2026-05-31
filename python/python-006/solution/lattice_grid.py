"""
lattice_grid.py
中子星地壳晶格结构的多边形网格与四面体离散化模块

原项目映射:
- 885_polygon_grid  -> 多边形网格点生成，用于构建晶格Wigner-Seitz原胞
- 1230_tet_mesh     -> 四面体网格排序与线性方程组求解 (Gauss-Jordan消元)
"""

import numpy as np
import math
from typing import List, Tuple


# =============================================================================
# 多边形网格点生成 (源自 885_polygon_grid)
# =============================================================================
def polygon_grid_points(n: int, nv: int, v: np.ndarray, ng: int) -> np.ndarray:
    """
    在任意多边形内部生成规则网格点。

    算法: 将多边形分解为以形心为公共顶点的 nv 个三角形，
    在每个三角形内用重心坐标生成网格点。

    Parameters
    ----------
    n : int
        每条边上的子区间数。
    nv : int
        多边形顶点数。
    v : np.ndarray, shape (nv, 2)
        顶点坐标。
    ng : int
        预期的网格点总数（包含形心）。

    Returns
    -------
    xg : np.ndarray, shape (ng, 2)
        网格点坐标。
    """
    if n < 1:
        raise ValueError("n must be at least 1.")
    if nv < 3:
        raise ValueError("A polygon must have at least 3 vertices.")
    if ng < 1:
        raise ValueError("ng must be positive.")

    xg = np.zeros((ng, 2))
    p = 0

    # 形心作为第一个点
    vc = np.array([np.sum(v[:, 0]) / nv, np.sum(v[:, 1]) / nv])
    xg[p, :] = vc
    p += 1

    # 遍历每个由相邻顶点和形心构成的三角形
    for l in range(nv):
        lp1 = (l + 1) % nv
        for i in range(1, n + 1):
            for j in range(0, n - i + 1):
                if p >= ng:
                    return xg
                k = n - i - j
                xg[p, :] = (i * v[l, :] + j * v[lp1, :] + k * vc) / n
                p += 1

    return xg


def polygon_grid_count(n: int, nv: int) -> int:
    """
    计算 n 细分下多边形内部网格点的总数。

    公式:
        ng = 1 + nv * n * (n + 1) / 2
    """
    if n < 1 or nv < 3:
        raise ValueError("Invalid parameters.")
    return 1 + nv * n * (n + 1) // 2


# =============================================================================
# 四面体网格与线性求解 (源自 1230_tet_mesh)
# =============================================================================
def r8mat_solve(n: int, nrhs: int, a: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    使用Gauss-Jordan消元法求解 N×N 线性方程组。

    源自 1230_tet_mesh 中 r8mat_solve.m 的核心算法。
    增广矩阵 a 的形状为 (n, n+nrhs)，前 n 列为系数矩阵，
    后 nrhs 列为右端项。

    Parameters
    ----------
    n : int
        矩阵阶数。
    nrhs : int
        右端项列数。
    a : np.ndarray
        增广矩阵。

    Returns
    -------
    a : np.ndarray
        消元后的矩阵，后 nrhs 列被替换为解。
    info : int
        0 表示成功，非零表示第 info 步主元为零。
    """
    a = np.array(a, dtype=float)
    info = 0

    for j in range(n):
        # 选主元
        ipivot = j
        apivot = abs(a[j, j])
        for i in range(j + 1, n):
            if abs(a[i, j]) > apivot:
                apivot = abs(a[i, j])
                ipivot = i

        if apivot < 1e-30:
            info = j + 1
            return a, info

        # 交换行
        if ipivot != j:
            a[[j, ipivot], :] = a[[ipivot, j], :]

        # 归一化主元行
        pivot = a[j, j]
        a[j, :] /= pivot

        # 消去其他行
        for i in range(n):
            if i != j:
                factor = a[i, j]
                a[i, :] -= factor * a[j, :]

    return a, info


def solve_lattice_elasticity(nodes: np.ndarray, elements: List[Tuple[int, ...]],
                             boundary_nodes: List[int],
                             external_force: np.ndarray) -> np.ndarray:
    """
    求解晶格结构的弹性平衡方程 K u = f。

    使用有限元方法构建刚度矩阵 K，然后用 Gauss-Jordan 消元求解位移 u。

    公式:
        K_ij = Σ_e ∫_Ω_e B^T D B dΩ
        f_i  = 节点外力

    Parameters
    ----------
    nodes : np.ndarray, shape (n_nodes, 2)
        节点坐标。
    elements : list of tuple
        每个单元包含的节点编号。
    boundary_nodes : list of int
        边界约束节点编号。
    external_force : np.ndarray, shape (n_nodes, 2)
        节点外力。

    Returns
    -------
    displacement : np.ndarray, shape (n_nodes, 2)
        节点位移。
    """
    n_nodes = nodes.shape[0]
    ndof = 2 * n_nodes

    K = np.zeros((ndof, ndof))
    F = np.zeros(ndof)

    for i in range(n_nodes):
        F[2 * i] = external_force[i, 0]
        F[2 * i + 1] = external_force[i, 1]

    # 简化刚度矩阵组装（假设各向同性线弹性，E=1, ν=0.3）
    E_mod = 1.0
    nu = 0.3
    D_mat = (E_mod / (1.0 - nu**2)) * np.array([
        [1.0, nu, 0.0],
        [nu, 1.0, 0.0],
        [0.0, 0.0, (1.0 - nu) / 2.0]
    ])

    for elem in elements:
        if len(elem) == 3:
            # 三角形单元
            i, j, k = elem
            xi, yi = nodes[i]
            xj, yj = nodes[j]
            xk, yk = nodes[k]

            area = 0.5 * abs((xj - xi) * (yk - yi) - (xk - xi) * (yj - yi))
            if area < 1e-14:
                continue

            # B 矩阵 (3x6)
            b1 = yj - yk
            b2 = yk - yi
            b3 = yi - yj
            c1 = xk - xj
            c2 = xi - xk
            c3 = xj - xi

            B = (1.0 / (2.0 * area)) * np.array([
                [b1, 0.0, b2, 0.0, b3, 0.0],
                [0.0, c1, 0.0, c2, 0.0, c3],
                [c1, b1, c2, b2, c3, b3]
            ])

            Ke = area * B.T @ D_mat @ B

            local_dofs = [2 * i, 2 * i + 1, 2 * j, 2 * j + 1, 2 * k, 2 * k + 1]
            for ii in range(6):
                for jj in range(6):
                    K[local_dofs[ii], local_dofs[jj]] += Ke[ii, jj]

    # 施加边界条件（置零法）
    for bn in boundary_nodes:
        for d in range(2):
            dof = 2 * bn + d
            K[dof, :] = 0.0
            K[:, dof] = 0.0
            K[dof, dof] = 1.0
            F[dof] = 0.0

    # 求解
    aug = np.hstack([K, F.reshape(-1, 1)])
    sol, info = r8mat_solve(ndof, 1, aug)

    if info != 0:
        raise RuntimeError(f"Linear system is singular at step {info}.")

    displacement = np.zeros((n_nodes, 2))
    for i in range(n_nodes):
        displacement[i, 0] = sol[ndof - 1, 2 * i]
        displacement[i, 1] = sol[ndof - 1, 2 * i + 1]

    return displacement


# =============================================================================
# 中子星地壳晶格生成
# =============================================================================
def generate_crust_lattice_hexagonal(
    lattice_constant: float,
    n_layers: int
) -> Tuple[np.ndarray, List[Tuple[int, int, int]]]:
    """
    生成六边形紧密排列的晶格节点和三角形单元。

    这种结构模拟中子星地壳中球形核（gnocchi相）的二维截面排列。

    Parameters
    ----------
    lattice_constant : float
        晶格常数 (fm)。
    n_layers : int
        层数（从中心向外）。

    Returns
    -------
    nodes : np.ndarray
        节点坐标。
    elements : list of tuple
        三角形单元连接关系。
    """
    if lattice_constant <= 0.0 or n_layers < 1:
        raise ValueError("Invalid lattice parameters.")

    nodes_list = []
    # 中心点
    nodes_list.append([0.0, 0.0])

    # 六边形环
    for layer in range(1, n_layers + 1):
        for k in range(6 * layer):
            angle = 2.0 * math.pi * k / (6.0 * layer)
            r = layer * lattice_constant
            x = r * math.cos(angle)
            y = r * math.sin(angle)
            nodes_list.append([x, y])

    nodes = np.array(nodes_list)

    # 生成Delaunay-like三角形（简化：以原点为中心的扇形三角化）
    elements = []
    n_total = len(nodes_list)

    # 中心点到第一层
    for k in range(6):
        n0 = 0
        n1 = 1 + k
        n2 = 1 + (k + 1) % 6
        elements.append((n0, n1, n2))

    # 层间三角化
    for layer in range(1, n_layers):
        start_inner = 1 + 3 * layer * (layer - 1)
        n_inner = 6 * layer
        start_outer = start_inner + n_inner
        n_outer = 6 * (layer + 1)

        for k in range(n_inner):
            i1 = start_inner + k
            i2 = start_inner + (k + 1) % n_inner
            # 找到对应的外层点
            ratio = n_outer / n_inner
            o1 = start_outer + int(k * ratio) % n_outer
            o2 = start_outer + int((k + 1) * ratio) % n_outer
            elements.append((i1, i2, o1))
            elements.append((i2, o2, o1))

    return nodes, elements


def compute_crust_shear_modulus(nodes: np.ndarray, elements: List[Tuple[int, int, int]],
                                young_modulus: float = 1.0e35) -> float:
    """
    估算地壳剪切模量。

    公式（简化模型）:
        μ ≈ E / (2(1 + ν)) * (填充因子)

    其中填充因子由核物质占据面积比例决定。
    """
    if len(elements) == 0:
        return 0.0

    total_area = 0.0
    for elem in elements:
        i, j, k = elem
        xi, yi = nodes[i]
        xj, yj = nodes[j]
        xk, yk = nodes[k]
        area = 0.5 * abs((xj - xi) * (yk - yi) - (xk - xi) * (yj - yi))
        total_area += area

    nu = 0.3
    shear = young_modulus / (2.0 * (1.0 + nu))
    return shear
