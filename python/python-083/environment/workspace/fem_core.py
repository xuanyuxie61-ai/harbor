"""
fem_core.py
===========
有限元核心模块
整合自：
  - 1353_triangulation_t3_to_t4：T3→T4 网格升级
  - 756_mesh_vtoe：节点→单元反向映射
  - 893_polynomial：多项式基函数与 graded lexicographic 排序思想

提供二维线弹性有限元的完整求解基础设施：网格处理、高阶单元刚度矩阵
组装、边界条件施加。
"""

import numpy as np
from typing import Tuple, List, Optional


# =============================================================================
# 1. 网格拓扑与反向映射 (mesh_vtoe 思想)
# =============================================================================

def build_vtoe(element_node: np.ndarray, n_nodes: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    由单元→节点映射 (ETOV) 反向构建节点→单元映射 (VTOE)。
    采用 CSR 风格指针数组 + 邻接数组，时间复杂度 O(n_elements * n_local)。

    Parameters
    ----------
    element_node : ndarray, shape (n_elements, n_local)
        每个单元包含的局部节点全局编号。
    n_nodes : int
        节点总数。

    Returns
    -------
    vtoe_ptr : ndarray, shape (n_nodes + 1,)
        CSR 指针数组；vtoe_ptr[i] 到 vtoe_ptr[i+1]-1 为节点 i 所属的单元列表。
    vtoe : ndarray
        节点所属单元编号展平数组。
    """
    n_elements, n_local = element_node.shape
    # 每个节点被多少个单元引用
    degree = np.zeros(n_nodes, dtype=np.int32)
    for e in range(n_elements):
        for k in range(n_local):
            v = element_node[e, k]
            if 0 <= v < n_nodes:
                degree[v] += 1

    vtoe_ptr = np.zeros(n_nodes + 1, dtype=np.int32)
    vtoe_ptr[1:] = np.cumsum(degree)
    # 临时指针用于填充
    temp_ptr = vtoe_ptr[:-1].copy()
    vtoe = np.empty(vtoe_ptr[-1], dtype=np.int32)

    for e in range(n_elements):
        for k in range(n_local):
            v = element_node[e, k]
            if 0 <= v < n_nodes:
                idx = temp_ptr[v]
                vtoe[idx] = e
                temp_ptr[v] += 1

    return vtoe_ptr, vtoe


def triangulation_t3_to_t4(node_xy: np.ndarray, element_node: np.ndarray):
    """
    将 3 节点线性三角形 (T3) 网格升级为 4 节点二次三角形 (T4)。
    在每个三角形重心处插入新节点作为 bubble 节点。

    T4 节点局部编号：0,1,2 为原角节点，3 为重心节点。
    重心坐标：ξ_c = (1/3, 1/3, 1/3)

    Parameters
    ----------
    node_xy : ndarray, shape (n_nodes, 2)
        原始节点坐标。
    element_node : ndarray, shape (n_elements, 3)
        T3 单元连接矩阵。

    Returns
    -------
    node_xy4 : ndarray
        升级后的节点坐标（新节点追加在后）。
    element_node4 : ndarray, shape (n_elements, 4)
        T4 单元连接矩阵。
    """
    n_elements = element_node.shape[0]
    n_nodes_orig = node_xy.shape[0]
    # 新节点：每个单元的重心
    new_xy = np.zeros((n_elements, 2), dtype=np.float64)
    for e in range(n_elements):
        v0, v1, v2 = element_node[e, :3]
        new_xy[e, 0] = (node_xy[v0, 0] + node_xy[v1, 0] + node_xy[v2, 0]) / 3.0
        new_xy[e, 1] = (node_xy[v0, 1] + node_xy[v1, 1] + node_xy[v2, 1]) / 3.0

    node_xy4 = np.vstack([node_xy, new_xy])
    element_node4 = np.zeros((n_elements, 4), dtype=np.int32)
    element_node4[:, :3] = element_node[:, :3]
    element_node4[:, 3] = np.arange(n_nodes_orig, n_nodes_orig + n_elements)
    return node_xy4, element_node4


# =============================================================================
# 2. 多项式基函数与数值积分 (polynomial 思想)
# =============================================================================

def grlex_rank(mono: Tuple[int, ...]) -> int:
    """
    计算 d 维单项式在 graded lexicographic (grlex) 排序下的秩(rank)。
    单项式 x1^a1 * x2^a2 * ... * xd^ad 的 grlex 排序先按总次数 sum(ai) 再按字典序。

    秩计算公式基于组合数学：总次数小于 t 的单项式个数为 C(t + d, d)。
    """
    d = len(mono)
    t = sum(mono)
    # 次数小于 t 的单项式数
    rank = _nchoosek(t + d, d) - 1
    # 同次中按字典序定位
    s = t
    for i in range(d):
        for j in range(mono[i]):
            rank += _nchoosek(s - j + d - i - 1, d - i - 1)
        s -= mono[i]
    return rank


def _nchoosek(n: int, k: int) -> int:
    """二项式系数 C(n, k)，带边界保护。"""
    if k < 0 or k > n or n < 0:
        return 0
    if k == 0 or k == n:
        return 1
    k = min(k, n - k)
    res = 1
    for i in range(1, k + 1):
        res = res * (n - k + i) // i
    return res


def shape_function_t3(xi: np.ndarray, eta: np.ndarray) -> np.ndarray:
    """
    T3 单元形函数在参考坐标 (ξ, η) 下的值。
    参考三角形顶点：(0,0), (1,0), (0,1)。
    N = [1-ξ-η, ξ, η]
    """
    return np.array([1.0 - xi - eta, xi, eta], dtype=np.float64)


def shape_function_t4(xi: np.ndarray, eta: np.ndarray) -> np.ndarray:
    """
    T4 单元形函数（含 bubble 节点）。
    参考三角形顶点：(0,0), (1,0), (0,1)，重心 (1/3, 1/3)。
    采用二次拉格朗日形函数 + bubble：
        N0 = 1 - ξ - η - 9*(1/3-ξ-η)*(1/3-ξ)*(1/3-η)  ... 实际采用标准二次+修正
    这里使用更稳定的表达：
        N0 = 2*(1-ξ-η)*(0.5-ξ-η)
        N1 = 2*ξ*(ξ-0.5)       -> 不对，标准二次应为：
    标准 T6 二次形函数的前 3 个角点 + 3 个中点；T4 简化为角点 + 重心 bubble。
    为简化计算，我们使用：
        N0 = 1 - ξ - η - 27*(1/3)*(1-ξ-η)*(ξ)*(η)  ???

    更严谨地，T4 bubble 形函数：
        N_i (i=0,1,2) = L_i - (1/3) * L_b   where L_i 是线性形函数, L_b = 27*L0*L1*L2
        N_b = 27 * L0 * L1 * L2
    这样保证在重心 (1/3,1/3,1/3) 处 N_b=1, N_i=0。
    """
    L0 = 1.0 - xi - eta
    L1 = xi
    L2 = eta
    Lb = 27.0 * L0 * L1 * L2
    N = np.array([L0 - Lb / 3.0, L1 - Lb / 3.0, L2 - Lb / 3.0, Lb], dtype=np.float64)
    return N


def dshape_t3(xi: np.ndarray, eta: np.ndarray) -> np.ndarray:
    """
    T3 形函数对参考坐标的梯度，shape (3, 2)。
    dN/dξ = [-1, 1, 0]
    dN/dη = [-1, 0, 1]
    """
    return np.array([[-1.0, 1.0, 0.0],
                     [-1.0, 0.0, 1.0]], dtype=np.float64).T


def dshape_t4(xi: np.ndarray, eta: np.ndarray) -> np.ndarray:
    """
    T4 形函数对参考坐标的梯度，shape (4, 2)。
    """
    L0 = 1.0 - xi - eta
    L1 = xi
    L2 = eta
    # dLb/dxi = 27*(L0*L1)'_xi considering L0=1-xi-eta, L1=xi
    dLb_dxi = 27.0 * ((-1.0) * L1 * L2 + L0 * 1.0 * L2 + L0 * L1 * 0.0)
    dLb_deta = 27.0 * ((-1.0) * L1 * L2 + L0 * 0.0 * L2 + L0 * L1 * 1.0)
    dN = np.zeros((4, 2), dtype=np.float64)
    dN[0, 0] = -1.0 - dLb_dxi / 3.0
    dN[0, 1] = -1.0 - dLb_deta / 3.0
    dN[1, 0] = 1.0 - dLb_dxi / 3.0
    dN[1, 1] = 0.0 - dLb_deta / 3.0
    dN[2, 0] = 0.0 - dLb_dxi / 3.0
    dN[2, 1] = 1.0 - dLb_deta / 3.0
    dN[3, 0] = dLb_dxi
    dN[3, 1] = dLb_deta
    return dN


# =============================================================================
# 3. 高斯积分规则
# =============================================================================

def gauss_triangle(order: int = 3):
    """
    三角形高斯积分点与权重（参考坐标下）。
    支持 1/3/4/7 点规则。
    """
    if order == 1:
        pts = np.array([[1.0/3.0, 1.0/3.0]], dtype=np.float64)
        wts = np.array([1.0], dtype=np.float64)
    elif order == 3:
        pts = np.array([[2.0/3.0, 1.0/6.0],
                        [1.0/6.0, 2.0/3.0],
                        [1.0/6.0, 1.0/6.0]], dtype=np.float64)
        wts = np.array([1.0/3.0, 1.0/3.0, 1.0/3.0], dtype=np.float64)
    elif order == 4:
        a = 0.816847572980459
        b = 0.091576213509771
        c = 0.108103018168070
        d = 0.445948490915965
        w1 = 0.109951743655322
        w2 = 0.223381589678011
        pts = np.array([[a, b], [b, a], [b, b],
                        [d, c], [c, d], [d, d]], dtype=np.float64)
        wts = np.array([w1, w1, w1, w2, w2, w2], dtype=np.float64)
    elif order == 7:
        # 7点 Hammer 规则
        a1, a2 = 0.101286507323456, 0.797426985353087
        b1, b2 = 0.470142064105115, 0.059715871789770
        c = 0.333333333333333
        w1 = 0.125939180544827
        w2 = 0.132394152788506
        w3 = 0.225000000000000
        pts = np.array([[a1, a2], [a2, a1], [a1, a1],
                        [b1, b2], [b2, b1], [b1, b1],
                        [c, c]], dtype=np.float64)
        wts = np.array([w1, w1, w1, w2, w2, w2, w3], dtype=np.float64)
    else:
        raise ValueError(f"Unsupported Gauss order {order}")
    # 参考三角形面积为 1/2，权重需乘以该面积
    wts = wts * 0.5
    return pts, wts


# =============================================================================
# 4. 弹性力学基础：D 矩阵
# =============================================================================

def elastic_d_matrix(E: float, nu: float, plane_stress: bool = True) -> np.ndarray:
    """
    构建平面应力/平面应变问题的弹性矩阵 D (3x3)。

    平面应力：
        D = E/(1-ν²) * [[1, ν, 0],
                         [ν, 1, 0],
                         [0, 0, (1-ν)/2]]
    平面应变：
        D = E/((1+ν)(1-2ν)) * [[1-ν, ν,   0],
                                [ν,   1-ν, 0],
                                [0,   0,   (1-2ν)/2]]
    """
    D = np.zeros((3, 3), dtype=np.float64)
    # TODO Hole 1: 实现平面应力/平面应变的弹性矩阵 D (3x3)
    # 平面应力: D = E/(1-ν²) * [[1, ν, 0], [ν, 1, 0], [0, 0, (1-ν)/2]]
    # 平面应变: D = E/((1+ν)(1-2ν)) * [[1-ν, ν, 0], [ν, 1-ν, 0], [0, 0, (1-2ν)/2]]
    raise NotImplementedError("Hole 1: elastic_d_matrix 未实现")
    return D


# =============================================================================
# 5. 单元刚度矩阵组装
# =============================================================================

def compute_element_stiffness_t3(node_xy_e: np.ndarray, E: float, nu: float,
                                  plane_stress: bool = True) -> np.ndarray:
    """
    计算 T3 单元的刚度矩阵 (6x6)。

    参数
    ----
    node_xy_e : ndarray, shape (3, 2)
        单元三个节点的实际坐标。
    """
    D = elastic_d_matrix(E, nu, plane_stress)
    pts, wts = gauss_triangle(order=3)
    Ke = np.zeros((6, 6), dtype=np.float64)
    # T3 的 Jacobian 是常数
    dN_dxi = dshape_t3(0.0, 0.0)  # (3, 2)
    J = dN_dxi.T @ node_xy_e      # (2, 2)
    detJ = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]
    if abs(detJ) < 1e-14:
        raise ValueError("Degenerate triangle element detected (detJ ~ 0).")
    invJ = np.linalg.inv(J)
    dN_dx = dN_dxi @ invJ         # (3, 2)

    for q in range(len(wts)):
        B = np.zeros((3, 6), dtype=np.float64)
        for i in range(3):
            B[0, 2*i]   = dN_dx[i, 0]
            B[1, 2*i+1] = dN_dx[i, 1]
            B[2, 2*i]   = dN_dx[i, 1]
            B[2, 2*i+1] = dN_dx[i, 0]
        Ke += B.T @ D @ B * wts[q] * abs(detJ)
    return Ke


def compute_element_stiffness_t4(node_xy_e: np.ndarray, E: float, nu: float,
                                  plane_stress: bool = True) -> np.ndarray:
    """
    计算 T4 单元的刚度矩阵 (8x8)。
    """
    D = elastic_d_matrix(E, nu, plane_stress)
    pts, wts = gauss_triangle(order=4)
    Ke = np.zeros((8, 8), dtype=np.float64)
    for q in range(len(wts)):
        xi, eta = pts[q, 0], pts[q, 1]
        dN_dxi = dshape_t4(xi, eta)   # (4, 2)
        J = dN_dxi.T @ node_xy_e       # (2, 2)
        detJ = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]
        if abs(detJ) < 1e-14:
            raise ValueError("Degenerate T4 element detected.")
        invJ = np.linalg.inv(J)
        dN_dx = dN_dxi @ invJ
        B = np.zeros((3, 8), dtype=np.float64)
        for i in range(4):
            B[0, 2*i]   = dN_dx[i, 0]
            B[1, 2*i+1] = dN_dx[i, 1]
            B[2, 2*i]   = dN_dx[i, 1]
            B[2, 2*i+1] = dN_dx[i, 0]
        Ke += B.T @ D @ B * wts[q] * abs(detJ)
    return Ke


# =============================================================================
# 6. 全局刚度矩阵稀疏组装 (COO 格式，与 r8st 思想一致)
# =============================================================================

def assemble_global_stiffness(node_xy: np.ndarray, element_node: np.ndarray,
                               E: float, nu: float, plane_stress: bool = True,
                               element_type: str = "T3") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    组装全局刚度矩阵的 COO 格式三元组 (row, col, val)。

    Returns
    -------
    rows, cols, vals : ndarray
        非零元的行索引、列索引、值。
    """
    n_nodes = node_xy.shape[0]
    n_elements = element_node.shape[0]
    n_local = element_node.shape[1]
    dof_per_node = 2
    n_edof = n_local * dof_per_node

    rows = []
    cols = []
    vals = []

    for e in range(n_elements):
        enodes = element_node[e, :]
        node_xy_e = node_xy[enodes, :]
        if element_type.upper() == "T3":
            Ke = compute_element_stiffness_t3(node_xy_e, E, nu, plane_stress)
        elif element_type.upper() == "T4":
            Ke = compute_element_stiffness_t4(node_xy_e, E, nu, plane_stress)
        else:
            raise ValueError(f"Unknown element type: {element_type}")

        edof = np.zeros(n_edof, dtype=np.int32)
        for i in range(n_local):
            edof[2*i]   = enodes[i] * 2
            edof[2*i+1] = enodes[i] * 2 + 1

        for i in range(n_edof):
            for j in range(n_edof):
                rows.append(edof[i])
                cols.append(edof[j])
                vals.append(Ke[i, j])

    return np.array(rows, dtype=np.int32), np.array(cols, dtype=np.int32), np.array(vals, dtype=np.float64)


# =============================================================================
# 7. 边界条件与求解
# =============================================================================

def apply_dirichlet_bcs(K_dense: np.ndarray, F: np.ndarray, bc_nodes: np.ndarray,
                         bc_values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    直接法施加 Dirichlet 边界条件（大数法/置1法）。
    这里采用置1法，保持对称性。

    参数
    ----
    K_dense : ndarray, shape (n_dof, n_dof)
        稠密全局刚度矩阵。
    F : ndarray, shape (n_dof,)
        载荷向量。
    bc_nodes : ndarray
        受约束自由度索引。
    bc_values : ndarray
        对应的位移值。
    """
    K_mod = K_dense.copy()
    F_mod = F.copy()
    penalty = 1e12 * np.max(np.abs(K_dense))
    if penalty == 0.0:
        penalty = 1e12

    for i, dof in enumerate(bc_nodes):
        val = bc_values[i]
        K_mod[dof, :] = 0.0
        K_mod[:, dof] = 0.0
        K_mod[dof, dof] = penalty
        F_mod[dof] = val * penalty
    return K_mod, F_mod


def solve_fem_system(node_xy: np.ndarray, element_node: np.ndarray,
                      E: float, nu: float, F_ext: np.ndarray,
                      bc_nodes: np.ndarray, bc_values: np.ndarray,
                      plane_stress: bool = True, element_type: str = "T3") -> np.ndarray:
    """
    完整 FEM 求解流程：组装 → 施加 BC → 求解。

    返回位移向量 U。
    """
    n_dof = node_xy.shape[0] * 2
    rows, cols, vals = assemble_global_stiffness(
        node_xy, element_node, E, nu, plane_stress, element_type)

    K = np.zeros((n_dof, n_dof), dtype=np.float64)
    for r, c, v in zip(rows, cols, vals):
        K[r, c] += v

    K_mod, F_mod = apply_dirichlet_bcs(K, F_ext, bc_nodes, bc_values)

    # 检查矩阵条件数，给出数值稳定性提示
    cond_est = np.linalg.cond(K_mod)
    if cond_est > 1e16:
        # 使用正则化
        K_mod += np.eye(n_dof) * 1e-8 * np.max(np.abs(K_mod))

    U = np.linalg.solve(K_mod, F_mod)
    return U


# =============================================================================
# 8. 后处理：应力和应变
# =============================================================================

def compute_element_stress(node_xy: np.ndarray, element_node: np.ndarray,
                            U: np.ndarray, E: float, nu: float,
                            plane_stress: bool = True, element_type: str = "T3") -> np.ndarray:
    """
    计算每个单元中心处的应力向量 [σ_xx, σ_yy, τ_xy]。
    """
    n_elements = element_node.shape[0]
    stress = np.zeros((n_elements, 3), dtype=np.float64)
    D = elastic_d_matrix(E, nu, plane_stress)

    for e in range(n_elements):
        enodes = element_node[e, :]
        node_xy_e = node_xy[enodes, :]
        n_local = len(enodes)
        if element_type.upper() == "T3":
            dN_dxi = dshape_t3(0.0, 0.0)
        else:
            dN_dxi = dshape_t4(1.0/3.0, 1.0/3.0)
        J = dN_dxi.T @ node_xy_e
        detJ = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]
        if abs(detJ) < 1e-14:
            continue
        invJ = np.linalg.inv(J)
        dN_dx = dN_dxi @ invJ
        B = np.zeros((3, n_local * 2), dtype=np.float64)
        for i in range(n_local):
            B[0, 2*i]   = dN_dx[i, 0]
            B[1, 2*i+1] = dN_dx[i, 1]
            B[2, 2*i]   = dN_dx[i, 1]
            B[2, 2*i+1] = dN_dx[i, 0]
        edof = np.zeros(n_local * 2, dtype=np.int32)
        for i in range(n_local):
            edof[2*i]   = enodes[i] * 2
            edof[2*i+1] = enodes[i] * 2 + 1
        u_e = U[edof]
        eps = B @ u_e
        sigma = D @ eps
        stress[e, :] = sigma
    return stress


# =============================================================================
# 9. 网格生成：规则矩形域三角形化
# =============================================================================

def generate_rectangular_mesh(lx: float, ly: float, nx: int, ny: int):
    """
    生成规则矩形域的 T3 三角形网格。
    域范围 [0, lx] × [0, ly]，均匀划分为 nx × ny 个矩形，每个矩形再对角剖分为2个三角形。

    Returns
    -------
    node_xy : ndarray, shape ((nx+1)*(ny+1), 2)
    element_node : ndarray, shape (2*nx*ny, 3)
    """
    x = np.linspace(0.0, lx, nx + 1)
    y = np.linspace(0.0, ly, ny + 1)
    X, Y = np.meshgrid(x, y)
    node_xy = np.column_stack([X.ravel(), Y.ravel()])

    n_nodes_row = nx + 1
    elements = []
    for j in range(ny):
        for i in range(nx):
            n0 = j * n_nodes_row + i
            n1 = n0 + 1
            n2 = n0 + n_nodes_row
            n3 = n2 + 1
            # 三角形1: n0-n1-n2
            elements.append([n0, n1, n2])
            # 三角形2: n1-n3-n2
            elements.append([n1, n3, n2])
    element_node = np.array(elements, dtype=np.int32)
    return node_xy, element_node
