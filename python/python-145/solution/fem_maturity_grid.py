"""
fem_maturity_grid.py
====================
博士级有限元空间离散：期限结构 (t, T) 域的三角网格与二次基函数

本模块实现了二维有限元方法中的核心几何与代数操作，用于将利率期限结构
的随机偏微分方程在空间（剩余期限 T）方向离散化：

  1. 二维非结构化三角网格生成（基于 Delaunay 思想）
  2. 三角形面积与质心计算
  3. 二次三角形元（T6）的基函数与导数
  4. 泊松方程与热方程的 FEM 求解器（稀疏格式）
  5. 边界条件处理（Dirichlet）

数学理论
--------
考虑前向利率 f(t, T) 在 (t, T) 平面上的演化，空间离散化将 T 域剖分为
三角形网格 {K_e}，在每个单元上采用二次拉格朗日基函数：

    φ_i(x, y) 在节点 i 处为 1，在其他节点处为 0

刚度矩阵组装:
    A_{ij} = Σ_e ∫_{K_e} [ ∇φ_i · ∇φ_j + k(x,y) φ_i φ_j ] dx dy

质量矩阵组装:
    M_{ij} = Σ_e ∫_{K_e} φ_i φ_j dx dy

右端项:
    F_i = Σ_e ∫_{K_e} f(x,y) φ_i dx dy

采用三点 Gauss 积分公式保证二次精度。
"""

import numpy as np
from scipy import sparse as sp
from scipy.sparse.linalg import spsolve


def generate_rectangular_grid(nx, ny, xl=0.0, xr=1.0, yb=0.0, yt=1.0):
    """
    在矩形域 [xl, xr] × [yb, yt] 上生成规则节点坐标与 T6 二次三角形单元。

    节点排列:
        底层网格为 nx × ny 的角点，中边节点插入后总节点数为 (2*nx-1)*(2*ny-1)。

    Parameters
    ----------
    nx, ny : int
        x 与 y 方向的角点数。
    xl, xr : float
        x 方向边界。
    yb, yt : float
        y 方向边界。

    Returns
    -------
    node_xy : np.ndarray, shape (node_num, 2)
        节点坐标。
    element_node : np.ndarray, shape (element_num, 6)
        每个单元的 6 个节点编号（0-based）。
    """
    if nx < 2 or ny < 2:
        raise ValueError("generate_rectangular_grid: nx, ny 必须至少为 2")

    node_num = (2 * nx - 1) * (2 * ny - 1)
    element_num = (nx - 1) * (ny - 1) * 2

    node_xy = np.zeros((node_num, 2), dtype=float)

    # 生成节点坐标（角点 + 中边 + 中心）
    dx = (xr - xl) / (nx - 1)
    dy = (yt - yb) / (ny - 1)

    node = 0
    for j in range(2 * ny - 1):
        for i in range(2 * nx - 1):
            if j % 2 == 0 and i % 2 == 0:
                # 角点
                x = xl + (i // 2) * dx
                y = yb + (j // 2) * dy
            elif j % 2 == 0 and i % 2 == 1:
                # 水平边中点
                x = xl + (i // 2) * dx + dx / 2
                y = yb + (j // 2) * dy
            elif j % 2 == 1 and i % 2 == 0:
                # 垂直边中点
                x = xl + (i // 2) * dx
                y = yb + (j // 2) * dy + dy / 2
            else:
                # 单元中心
                x = xl + (i // 2) * dx + dx / 2
                y = yb + (j // 2) * dy + dy / 2
            node_xy[node, 0] = x
            node_xy[node, 1] = y
            node += 1

    # 生成 T6 单元
    element_node = np.zeros((element_num, 6), dtype=int)
    element = 0
    row_nodes = 2 * nx - 1

    for j in range(ny - 1):
        for i in range(nx - 1):
            sw = j * 2 * row_nodes + 2 * i
            w = sw + 1
            nw = sw + 2
            s = sw + row_nodes
            c = s + 1
            n = s + 2
            se = s + row_nodes
            e = se + 1
            ne = se + 2

            # 左下三角形
            element_node[element, :] = [sw, se, nw, s, c, w]
            element += 1
            # 右上三角形
            element_node[element, :] = [ne, nw, se, n, c, e]
            element += 1

    return node_xy, element_node


def triangle_area(p1, p2, p3):
    """
    计算三角形有向面积（假设逆时针节点顺序为正）。

    公式:
        A = 0.5 * | (x2 - x1)(y3 - y1) - (x3 - x1)(y2 - y1) |

    Parameters
    ----------
    p1, p2, p3 : np.ndarray, shape (2,)
        顶点坐标。

    Returns
    -------
    float
        三角形面积（绝对值）。
    """
    p1, p2, p3 = np.asarray(p1), np.asarray(p2), np.asarray(p3)
    return 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))


def reference_to_physical_t3(t3, quad_xy):
    """
    将参考三角形上的点映射到物理三角形。

    参考三角形: (0,0), (1,0), (0,1)
    映射: x = x1 + (x2-x1)*r + (x3-x1)*s
         y = y1 + (y2-y1)*r + (y3-y1)*s

    Parameters
    ----------
    t3 : np.ndarray, shape (3, 2)
        物理三角形三个顶点。
    quad_xy : np.ndarray, shape (n_quad, 2)
        参考坐标 (r, s)。

    Returns
    -------
    np.ndarray, shape (n_quad, 2)
        物理坐标。
    """
    t3 = np.asarray(t3, dtype=float)
    quad_xy = np.asarray(quad_xy, dtype=float)
    nq = quad_xy.shape[0]
    xy = np.zeros((nq, 2), dtype=float)
    for q in range(nq):
        r, s = quad_xy[q, 0], quad_xy[q, 1]
        xy[q, 0] = t3[0, 0] + (t3[1, 0] - t3[0, 0]) * r + (t3[2, 0] - t3[0, 0]) * s
        xy[q, 1] = t3[0, 1] + (t3[1, 1] - t3[0, 1]) * r + (t3[2, 1] - t3[0, 1]) * s
    return xy


def basis_11_t6(t6, i, p):
    """
    计算二次三角形元 T6 的第 i 个基函数及其偏导数。

    编号约定:
        1: 顶点 1, 2: 顶点 2, 3: 顶点 3
        4: 边 1-2 中点, 5: 边 2-3 中点, 6: 边 3-1 中点

    参数化方法:
        通过面积坐标 (L1, L2, L3) 构造二次基:
        N1 = L1(2L1-1), N2 = L2(2L2-1), N3 = L3(2L3-1)
        N4 = 4 L1 L2, N5 = 4 L2 L3, N6 = 4 L3 L1

    Parameters
    ----------
    t6 : np.ndarray, shape (6, 2)
        六个节点坐标（顶点 + 中边）。
    i : int
        基函数局部编号，1-based（1..6）。
    p : np.ndarray, shape (2,)
        求值点 (x, y)。

    Returns
    -------
    bi : float
        基函数值。
    dbidx : float
        x 方向偏导。
    dbidy : float
        y 方向偏导。
    """
    if i < 1 or i > 6:
        raise ValueError("basis_11_t6: i 必须在 1..6 之间")
    t6 = np.asarray(t6, dtype=float)
    p = np.asarray(p, dtype=float)

    # 使用顶点计算面积坐标
    x1, y1 = t6[0, 0], t6[0, 1]
    x2, y2 = t6[1, 0], t6[1, 1]
    x3, y3 = t6[2, 0], t6[2, 1]

    area2 = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
    if abs(area2) < 1e-14:
        raise ValueError("basis_11_t6: 三角形退化")

    # 面积坐标
    L1 = ((x2 - p[0]) * (y3 - p[1]) - (x3 - p[0]) * (y2 - p[1])) / area2
    L2 = ((x3 - p[0]) * (y1 - p[1]) - (x1 - p[0]) * (y3 - p[1])) / area2
    L3 = 1.0 - L1 - L2

    # 面积坐标梯度
    dL1dx = (y2 - y3) / area2
    dL1dy = (x3 - x2) / area2
    dL2dx = (y3 - y1) / area2
    dL2dy = (x1 - x3) / area2
    dL3dx = -dL1dx - dL2dx
    dL3dy = -dL1dy - dL2dy

    if i == 1:
        bi = L1 * (2.0 * L1 - 1.0)
        dbidx = dL1dx * (4.0 * L1 - 1.0)
        dbidy = dL1dy * (4.0 * L1 - 1.0)
    elif i == 2:
        bi = L2 * (2.0 * L2 - 1.0)
        dbidx = dL2dx * (4.0 * L2 - 1.0)
        dbidy = dL2dy * (4.0 * L2 - 1.0)
    elif i == 3:
        bi = L3 * (2.0 * L3 - 1.0)
        dbidx = dL3dx * (4.0 * L3 - 1.0)
        dbidy = dL3dy * (4.0 * L3 - 1.0)
    elif i == 4:
        bi = 4.0 * L1 * L2
        dbidx = 4.0 * (dL1dx * L2 + L1 * dL2dx)
        dbidy = 4.0 * (dL1dy * L2 + L1 * dL2dy)
    elif i == 5:
        bi = 4.0 * L2 * L3
        dbidx = 4.0 * (dL2dx * L3 + L2 * dL3dx)
        dbidy = 4.0 * (dL2dy * L3 + L2 * dL3dy)
    else:  # i == 6
        bi = 4.0 * L3 * L1
        dbidx = 4.0 * (dL3dx * L1 + L3 * dL1dx)
        dbidy = 4.0 * (dL3dy * L1 + L3 * dL1dy)

    return bi, dbidx, dbidy


def get_quad_rule_triangle(nq=3):
    """
    返回三角形上的 Gauss 积分点与权重（参考三角形）。

    Parameters
    ----------
    nq : int
        积分点数，支持 1 或 3。

    Returns
    -------
    w : np.ndarray
        权重。
    xy : np.ndarray, shape (nq, 2)
        积分点坐标 (r, s)。
    """
    if nq == 1:
        w = np.array([0.5])
        xy = np.array([[1.0 / 3.0, 1.0 / 3.0]])
    elif nq == 3:
        w = np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
        xy = np.array([
            [0.5, 0.0],
            [0.5, 0.5],
            [0.0, 0.5]
        ])
    else:
        raise ValueError("get_quad_rule_triangle: 仅支持 nq=1 或 3")
    return w, xy


def assemble_fem_matrices(node_xy, element_node, element_order=6,
                          k_coef=None, nq=3):
    """
    组装 FEM 刚度矩阵 A 与质量矩阵 M。

    公式:
        A_{ij} = Σ_e Σ_q w_q |J_e| [ ∇φ_i · ∇φ_j + k(ξ_q) φ_i φ_j ]
        M_{ij} = Σ_e Σ_q w_q |J_e| φ_i φ_j

    Parameters
    ----------
    node_xy : np.ndarray, shape (node_num, 2)
        节点坐标。
    element_node : np.ndarray, shape (element_num, element_order)
        单元节点编号。
    element_order : int
        单元阶数，默认 6（T6 二次元）。
    k_coef : callable or None
        反应系数函数 k(x, y)；None 时取 0。
    nq : int
        积分点数。

    Returns
    -------
    A : scipy.sparse.csr_matrix
        刚度矩阵。
    M : scipy.sparse.csr_matrix
        质量矩阵。
    """
    node_num = node_xy.shape[0]
    element_num = element_node.shape[0]

    if k_coef is None:
        def k_coef_default(x, y):
            return 0.0
        k_coef = k_coef_default

    quad_w, quad_xy = get_quad_rule_triangle(nq)

    row_A, col_A, data_A = [], [], []
    row_M, col_M, data_M = [], [], []

    for e in range(element_num):
        nodes_e = element_node[e, :element_order]
        t3 = node_xy[nodes_e[:3], :]
        area = triangle_area(t3[0], t3[1], t3[2])
        if area < 1e-14:
            continue

        # 映射积分点到物理单元
        xy_phys = reference_to_physical_t3(t3, quad_xy)

        for q in range(nq):
            xq, yq = xy_phys[q, 0], xy_phys[q, 1]
            w = area * quad_w[q]
            kq = k_coef(xq, yq)

            for test in range(element_order):
                i = nodes_e[test]
                bi, dbidx, dbidy = basis_11_t6(node_xy[nodes_e, :], test + 1, xy_phys[q, :])

                for basis in range(element_order):
                    j = nodes_e[basis]
                    bj, dbjdx, dbjdy = basis_11_t6(node_xy[nodes_e, :], basis + 1, xy_phys[q, :])

                    aij = dbidx * dbjdx + dbidy * dbjdy + kq * bi * bj
                    mij = bi * bj

                    row_A.append(i)
                    col_A.append(j)
                    data_A.append(w * aij)

                    row_M.append(i)
                    col_M.append(j)
                    data_M.append(w * mij)

    A = sp.coo_matrix((data_A, (row_A, col_A)), shape=(node_num, node_num)).tocsr()
    M = sp.coo_matrix((data_M, (row_M, col_M)), shape=(node_num, node_num)).tocsr()
    return A, M


def apply_dirichlet_bc(A, rhs, node_xy, boundary_func):
    """
    施加 Dirichlet 边界条件。

    策略: 对边界节点，将矩阵对应行置为 e_i，右端项置为边界值。

    Parameters
    ----------
    A : scipy.sparse.csr_matrix
        刚度矩阵。
    rhs : np.ndarray
        右端向量。
    node_xy : np.ndarray, shape (node_num, 2)
        节点坐标。
    boundary_func : callable
        边界值函数 boundary_func(x, y) -> float。

    Returns
    -------
    A : scipy.sparse.csr_matrix
        修改后的矩阵。
    rhs : np.ndarray
        修改后的右端向量。
    bc_indices : np.ndarray
        边界节点索引。
    bc_values : np.ndarray
        边界值。
    """
    node_num = node_xy.shape[0]
    # 识别边界节点：位于矩形边界的节点
    xl, xr = node_xy[:, 0].min(), node_xy[:, 0].max()
    yb, yt = node_xy[:, 1].min(), node_xy[:, 1].max()
    tol = 1e-10 * max(xr - xl, yt - yb)

    bc_indices = []
    bc_values = []
    for i in range(node_num):
        x, y = node_xy[i, 0], node_xy[i, 1]
        if (abs(x - xl) < tol or abs(x - xr) < tol or
                abs(y - yb) < tol or abs(y - yt) < tol):
            bc_indices.append(i)
            bc_values.append(boundary_func(x, y))

    bc_indices = np.array(bc_indices, dtype=int)
    bc_values = np.array(bc_values, dtype=float)

    # 修改矩阵和右端项
    A = A.tolil()
    for idx, val in zip(bc_indices, bc_values):
        A[idx, :] = 0.0
        A[idx, idx] = 1.0
        rhs[idx] = val
    A = A.tocsr()

    return A, rhs, bc_indices, bc_values


def solve_poisson_fem(node_xy, element_node, rhs_func, boundary_func,
                      element_order=6, nq=3):
    """
    求解二维泊松方程 -Δu = f，带 Dirichlet 边界条件。

    方程:
        -∇² u(x, y) = f(x, y)  在 Ω 内
        u(x, y) = g(x, y)       在 ∂Ω 上

    Parameters
    ----------
    node_xy : np.ndarray
        节点坐标。
    element_node : np.ndarray
        单元节点。
    rhs_func : callable
        右端项函数 f(x, y)。
    boundary_func : callable
        边界值函数 g(x, y)。
    element_order : int
        单元阶数。
    nq : int
        积分点数。

    Returns
    -------
    u : np.ndarray
        有限元解。
    """
    node_num = node_xy.shape[0]
    A, M = assemble_fem_matrices(node_xy, element_node, element_order, nq=nq)

    # 组装右端项: F_i = ∫ f φ_i
    element_num = element_node.shape[0]
    quad_w, quad_xy = get_quad_rule_triangle(nq)
    rhs = np.zeros(node_num, dtype=float)

    for e in range(element_num):
        nodes_e = element_node[e, :element_order]
        t3 = node_xy[nodes_e[:3], :]
        area = triangle_area(t3[0], t3[1], t3[2])
        if area < 1e-14:
            continue
        xy_phys = reference_to_physical_t3(t3, quad_xy)
        for q in range(nq):
            w = area * quad_w[q]
            for test in range(element_order):
                i = nodes_e[test]
                bi, _, _ = basis_11_t6(node_xy[nodes_e, :], test + 1, xy_phys[q, :])
                rhs[i] += w * rhs_func(xy_phys[q, 0], xy_phys[q, 1]) * bi

    A, rhs, bc_indices, bc_values = apply_dirichlet_bc(A, rhs, node_xy, boundary_func)
    u = spsolve(A, rhs)
    return u


def solve_heat_fem(node_xy, element_node, u_init, dt, n_steps,
                   rhs_func, boundary_func, element_order=6, nq=3):
    """
    使用后向 Euler 求解二维热方程 u_t - Δu = f。

    方程:
        ∂u/∂t - ∇²u = f(x, y, t)  在 Ω × (0, T] 内
        u(x, y, t) = g(x, y, t)   在 ∂Ω × (0, T] 上
        u(x, y, 0) = u0(x, y)     在 Ω 内

    Parameters
    ----------
    node_xy : np.ndarray
        节点坐标。
    element_node : np.ndarray
        单元节点。
    u_init : np.ndarray
        初始条件。
    dt : float
        时间步长。
    n_steps : int
        时间步数。
    rhs_func : callable
        右端项 f(x, y, t)。
    boundary_func : callable
        边界值 g(x, y, t)。

    Returns
    -------
    u : np.ndarray
        最终时刻的解。
    u_history : list of np.ndarray
        每步的解（可选输出，用于时序分析）。
    """
    node_num = node_xy.shape[0]
    A, M = assemble_fem_matrices(node_xy, element_node, element_order, nq=nq)

    u = np.asarray(u_init, dtype=float).copy()
    u_history = [u.copy()]

    for step in range(n_steps):
        t = (step + 1) * dt

        # 组装当前时刻右端项
        quad_w, quad_xy = get_quad_rule_triangle(nq)
        element_num = element_node.shape[0]
        f_rhs = np.zeros(node_num, dtype=float)
        for e in range(element_num):
            nodes_e = element_node[e, :element_order]
            t3 = node_xy[nodes_e[:3], :]
            area = triangle_area(t3[0], t3[1], t3[2])
            if area < 1e-14:
                continue
            xy_phys = reference_to_physical_t3(t3, quad_xy)
            for q in range(nq):
                w = area * quad_w[q]
                for test in range(element_order):
                    i = nodes_e[test]
                    bi, _, _ = basis_11_t6(node_xy[nodes_e, :], test + 1, xy_phys[q, :])
                    f_rhs[i] += w * rhs_func(xy_phys[q, 0], xy_phys[q, 1], t) * bi

        # 后向 Euler: (M + dt A) u_new = M u_old + dt f_rhs
        lhs = M + dt * A
        rhs = M @ u + dt * f_rhs

        # 边界条件
        lhs, rhs, bc_indices, bc_values = apply_dirichlet_bc(lhs, rhs, node_xy,
                                                              lambda x, y: boundary_func(x, y, t))
        u = spsolve(lhs, rhs)
        u_history.append(u.copy())

    return u, u_history
