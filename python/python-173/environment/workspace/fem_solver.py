"""
二维三角形有限元求解器模块

融合自:
- 387_fem1d_bvp_quadratic: 1D FEM 刚度矩阵组装、Gauss 数值积分
- 1256_tetrahedron_witherden_rule: 高阶数值积分

数学模型:
    在区域 Ω 上求解椭圆型 PDE:
        -∇·(D ∇u) + c u = f     in Ω
        u = g_D                  on Γ_D
        D ∂u/∂n = g_N            on Γ_N
    
    弱形式:
        找 u ∈ V_g 使得 ∀ v ∈ V_0:
        ∫_Ω D ∇u·∇v dx + ∫_Ω c u v dx = ∫_Ω f v dx + ∫_{Γ_N} g_N v ds
    
    其中 V_g = {v ∈ H^1(Ω): v|_{Γ_D} = g_D}。

离散化:
    采用分片线性 Lagrange 元 (P1) 于三角形网格:
        u_h(x,y) = Σ_j U_j φ_j(x,y)
    
    其中 φ_j 是节点 j 上的帽基函数，满足 φ_j(x_k) = δ_{jk}。

刚度矩阵组装:
    A_{ij} = Σ_T ∫_T [D ∇φ_i·∇φ_j + c φ_i φ_j] dx
    b_i = Σ_T [∫_T f φ_i dx + ∫_{∂T∩Γ_N} g_N φ_i ds]
"""

import numpy as np
from quadrature_rules import TriangleQuadrature


def shape_functions_p1(xi, eta):
    """
    参考三角形上的 P1 形函数及其梯度。
    
    参考三角形顶点:
        v1 = (0, 0),  v2 = (1, 0),  v3 = (0, 1)
    
    形函数:
        φ1(xi, eta) = 1 - xi - eta
        φ2(xi, eta) = xi
        φ3(xi, eta) = eta
    
    梯度 (在参考坐标系下):
        ∇φ1 = (-1, -1)
        ∇φ2 = (1, 0)
        ∇φ3 = (0, 1)
    
    Parameters
    ----------
    xi, eta : float
        参考坐标
    
    Returns
    -------
    phi : ndarray, shape (3,)
        形函数值
    grad_phi_ref : ndarray, shape (3, 2)
        参考坐标系下的梯度
    """
    phi = np.array([1.0 - xi - eta, xi, eta])
    grad_phi_ref = np.array([[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]])
    return phi, grad_phi_ref


def compute_element_stiffness(nodes, triangle, D_func, c_func, quad_rule):
    """
    计算单个三角形单元的局部刚度矩阵和载荷向量。
    
    Parameters
    ----------
    nodes : ndarray, shape (n_nodes, 2)
    triangle : array-like, shape (3,)
        三个顶点索引
    D_func : callable
        扩散系数 D(x, y)
    c_func : callable
        反应系数 c(x, y)
    quad_rule : TriangleQuadrature
        数值积分规则
    
    Returns
    -------
    A_local : ndarray, shape (3, 3)
        局部刚度矩阵
    b_local : ndarray, shape (3,)
        局部载荷向量
    """
    p1 = nodes[triangle[0]]
    p2 = nodes[triangle[1]]
    p3 = nodes[triangle[2]]

    # 仿射变换的 Jacobian
    J = np.array([
        [p2[0] - p1[0], p3[0] - p1[0]],
        [p2[1] - p1[1], p3[1] - p1[1]]
    ])
    det_J = abs(np.linalg.det(J))

    if det_J < 1e-14:
        raise ValueError("compute_element_stiffness: 退化三角形")

    # 物理坐标系下梯度 = J^{-T} * 参考坐标系下梯度
    J_inv_T = np.linalg.inv(J).T

    A_local = np.zeros((3, 3))
    b_local = np.zeros(3)

    for q in range(quad_rule.n_points):
        xi = quad_rule.points[q, 0]
        eta = quad_rule.points[q, 1]
        w = quad_rule.weights[q]

        phi, grad_phi_ref = shape_functions_p1(xi, eta)

        # 映射到物理坐标
        x = p1[0] + (p2[0] - p1[0]) * xi + (p3[0] - p1[0]) * eta
        y = p1[1] + (p2[1] - p1[1]) * xi + (p3[1] - p1[1]) * eta

        # 物理梯度
        grad_phi = grad_phi_ref @ J_inv_T.T

        D_val = D_func(x, y)
        c_val = c_func(x, y)

        # TODO: Hole_1 需修复 - 刚度矩阵单元贡献核心公式
        # 科学知识点: FEM 弱形式中扩散项与反应项的单元刚度矩阵组装
        # A_{ij}^{(T)} = ∫_T [D ∇φ_i·∇φ_j + c φ_i φ_j] dx
        # 提示: 需正确实现 D_val * (∇φ_i·∇φ_j) + c_val * φ_i * φ_j 的数值积分
        for i in range(3):
            for j in range(3):
                A_local[i, j] += w * (
                    # FIXME: 扩散项与反应项的刚度矩阵贡献公式缺失
                    0.0
                ) * det_J

            # 载荷向量 (假设 f=0 在这里，由外部组装)
            b_local[i] += 0.0

    return A_local, b_local


def assemble_fem_system(
    nodes, triangles,
    D_func, c_func, f_func,
    dirichlet_nodes=None, dirichlet_values=None,
    neumann_edges=None, neumann_func=None,
    quad_degree=3
):
    """
    组装 FEM 全局刚度矩阵和载荷向量。
    
    Parameters
    ----------
    nodes : ndarray, shape (n_nodes, 2)
    triangles : ndarray, shape (n_tri, 3)
    D_func : callable
        扩散系数 D(x, y)
    c_func : callable
        反应系数 c(x, y)
    f_func : callable
        右端项 f(x, y)
    dirichlet_nodes : array-like, optional
        Dirichlet 边界节点索引
    dirichlet_values : array-like, optional
        Dirichlet 边界值
    neumann_edges : list of tuple, optional
        Neumann 边界边 [(node_i, node_j), ...]
    neumann_func : callable, optional
        Neumann 边界值 g_N(x, y)
    quad_degree : int
        三角形积分阶数
    
    Returns
    -------
    A : ndarray, shape (n_nodes, n_nodes)
        刚度矩阵
    b : ndarray, shape (n_nodes,)
        载荷向量
    """
    n_nodes = len(nodes)
    n_tri = len(triangles)
    quad_rule = TriangleQuadrature(quad_degree)

    A = np.zeros((n_nodes, n_nodes))
    b = np.zeros(n_nodes)

    for t in range(n_tri):
        tri = triangles[t]
        A_local, _ = compute_element_stiffness(nodes, tri, D_func, c_func, quad_rule)

        # 组装局部到全局
        for i in range(3):
            for j in range(3):
                A[tri[i], tri[j]] += A_local[i, j]

        # 载荷向量 (右端项 f)
        p1 = nodes[tri[0]]
        p2 = nodes[tri[1]]
        p3 = nodes[tri[2]]
        det_J = abs(
            (p2[0] - p1[0]) * (p3[1] - p1[1]) -
            (p3[0] - p1[0]) * (p2[1] - p1[1])
        )

        for q in range(quad_rule.n_points):
            xi = quad_rule.points[q, 0]
            eta = quad_rule.points[q, 1]
            w = quad_rule.weights[q]

            phi, _ = shape_functions_p1(xi, eta)
            x = p1[0] + (p2[0] - p1[0]) * xi + (p3[0] - p1[0]) * eta
            y = p1[1] + (p2[1] - p1[1]) * xi + (p3[1] - p1[1]) * eta

            f_val = f_func(x, y)
            for i in range(3):
                b[tri[i]] += w * f_val * phi[i] * det_J

    # Neumann 边界条件
    if neumann_edges is not None and neumann_func is not None:
        for edge in neumann_edges:
            n1, n2 = edge
            p1 = nodes[n1]
            p2 = nodes[n2]
            edge_len = np.linalg.norm(p2 - p1)

            if edge_len < 1e-14:
                continue

            # 使用中点积分 (1 点 Gauss)
            mid = (p1 + p2) / 2.0
            g_val = neumann_func(mid[0], mid[1])

            # 两节点线性形函数在边上的积分
            b[n1] += g_val * edge_len / 2.0
            b[n2] += g_val * edge_len / 2.0

    # Dirichlet 边界条件 (直接消去法)
    if dirichlet_nodes is not None and dirichlet_values is not None:
        dirichlet_nodes = np.array(dirichlet_nodes, dtype=int)
        dirichlet_values = np.array(dirichlet_values)

        for idx, node in enumerate(dirichlet_nodes):
            val = dirichlet_values[idx]
            # 修改刚度矩阵
            A[node, :] = 0.0
            A[node, node] = 1.0
            b[node] = val

            # 从右端项中消去已知项
            for i in range(n_nodes):
                if i != node and A[i, node] != 0:
                    b[i] -= A[i, node] * val
                    A[i, node] = 0.0

    return A, b


def solve_steady_fem(nodes, triangles, D_func, c_func, f_func,
                     dirichlet_nodes=None, dirichlet_values=None,
                     neumann_edges=None, neumann_func=None,
                     quad_degree=3):
    """
    求解稳态椭圆型 PDE 的 FEM 解。
    
    Parameters
    ----------
    nodes : ndarray
    triangles : ndarray
    D_func : callable
    c_func : callable
    f_func : callable
    dirichlet_nodes : array-like, optional
    dirichlet_values : array-like, optional
    neumann_edges : list, optional
    neumann_func : callable, optional
    quad_degree : int
    
    Returns
    -------
    solution : ndarray, shape (n_nodes,)
        节点处的解值
    A : ndarray
        刚度矩阵（可用于后续分析）
    b : ndarray
        载荷向量
    """
    A, b = assemble_fem_system(
        nodes, triangles,
        D_func, c_func, f_func,
        dirichlet_nodes, dirichlet_values,
        neumann_edges, neumann_func,
        quad_degree
    )

    # 检查矩阵条件数
    cond_est = np.linalg.cond(A)
    if cond_est > 1e14:
        # 矩阵病态，使用正则化
        reg = 1e-10 * np.mean(np.diag(A))
        A = A + reg * np.eye(len(A))

    try:
        solution = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        # 使用最小二乘求解
        solution = np.linalg.lstsq(A, b, rcond=None)[0]

    return solution, A, b
