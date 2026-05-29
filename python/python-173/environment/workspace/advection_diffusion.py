"""
对流-扩散-反应方程模块

融合自:
- 352_fd1d_advection_diffusion_steady: 1D 稳态对流扩散方程

物理模型:
    二维非定常对流-扩散-反应方程:
        ∂u/∂t + v·∇u = ∇·(D ∇u) + R(u) + f(x,y,t)    in Ω × (0,T]
    
    边界条件:
        u = g_D(x,y,t)    on Γ_D × (0,T]
        D ∂u/∂n = g_N(x,y,t)   on Γ_N × (0,T]
    
    初始条件:
        u(x,y,0) = u_0(x,y)    in Ω

其中:
    v = (v_x, v_y) 为对流速度场
    D = D(x,y) 为扩散系数张量（标量或各向异性张量）
    R(u) 为非线性反应项
    f(x,y,t) 为源项

无量纲数:
    Peclet 数: Pe = |v| L / D
    刻画对流与扩散的相对强度:
        Pe << 1: 扩散主导
        Pe >> 1: 对流主导（需要特殊稳定化）

稳定性条件 (显式 Euler):
    扩散限制: Δt <= h^2 / (4 D_max)
    对流限制 (CFL): Δt <= h / |v|_max
"""

import numpy as np
from fem_solver import assemble_fem_system, shape_functions_p1
from quadrature_rules import TriangleQuadrature


def identify_boundary_edges(nodes, triangles, domain_bounds, tol=1e-10):
    """
    识别边界边和边界节点。
    
    边界边只被一个三角形包含。
    
    Parameters
    ----------
    nodes : ndarray, shape (n_nodes, 2)
    triangles : ndarray, shape (n_tri, 3)
    domain_bounds : tuple
        ((xmin, xmax), (ymin, ymax))
    tol : float
    
    Returns
    -------
    boundary_nodes : ndarray
        边界节点索引
    dirichlet_nodes : ndarray
        Dirichlet 边界节点（默认为四条边）
    neumann_edges : list of tuple
        Neumann 边界边
    """
    n_nodes = len(nodes)
    edge_count = {}

    for tri in triangles:
        edges = [
            tuple(sorted([tri[0], tri[1]])),
            tuple(sorted([tri[1], tri[2]])),
            tuple(sorted([tri[2], tri[0]]))
        ]
        for edge in edges:
            edge_count[edge] = edge_count.get(edge, 0) + 1

    boundary_edges = [edge for edge, count in edge_count.items() if count == 1]

    boundary_nodes = set()
    for edge in boundary_edges:
        boundary_nodes.add(edge[0])
        boundary_nodes.add(edge[1])
    boundary_nodes = np.array(sorted(list(boundary_nodes)), dtype=int)

    # 默认 Dirichlet 边界：所有边界节点
    dirichlet_nodes = boundary_nodes.copy()

    # Neumann 边界为空（默认）
    neumann_edges = []

    return boundary_nodes, dirichlet_nodes, neumann_edges


def compute_advection_matrix(nodes, triangles, v_func, quad_degree=3):
    """
    组装对流项的刚度矩阵。
    
    对流项的弱形式贡献:
        ∫_Ω (v·∇u) v dx
    
    对于 Galerkin 方法，对流矩阵:
        C_{ij} = Σ_T ∫_T (v·∇φ_j) φ_i dx
    
    Parameters
    ----------
    nodes : ndarray
    triangles : ndarray
    v_func : callable
        速度场 v_func(x, y) -> (v_x, v_y)
    quad_degree : int
    
    Returns
    -------
    C : ndarray, shape (n_nodes, n_nodes)
        对流矩阵
    """
    n_nodes = len(nodes)
    n_tri = len(triangles)
    quad_rule = TriangleQuadrature(quad_degree)
    C = np.zeros((n_nodes, n_nodes))

    for t in range(n_tri):
        tri = triangles[t]
        p1, p2, p3 = nodes[tri[0]], nodes[tri[1]], nodes[tri[2]]

        J = np.array([
            [p2[0] - p1[0], p3[0] - p1[0]],
            [p2[1] - p1[1], p3[1] - p1[1]]
        ])
        det_J = abs(np.linalg.det(J))

        if det_J < 1e-14:
            continue

        J_inv_T = np.linalg.inv(J).T

        for q in range(quad_rule.n_points):
            xi = quad_rule.points[q, 0]
            eta = quad_rule.points[q, 1]
            w = quad_rule.weights[q]

            phi, grad_phi_ref = shape_functions_p1(xi, eta)
            grad_phi = grad_phi_ref @ J_inv_T.T

            x = p1[0] + (p2[0] - p1[0]) * xi + (p3[0] - p1[0]) * eta
            y = p1[1] + (p2[1] - p1[1]) * xi + (p3[1] - p1[1]) * eta

            v_x, v_y = v_func(x, y)

            for i in range(3):
                for j in range(3):
                    C[tri[i], tri[j]] += w * (
                        (v_x * grad_phi[j, 0] + v_y * grad_phi[j, 1]) * phi[i]
                    ) * det_J

    return C


def compute_reaction_term(nodes, triangles, u_current, R_func, quad_degree=3):
    """
    计算非线性反应项的载荷向量。
    
    R(u) 在右端项中的贡献:
        b_i = Σ_T ∫_T R(u_h) φ_i dx
    
    Parameters
    ----------
    nodes : ndarray
    triangles : ndarray
    u_current : ndarray
        当前解
    R_func : callable
        反应项 R(u, x, y)
    quad_degree : int
    
    Returns
    -------
    b_R : ndarray, shape (n_nodes,)
    """
    n_nodes = len(nodes)
    n_tri = len(triangles)
    quad_rule = TriangleQuadrature(quad_degree)
    b_R = np.zeros(n_nodes)

    for t in range(n_tri):
        tri = triangles[t]
        p1, p2, p3 = nodes[tri[0]], nodes[tri[1]], nodes[tri[2]]

        det_J = abs(
            (p2[0] - p1[0]) * (p3[1] - p1[1]) -
            (p3[0] - p1[0]) * (p2[1] - p1[1])
        )

        if det_J < 1e-14:
            continue

        for q in range(quad_rule.n_points):
            xi = quad_rule.points[q, 0]
            eta = quad_rule.points[q, 1]
            w = quad_rule.weights[q]

            phi, _ = shape_functions_p1(xi, eta)

            x = p1[0] + (p2[0] - p1[0]) * xi + (p3[0] - p1[0]) * eta
            y = p1[1] + (p2[1] - p1[1]) * xi + (p3[1] - p1[1]) * eta

            # 在积分点处插值解
            u_val = np.dot(phi, u_current[tri])
            R_val = R_func(u_val, x, y)

            for i in range(3):
                b_R[tri[i]] += w * R_val * phi[i] * det_J

    return b_R


def compute_mass_matrix_lumped(nodes, triangles):
    """
    计算质量矩阵的 lumped 近似。
    
    Lumped 质量矩阵:
        M_{ii} = Σ_T |T| / 3   (节点 i 属于三角形 T)
        M_{ij} = 0   (i ≠ j)
    
    这对应于集中质量近似，在时间步进中可显式求逆。
    
    Parameters
    ----------
    nodes : ndarray
    triangles : ndarray
    
    Returns
    -------
    M_lumped : ndarray, shape (n_nodes,)
        对角质量矩阵的_diag_元素
    """
    n_nodes = len(nodes)
    M_lumped = np.zeros(n_nodes)

    for tri in triangles:
        p1 = nodes[tri[0]]
        p2 = nodes[tri[1]]
        p3 = nodes[tri[2]]

        area = 0.5 * abs(
            (p2[0] - p1[0]) * (p3[1] - p1[1]) -
            (p3[0] - p1[0]) * (p2[1] - p1[1])
        )

        for i in range(3):
            M_lumped[tri[i]] += area / 3.0

    # 避免零质量
    M_lumped = np.maximum(M_lumped, 1e-14)
    return M_lumped


def compute_cfl_condition(nodes, triangles, v_func, D_func):
    """
    基于网格尺寸和物理参数计算 CFL 稳定性条件。
    
    Parameters
    ----------
    nodes : ndarray
    triangles : ndarray
    v_func : callable
    D_func : callable
    
    Returns
    -------
    dt_max : float
        最大允许时间步长
    h_min : float
        最小单元尺寸
    pe_max : float
        最大局部 Peclet 数
    """
    h_min = np.inf
    pe_max = 0.0

    for tri in triangles:
        p1, p2, p3 = nodes[tri[0]], nodes[tri[1]], nodes[tri[2]]

        # 单元尺寸 (最长边)
        e1 = np.linalg.norm(p2 - p1)
        e2 = np.linalg.norm(p3 - p2)
        e3 = np.linalg.norm(p1 - p3)
        h_T = max(e1, e2, e3)
        h_min = min(h_min, h_T)

        # 中心点处的物理参数
        centroid = (p1 + p2 + p3) / 3.0
        v_x, v_y = v_func(centroid[0], centroid[1])
        v_mag = np.sqrt(v_x ** 2 + v_y ** 2)
        D_val = D_func(centroid[0], centroid[1])

        if D_val > 1e-14:
            pe_T = v_mag * h_T / D_val
            pe_max = max(pe_max, pe_T)

    # CFL 条件
    dt_conv = h_min / (max(v_mag, 1e-10))
    dt_diff = h_min ** 2 / (4.0 * max(D_val, 1e-10))
    dt_max = min(dt_conv, dt_diff)

    return dt_max, h_min, pe_max


def advection_diffusion_reaction_step(
    nodes, triangles,
    u_current, dt,
    D_func, c_func, v_func, R_func, f_func,
    dirichlet_nodes, dirichlet_values,
    M_lumped, scheme='implicit'
):
    """
    执行一个时间步的对流-扩散-反应方程求解。
    
    Parameters
    ----------
    nodes : ndarray
    triangles : ndarray
    u_current : ndarray
        当前时刻的解
    dt : float
        时间步长
    D_func, c_func, v_func, R_func, f_func : callable
        物理参数函数
    dirichlet_nodes : ndarray
    dirichlet_values : ndarray
    M_lumped : ndarray
        Lumped 质量矩阵对角元
    scheme : str
        'implicit' 或 'crank_nicolson'
    
    Returns
    -------
    u_new : ndarray
        下一时刻的解
    """
    n_nodes = len(nodes)

    # 组装扩散+反应矩阵
    A_diff, b_source = assemble_fem_system(
        nodes, triangles,
        D_func, c_func, f_func,
        dirichlet_nodes=None,  # 稍后处理
        quad_degree=3
    )

    # 对流矩阵
    C_adv = compute_advection_matrix(nodes, triangles, v_func, quad_degree=3)

    # 反应项载荷
    b_R = compute_reaction_term(nodes, triangles, u_current, R_func, quad_degree=3)

    # 总右端项
    b_total = b_source + b_R

    # TODO: Hole_2 需修复 - 隐式时间步进的 LHS/RHS 构造
    # 科学知识点: 向后 Euler / Crank-Nicolson 时间离散格式
    # 提示: scheme 参数由调用方传入，需同时支持 'implicit' 和 'crank_nicolson'
    # 注意 LHS 和 RHS 的构造必须与 fem_solver 产生的 A_diff 符号相容
    if scheme == 'implicit':
        lhs = np.eye(n_nodes)  # FIXME: 隐式格式的 LHS 构造缺失
        rhs = np.zeros(n_nodes)  # FIXME: 隐式格式的 RHS 构造缺失
    elif scheme == 'crank_nicolson':
        lhs = np.eye(n_nodes)  # FIXME: CN 格式的 LHS 构造缺失
        rhs = np.zeros(n_nodes)  # FIXME: CN 格式的 RHS 构造缺失
    else:
        raise ValueError(f"advection_diffusion_reaction_step: 未知 scheme={scheme}")

    # 应用 Dirichlet 边界条件
    if dirichlet_nodes is not None:
        for idx, node in enumerate(dirichlet_nodes):
            lhs[node, :] = 0.0
            lhs[node, node] = 1.0
            rhs[node] = dirichlet_values[idx]

            for i in range(n_nodes):
                if i != node and lhs[i, node] != 0:
                    rhs[i] -= lhs[i, node] * dirichlet_values[idx]
                    lhs[i, node] = 0.0

    try:
        u_new = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        u_new = np.linalg.lstsq(lhs, rhs, rcond=None)[0]

    return u_new
