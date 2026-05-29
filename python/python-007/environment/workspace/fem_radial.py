"""
一维有限元径向求解模块
整合自：388_fem1d_display（1D FEM基函数构造与函数求值）

在吸积盘模拟中用于：
  1. 求解径向方向上的盘结构方程（表面密度、温度分布）
  2. 构造高阶拉格朗日基函数进行高精度单元内插值
  3. 计算径向角动量输运方程
"""
import numpy as np


def lagrange_basis_1d(x_nodes, x_query):
    """
    计算1D拉格朗日基函数在查询点的值。

    对于 n 个节点 {x_i}，第 k 个基函数为：
        phi_k(x) = prod_{j!=k} (x - x_j) / (x_k - x_j)

    满足 Kronecker delta 性质：phi_k(x_i) = delta_{ki}

    参数:
        x_nodes: (n,) 单元节点坐标
        x_query: 查询点（标量或数组）

    返回:
        phi: (n,) 或 (n, n_query) 基函数值
    """
    x_nodes = np.asarray(x_nodes, dtype=np.float64)
    n = len(x_nodes)

    scalar_input = np.isscalar(x_query)
    xq = np.atleast_1d(x_query).astype(np.float64)

    phi = np.zeros((n, len(xq)), dtype=np.float64)

    for k in range(n):
        # 计算 phi_k(x) = prod_{j!=k} (x - x_j) / (x_k - x_j)
        denom = 1.0
        for j in range(n):
            if j != k:
                diff = x_nodes[k] - x_nodes[j]
                if abs(diff) < 1e-15:
                    diff = 1e-15
                denom *= diff

        num = np.ones(len(xq), dtype=np.float64)
        for j in range(n):
            if j != k:
                num *= (xq - x_nodes[j])

        phi[k, :] = num / denom

    if scalar_input:
        return phi[:, 0]
    return phi


def lagrange_basis_derivative_1d(x_nodes, x_query):
    """
    计算拉格朗日基函数的导数 dphi_k/dx。

    导数公式：
        dphi_k/dx = phi_k(x) * sum_{j!=k} 1/(x - x_j)
    """
    x_nodes = np.asarray(x_nodes, dtype=np.float64)
    n = len(x_nodes)
    xq = np.atleast_1d(x_query).astype(np.float64)

    dphi = np.zeros((n, len(xq)), dtype=np.float64)
    phi = lagrange_basis_1d(x_nodes, xq)

    for k in range(n):
        for j in range(n):
            if j != k:
                diff = xq - x_nodes[j]
                diff = np.where(np.abs(diff) < 1e-15, 1e-15, diff)
                dphi[k, :] += phi[k, :] / diff

    return dphi


def fem1d_mass_matrix(x_nodes_per_element, element_connectivity):
    """
    组装1D有限元质量矩阵（一致质量矩阵）。

    质量矩阵元素：
        M_{ij} = integral phi_i(x) * phi_j(x) dx

    采用高斯数值积分计算。

    参数:
        x_nodes_per_element: list of arrays, 每个单元的节点坐标
        element_connectivity: (n_elements, n_nodes_per_element) 全局节点编号

    返回:
        M: (n_global_nodes, n_global_nodes) 稀疏表示（字典）
    """
    n_elements = len(element_connectivity)
    n_global = np.max(element_connectivity) + 1

    M = {}

    # 3点高斯积分
    gauss_pts = np.array([-np.sqrt(3.0/5.0), 0.0, np.sqrt(3.0/5.0)])
    gauss_wts = np.array([5.0/9.0, 8.0/9.0, 5.0/9.0])

    for e in range(n_elements):
        nodes_e = x_nodes_per_element[e]
        conn = element_connectivity[e]
        n_loc = len(conn)

        # 映射到参考单元 [-1, 1]
        x1, x2 = nodes_e[0], nodes_e[-1]
        jac = 0.5 * (x2 - x1)

        # 参考单元上的高斯点映射到物理单元
        x_phys = 0.5 * (x2 - x1) * gauss_pts + 0.5 * (x2 + x1)

        phi = lagrange_basis_1d(nodes_e, x_phys)

        for i_loc in range(n_loc):
            I = conn[i_loc]
            for j_loc in range(n_loc):
                J = conn[j_loc]
                val = np.sum(gauss_wts * phi[i_loc, :] * phi[j_loc, :]) * jac
                key = (min(I, J), max(I, J))
                M[key] = M.get(key, 0.0) + val

    return M


def fem1d_stiffness_matrix(x_nodes_per_element, element_connectivity):
    """
    组装1D有限元刚度矩阵。

    刚度矩阵元素：
        K_{ij} = integral dphi_i/dx * dphi_j/dx dx

    参数:
        x_nodes_per_element: 每个单元的节点坐标
        element_connectivity: 全局节点编号

    返回:
        K: 稀疏表示（字典）
    """
    n_elements = len(element_connectivity)
    n_global = np.max(element_connectivity) + 1

    K = {}

    gauss_pts = np.array([-np.sqrt(3.0/5.0), 0.0, np.sqrt(3.0/5.0)])
    gauss_wts = np.array([5.0/9.0, 8.0/9.0, 5.0/9.0])

    for e in range(n_elements):
        nodes_e = x_nodes_per_element[e]
        conn = element_connectivity[e]
        n_loc = len(conn)

        x1, x2 = nodes_e[0], nodes_e[-1]
        jac = 0.5 * (x2 - x1)

        x_phys = 0.5 * (x2 - x1) * gauss_pts + 0.5 * (x2 + x1)

        dphi = lagrange_basis_derivative_1d(nodes_e, x_phys)

        for i_loc in range(n_loc):
            I = conn[i_loc]
            for j_loc in range(n_loc):
                J = conn[j_loc]
                val = np.sum(gauss_wts * dphi[i_loc, :] * dphi[j_loc, :]) * jac
                key = (min(I, J), max(I, J))
                K[key] = K.get(key, 0.0) + val

    return K


def solve_fem1d_radial(r_in, r_out, n_elements, order=2,
                       source_func=None, bc_left=None, bc_right=None):
    """
    使用1D FEM求解径向方程：
        -d/dr(r * dSigma/dr) + alpha * Sigma = S(r)

    这是 Shakura-Sunyaev 盘表面密度方程的简化形式。

    参数:
        r_in, r_out: 径向范围
        n_elements: 单元数
        order: 每个单元的多项式阶数（节点数 = order+1）
        source_func: 源项函数 S(r)
        bc_left, bc_right: 边界条件字典 {'type': 'dirichlet'/'neumann', 'value': float}

    返回:
        r_nodes: 节点坐标
        Sigma: 解向量（如表面密度）
    """
    if source_func is None:
        source_func = lambda r: 0.0

    if bc_left is None:
        bc_left = {'type': 'dirichlet', 'value': 0.0}
    if bc_right is None:
        bc_right = {'type': 'neumann', 'value': 0.0}

    # 均匀网格
    n_nodes = n_elements * order + 1
    r_nodes = np.linspace(r_in, r_out, n_nodes)

    # 单元连接
    element_nodes = []
    connectivity = []
    for e in range(n_elements):
        start = e * order
        conn = np.arange(start, start + order + 1)
        connectivity.append(conn)
        element_nodes.append(r_nodes[conn])

    # 组装系统
    n_global = n_nodes
    A_mat = np.zeros((n_global, n_global), dtype=np.float64)
    b_vec = np.zeros(n_global, dtype=np.float64)

    gauss_pts = np.array([-np.sqrt(3.0/5.0), 0.0, np.sqrt(3.0/5.0)])
    gauss_wts = np.array([5.0/9.0, 8.0/9.0, 5.0/9.0])

    for e in range(n_elements):
        nodes_e = element_nodes[e]
        conn = connectivity[e]
        n_loc = len(conn)

        x1, x2 = nodes_e[0], nodes_e[-1]
        jac = 0.5 * (x2 - x1)
        x_phys = 0.5 * (x2 - x1) * gauss_pts + 0.5 * (x2 + x1)

        phi = lagrange_basis_1d(nodes_e, x_phys)
        dphi = lagrange_basis_derivative_1d(nodes_e, x_phys)

        for i_loc in range(n_loc):
            I = conn[i_loc]
            for j_loc in range(n_loc):
                J = conn[j_loc]
                # 刚度项：r * dphi_i * dphi_j
                stiff_val = np.sum(gauss_wts * x_phys * dphi[i_loc, :] * dphi[j_loc, :]) * jac
                # 质量项（小正则化）
                mass_val = 0.01 * np.sum(gauss_wts * phi[i_loc, :] * phi[j_loc, :]) * jac
                A_mat[I, J] += stiff_val + mass_val

            # 源项
            source_vals = np.array([source_func(x) for x in x_phys])
            b_vec[I] += np.sum(gauss_wts * phi[i_loc, :] * source_vals) * jac

    # 边界条件
    if bc_left['type'] == 'dirichlet':
        A_mat[0, :] = 0.0
        A_mat[0, 0] = 1.0
        b_vec[0] = bc_left['value']
    elif bc_left['type'] == 'neumann':
        b_vec[0] += bc_left['value']

    if bc_right['type'] == 'dirichlet':
        A_mat[-1, :] = 0.0
        A_mat[-1, -1] = 1.0
        b_vec[-1] = bc_right['value']
    elif bc_right['type'] == 'neumann':
        b_vec[-1] += bc_right['value']

    # 求解
    Sigma = np.linalg.solve(A_mat, b_vec)

    return r_nodes, Sigma


def fem_interpolate_1d(r_nodes, values, r_query):
    """
    在1D FEM网格上进行线性插值。
    """
    r_nodes = np.asarray(r_nodes, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    r_query = np.asarray(r_query, dtype=np.float64)

    result = np.zeros(len(r_query), dtype=np.float64)

    for i, rq in enumerate(r_query):
        # 找到所在单元
        if rq <= r_nodes[0]:
            result[i] = values[0]
        elif rq >= r_nodes[-1]:
            result[i] = values[-1]
        else:
            idx = np.searchsorted(r_nodes, rq) - 1
            idx = max(0, min(idx, len(r_nodes) - 2))
            r1, r2 = r_nodes[idx], r_nodes[idx + 1]
            t = (rq - r1) / (r2 - r1) if abs(r2 - r1) > 1e-15 else 0.0
            result[i] = (1.0 - t) * values[idx] + t * values[idx + 1]

    return result
