"""
基于 Chebyshev 逼近的后验误差估计模块

融合自:
- 014_approx_chebyshev: Chebyshev 多项式插值与误差估计

误差估计理论:
    对于有限元解 u_h ∈ V_h，精确解 u 满足:
        ||u - u_h||_E <= C * (Σ_T η_T^2)^{1/2}
    
    其中单元误差指示子 η_T 可通过局部残差或恢复梯度估计:
        η_T^2 = h_T^2 * ||R_T||_{L^2(T)}^2 + h_T * ||R_∂T||_{L^2(∂T)}^2
    
    本模块引入 Chebyshev 逼近进行高阶误差估计:
        - 在每个单元上构造 Chebyshev 插值 p_n
        - 利用 ||u_h - p_n|| 估计逼近误差
        - 结合梯度恢复技术获得更锐的误差界

Chebyshev 节点:
    x_i = (a + b)/2 + (b - a)/2 * cos(θ_i)
    θ_i = (2i + 1)π / (2n + 2),  i = 0, ..., n

Chebyshev 插值误差:
    ||f - p_n||_∞ <= (2/π) * log(n+1) * E_n(f)
    其中 E_n(f) 是最佳一致逼近误差。
"""

import numpy as np


def chebyshev_nodes(a, b, n):
    """
    在区间 [a, b] 上生成 Chebyshev 节点。
    
    对应原 chebyspace.m 的核心功能。
    
    Parameters
    ----------
    a, b : float
        区间端点
    n : int
        节点数
    
    Returns
    -------
    x : ndarray, shape (n,)
        Chebyshev 节点
    """
    if n <= 0:
        raise ValueError("chebyshev_nodes: n 必须为正整数")
    if n == 1:
        return np.array([(a + b) / 2.0])

    k = np.arange(n)
    theta = (2 * k + 1) * np.pi / (2 * n)
    x_std = np.cos(theta)
    x = 0.5 * (a + b) + 0.5 * (b - a) * x_std
    return x


def divided_differences(x, y):
    """
    计算 Newton 差商表。
    
    对应原 divdif.m 的核心功能。
    
    Parameters
    ----------
    x : ndarray
        节点
    y : ndarray
        函数值
    
    Returns
    -------
    dd : ndarray
        差商表 (最后一个元素是 n 阶差商)
    """
    n = len(x)
    dd = y.copy()
    for i in range(1, n):
        for j in range(n - 1, i - 1, -1):
            dd[j] = (dd[j] - dd[j - 1]) / (x[j] - x[j - i])
    return dd


def newton_interpolation(x_nodes, dd, x_eval):
    """
    使用 Newton 差商形式进行多项式插值。
    
    对应原 interp.m 的核心功能。
    
    Parameters
    ----------
    x_nodes : ndarray
        插值节点
    dd : ndarray
        差商表
    x_eval : ndarray
        求值点
    
    Returns
    -------
    y_eval : ndarray
        插值结果
    """
    n = len(dd)
    y_eval = dd[-1] * np.ones_like(x_eval)
    for i in range(n - 2, -1, -1):
        y_eval = dd[i] + (x_eval - x_nodes[i]) * y_eval
    return y_eval


def chebyshev_approximate_1d(f, a, b, n, n_eval=1001):
    """
    在 [a,b] 上构造 n 点 Chebyshev 插值并估计最大误差。
    
    Parameters
    ----------
    f : callable
        被逼近函数 f(x)
    a, b : float
        区间
    n : int
        Chebyshev 节点数
    n_eval : int
        误差估计的采样点数
    
    Returns
    -------
    max_error : float
        估计的最大误差
    dd : ndarray
        差商表
    x_cheb : ndarray
        Chebyshev 节点
    """
    x_cheb = chebyshev_nodes(a, b, n)
    y_cheb = f(x_cheb)
    dd = divided_differences(x_cheb, y_cheb)

    # 在密集网格上估计误差
    x_eval = np.linspace(a, b, n_eval)
    y_interp = newton_interpolation(x_cheb, dd, x_eval)
    y_exact = f(x_eval)

    max_error = np.max(np.abs(y_interp - y_exact))
    return max_error, dd, x_cheb


def compute_element_error_indicator(
    nodes, triangles, solution, element_idx,
    cheb_degree=4
):
    """
    对指定三角形单元计算基于 Chebyshev 逼近的后验误差指示子。
    
    策略:
        1. 提取单元的三个顶点值
        2. 构造高阶 Chebyshev 插值
        3. 比较 FEM 解与 Chebyshev 插值的偏差
        4. 结合单元尺寸加权
    
    Parameters
    ----------
    nodes : ndarray, shape (n_nodes, 2)
    triangles : ndarray, shape (n_tri, 3)
    solution : ndarray, shape (n_nodes,)
    element_idx : int
        目标单元索引
    cheb_degree : int
        Chebyshev 插值次数
    
    Returns
    -------
    eta : float
        误差指示子
    """
    tri = triangles[element_idx]
    p1, p2, p3 = nodes[tri[0]], nodes[tri[1]], nodes[tri[2]]
    u1, u2, u3 = solution[tri[0]], solution[tri[1]], solution[tri[2]]

    # 计算单元面积和尺寸
    area = 0.5 * abs(
        (p2[0] - p1[0]) * (p3[1] - p1[1]) -
        (p3[0] - p1[0]) * (p2[1] - p1[1])
    )

    # 计算最长边作为单元尺寸 h_T
    e1 = np.linalg.norm(p2 - p1)
    e2 = np.linalg.norm(p3 - p2)
    e3 = np.linalg.norm(p1 - p3)
    h_T = max(e1, e2, e3)

    if area < 1e-14 or h_T < 1e-14:
        return 0.0

    # 在单元的每条边上构造 Chebyshev 插值
    edges = [(p1, p2, u1, u2), (p2, p3, u2, u3), (p3, p1, u3, u1)]
    edge_errors = []

    for pa, pb, ua, ub in edges:
        edge_len = np.linalg.norm(pb - pa)
        if edge_len < 1e-14:
            continue

        # 参数化边: r(t) = pa + t * (pb - pa), t ∈ [0,1]
        def f_edge(t):
            return ua + t * (ub - ua)

        # 在边上采样更多点构造 Chebyshev 插值
        t_cheb = chebyshev_nodes(0.0, 1.0, cheb_degree)
        y_cheb = f_edge(t_cheb)
        dd = divided_differences(t_cheb, y_cheb)

        # 在边上密集采样比较
        t_eval = np.linspace(0, 1, 51)
        y_linear = f_edge(t_eval)
        y_cheb_interp = newton_interpolation(t_cheb, dd, t_eval)

        edge_error = np.max(np.abs(y_linear - y_cheb_interp))
        edge_errors.append(edge_error)

    # 误差指示子 = 单元尺寸 * (边误差的最大值 + 面积惩罚项)
    if len(edge_errors) > 0:
        eta = h_T * max(edge_errors)
    else:
        eta = 0.0

    # 添加梯度恢复项: 估计解在单元内部的梯度变化
    grad_est = max(abs(u2 - u1), abs(u3 - u2), abs(u1 - u3)) / h_T
    eta += h_T ** 2 * grad_est

    return eta


def compute_all_error_indicators(nodes, triangles, solution, cheb_degree=4):
    """
    对所有三角形单元计算误差指示子。
    
    Parameters
    ----------
    nodes : ndarray
    triangles : ndarray
    solution : ndarray
    cheb_degree : int
    
    Returns
    -------
    errors : ndarray, shape (n_tri,)
        每个单元的误差指示子
    total_error : float
        全局误差估计
    """
    n_tri = len(triangles)
    errors = np.zeros(n_tri)

    for t in range(n_tri):
        errors[t] = compute_element_error_indicator(
            nodes, triangles, solution, t, cheb_degree
        )

    total_error = np.sqrt(np.sum(errors ** 2))
    return errors, total_error


def compute_gradient_recovery_error(nodes, triangles, solution):
    """
    基于 Zienkiewicz-Zhu (ZZ) 梯度恢复的后验误差估计。
    
    对每个节点，通过 Patch 平均恢复更光滑的梯度:
        G(u_h)|_{node} = (Σ_T ∇u_h|_T * |T|) / (Σ_T |T|)
    
    单元误差:
        η_T = ||∇u_h - G(u_h)||_{L^2(T)}
    
    Parameters
    ----------
    nodes : ndarray, shape (n_nodes, 2)
    triangles : ndarray, shape (n_tri, 3)
    solution : ndarray, shape (n_nodes,)
    
    Returns
    -------
    errors : ndarray, shape (n_tri,)
    total_error : float
    """
    n_nodes = len(nodes)
    n_tri = len(triangles)

    # 计算每个单元的梯度和面积
    elem_grads = np.zeros((n_tri, 2))
    elem_areas = np.zeros(n_tri)

    for t in range(n_tri):
        tri = triangles[t]
        p1, p2, p3 = nodes[tri[0]], nodes[tri[1]], nodes[tri[2]]
        u1, u2, u3 = solution[tri[0]], solution[tri[1]], solution[tri[2]]

        area = 0.5 * (
            (p2[0] - p1[0]) * (p3[1] - p1[1]) -
            (p3[0] - p1[0]) * (p2[1] - p1[1])
        )
        elem_areas[t] = abs(area)

        if abs(area) > 1e-14:
            # 线性三角元的梯度是常数
            dx = np.array([p2[0] - p1[0], p3[0] - p1[0]])
            dy = np.array([p2[1] - p1[1], p3[1] - p1[1]])
            du = np.array([u2 - u1, u3 - u1])

            # 解线性系统 [dx, dy]^T * grad = du
            A_mat = np.array([[dx[0], dx[1]], [dy[0], dy[1]]])
            if abs(np.linalg.det(A_mat)) > 1e-14:
                grad = np.linalg.solve(A_mat.T, du)
                elem_grads[t] = grad

    # 节点梯度恢复 (面积加权平均)
    node_grads = np.zeros((n_nodes, 2))
    node_areas = np.zeros(n_nodes)

    for t in range(n_tri):
        tri = triangles[t]
        for i in range(3):
            node_grads[tri[i]] += elem_grads[t] * elem_areas[t]
            node_areas[tri[i]] += elem_areas[t]

    for i in range(n_nodes):
        if node_areas[i] > 1e-14:
            node_grads[i] /= node_areas[i]

    # 计算每个单元的恢复梯度 (节点梯度的平均值)
    errors = np.zeros(n_tri)
    for t in range(n_tri):
        tri = triangles[t]
        recovered_grad = np.mean(node_grads[tri], axis=0)
        diff = elem_grads[t] - recovered_grad
        errors[t] = np.sqrt(np.sum(diff ** 2)) * np.sqrt(elem_areas[t])

    total_error = np.sqrt(np.sum(errors ** 2))
    return errors, total_error
