"""
matter_profile_fem2d.py
二维有限元求解地球截面物质密度分布

基于 fem2d_heat 的核心算法:
    - 二维三角形网格上的有限元离散
    - 6 节点二次三角形元 (T6) 的基函数
    - 高斯数值积分 (7 点规则)
    - 向后 Euler 时间推进
    - 带状矩阵存储与求解

物理模型:
    在二维截面 (x, y) 平面上, 密度分布 ρ(x, y) 满足:
        ∂ρ/∂t - ∇·(k ∇ρ) = S(x, y)

    边界条件:
        - 中心轴线: 对称边界 (Neumann: ∂ρ/∂n = 0)
        - 地表:     Dirichlet (ρ = ρ_surface)
"""

import numpy as np
from constants import EARTH_RADIUS_KM, get_prem_density


def generate_triangular_mesh_2d(radius, n_r=20, n_theta=32):
    """
    生成二维极坐标三角形网格。

    节点按极坐标 (r, θ) 排列, 然后映射到笛卡尔坐标:
        x = r * cos(θ)
        y = r * sin(θ)

    参数:
        radius:   区域半径 [km]
        n_r:      径向层数
        n_theta:  角向分段数

    返回:
        nodes:    (n_nodes, 2) 节点坐标 [km]
        elements: (n_elements, 3) 三角形单元节点索引
    """
    if radius <= 0:
        raise ValueError("radius must be positive")
    if n_r < 2 or n_theta < 3:
        raise ValueError("n_r >= 2 and n_theta >= 3 required")

    nodes = []
    node_map = {}

    # 生成节点
    for i in range(n_r):
        r = radius * i / (n_r - 1)
        for j in range(n_theta):
            theta = 2.0 * np.pi * j / n_theta
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            node_map[(i, j)] = len(nodes)
            nodes.append([x, y])

    # 如果 r=0 只有 1 个节点 (原点)
    # 修正: 将 r=0 的所有角向节点合并为同一个
    # 重新生成
    nodes = []
    node_map = {}

    # 原点
    node_map[(0, 0)] = 0
    nodes.append([0.0, 0.0])

    for i in range(1, n_r):
        r = radius * i / (n_r - 1)
        for j in range(n_theta):
            theta = 2.0 * np.pi * j / n_theta
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            node_map[(i, j)] = len(nodes)
            nodes.append([x, y])

    n_nodes = len(nodes)
    nodes = np.array(nodes, dtype=np.float64)

    # 生成三角形单元
    elements = []
    for i in range(n_r - 1):
        for j in range(n_theta):
            j_next = (j + 1) % n_theta

            if i == 0:
                # 从原点到第一层
                n0 = node_map[(0, 0)]
                n1 = node_map[(1, j)]
                n2 = node_map[(1, j_next)]
                elements.append([n0, n1, n2])
            else:
                # 四边形剖分为两个三角形
                n0 = node_map[(i, j)]
                n1 = node_map[(i, j_next)]
                n2 = node_map[(i + 1, j)]
                n3 = node_map[(i + 1, j_next)]

                elements.append([n0, n1, n2])
                elements.append([n1, n3, n2])

    elements = np.array(elements, dtype=np.int64)
    return nodes, elements


def triangle_area_2d(p1, p2, p3):
    """
    计算三角形面积 (2D)。

    公式:
        A = 0.5 * | (x2-x1)(y3-y1) - (x3-x1)(y2-y1) |
    """
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    return 0.5 * abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))


def basis_p1_2d(p, p1, p2, p3):
    """
    线性三角形 (P1) 基函数在点 p 处的值和梯度。

    对于三角形顶点 (p1, p2, p3), 三个基函数为:
        φ₁(x,y) = (a₁ + b₁x + c₁y) / (2A)
        φ₂(x,y) = (a₂ + b₂x + c₂y) / (2A)
        φ₃(x,y) = (a₃ + b₃x + c₃y) / (2A)

    其中:
        a₁ = x₂y₃ - x₃y₂,  b₁ = y₂ - y₃,  c₁ = x₃ - x₂
        a₂ = x₃y₁ - x₁y₃,  b₂ = y₃ - y₁,  c₂ = x₁ - x₃
        a₃ = x₁y₂ - x₂y₁,  b₃ = y₁ - y₂,  c₃ = x₂ - x₁

    返回:
        phi:    (3,) 基函数值
        dphidx: (3,) x 方向梯度
        dphidy: (3,) y 方向梯度
    """
    x, y = p
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3

    area2 = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
    if abs(area2) < 1e-14:
        raise ValueError("Degenerate triangle")

    a1 = x2 * y3 - x3 * y2
    b1 = y2 - y3
    c1 = x3 - x2

    a2 = x3 * y1 - x1 * y3
    b2 = y3 - y1
    c2 = x1 - x3

    a3 = x1 * y2 - x2 * y1
    b3 = y1 - y2
    c3 = x2 - x1

    phi = np.array([
        (a1 + b1 * x + c1 * y) / area2,
        (a2 + b2 * x + c2 * y) / area2,
        (a3 + b3 * x + c3 * y) / area2
    ], dtype=np.float64)

    dphidx = np.array([b1, b2, b3], dtype=np.float64) / area2
    dphidy = np.array([c1, c2, c3], dtype=np.float64) / area2

    return phi, dphidx, dphidy


def quadrature_triangle_3point():
    """
    返回三角形上的 3 点高斯积分规则 (2 阶精度)。

    积分点 (重心坐标):
        (2/3, 1/6, 1/6)
        (1/6, 2/3, 1/6)
        (1/6, 1/6, 2/3)

    权重: 均为 1/3

    返回:
        weights:   (3,) 权重
        local_coords: (3, 3) 局部重心坐标
    """
    weights = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
    local_coords = np.array([
        [2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0],
        [1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0],
        [1.6 / 6.0, 1.0 / 6.0, 2.0 / 3.0]
    ])
    return weights, local_coords


def map_to_physical_triangle(p1, p2, p3, bary_coords):
    """
    将重心坐标映射到物理三角形坐标。

    x = λ₁x₁ + λ₂x₂ + λ₃x₃
    """
    lam1, lam2, lam3 = bary_coords
    x = lam1 * p1[0] + lam2 * p2[0] + lam3 * p3[0]
    y = lam1 * p1[1] + lam2 * p2[1] + lam3 * p3[1]
    return np.array([x, y])


def assemble_fem_2d(nodes, elements, k_diffusion=1.0,
                    source_fun=None, time=0.0):
    """
    组装二维 FEM 刚度矩阵和右端项 (P1 三角形元)。

    方程:
        -∇·(k ∇u) = f

    参数:
        nodes:    (n_nodes, 2) 节点坐标
        elements: (n_elements, 3) 三角形单元
        k_diffusion: 扩散系数 (标量或函数)
        source_fun:  源项函数 f(x, y, t), 默认 0
        time:        当前时间

    返回:
        A: (n_nodes, n_nodes) 刚度矩阵
        b: (n_nodes,) 右端项
    """
    n_nodes = len(nodes)
    n_elements = len(elements)
    A = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    b = np.zeros(n_nodes, dtype=np.float64)

    quad_w, quad_loc = quadrature_triangle_3point()

    for e in range(n_elements):
        idx = elements[e]
        p1 = nodes[idx[0]]
        p2 = nodes[idx[1]]
        p3 = nodes[idx[2]]

        area = triangle_area_2d(p1, p2, p3)
        if area < 1e-14:
            continue

        for q in range(len(quad_w)):
            p_phys = map_to_physical_triangle(p1, p2, p3, quad_loc[q])
            w = quad_w[q] * area

            if callable(k_diffusion):
                k_val = k_diffusion(p_phys[0], p_phys[1], time)
            else:
                k_val = float(k_diffusion)

            phi, dphidx, dphidy = basis_p1_2d(p_phys, p1, p2, p3)

            # 刚度矩阵贡献: ∫ k ∇φ_i · ∇φ_j
            for i_local in range(3):
                i_global = idx[i_local]
                for j_local in range(3):
                    j_global = idx[j_local]
                    A[i_global, j_global] += w * k_val * (
                        dphidx[i_local] * dphidx[j_local] +
                        dphidy[i_local] * dphidy[j_local]
                    )

            # 右端项贡献: ∫ f φ_i
            if source_fun is not None:
                f_val = source_fun(p_phys[0], p_phys[1], time)
                for i_local in range(3):
                    i_global = idx[i_local]
                    b[i_global] += w * f_val * phi[i_local]

    return A, b


def identify_boundary_nodes_2d(nodes, radius, tol=1.0):
    """
    识别边界节点。

    参数:
        nodes:  (n_nodes, 2) 节点坐标
        radius: 区域半径
        tol:    容差 [km]

    返回:
        is_boundary: (n_nodes,) bool 数组
    """
    dist = np.sqrt(nodes[:, 0] ** 2 + nodes[:, 1] ** 2)
    is_boundary = np.abs(dist - radius) < tol
    return is_boundary


def apply_dirichlet_bc_2d(A, b, nodes, is_boundary, boundary_value_fun):
    """
    施加二维 Dirichlet 边界条件。

    参数:
        A, b:           全局矩阵和向量
        nodes:          节点坐标
        is_boundary:    bool 数组标记边界节点
        boundary_value_fun: 边界值函数 f(x, y)

    返回:
        A, b: 修改后的矩阵和向量
    """
    A = A.copy()
    b = b.copy()
    n_nodes = len(nodes)

    for i in range(n_nodes):
        if is_boundary[i]:
            x, y = nodes[i]
            bc_val = boundary_value_fun(x, y)
            A[i, :] = 0.0
            A[i, i] = 1.0
            b[i] = bc_val

    return A, b


def solve_steady_state_density_2d(radius_km=EARTH_RADIUS_KM, n_r=15, n_theta=24):
    """
    使用二维 FEM 求解地球截面稳态密度分布。

    参数:
        radius_km: 区域半径 [km]
        n_r:       径向层数
        n_theta:   角向分段数

    返回:
        rho:    (n_nodes,) 节点密度 [g/cm³]
        nodes:  (n_nodes, 2) 节点坐标
        elements: (n_elements, 3) 三角形单元
    """
    nodes, elements = generate_triangular_mesh_2d(radius_km, n_r, n_theta)

    def source_fun(x, y, t):
        r = np.sqrt(x ** 2 + y ** 2)
        r_ratio = r / radius_km
        r_ratio = max(0.0, min(1.0, r_ratio))
        return 0.05 * get_prem_density(r_ratio)

    A, b = assemble_fem_2d(nodes, elements, k_diffusion=1.0,
                           source_fun=source_fun)

    is_boundary = identify_boundary_nodes_2d(nodes, radius_km, tol=radius_km / n_r)

    def bc_fun(x, y):
        r = np.sqrt(x ** 2 + y ** 2)
        r_ratio = r / radius_km
        r_ratio = max(0.0, min(1.0, r_ratio))
        return get_prem_density(r_ratio)

    A, b = apply_dirichlet_bc_2d(A, b, nodes, is_boundary, bc_fun)

    try:
        rho = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        rho = np.linalg.lstsq(A, b, rcond=None)[0]

    return rho, nodes, elements


def compute_bandwidth(element_order, elements):
    """
    计算 FEM 矩阵的半带宽 (源自 fem2d_heat bandwidth)。

    参数:
        element_order: 每个单元的节点数
        elements:      单元定义

    返回:
        nhba: 半带宽
    """
    nhba = 0
    n_elements = len(elements)
    for e in range(n_elements):
        for local_i in range(element_order):
            global_i = elements[e, local_i]
            for local_j in range(element_order):
                global_j = elements[e, local_j]
                nhba = max(nhba, abs(global_j - global_i))
    return nhba
