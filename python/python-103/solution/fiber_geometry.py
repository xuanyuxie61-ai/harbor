"""
fiber_geometry.py
光纤截面几何建模模块（对应种子项目 1308_triangle_integrands, 1333_triangulation_boundary_nodes）

在光子晶体光纤（PCF）中，截面几何结构对模式分布、非线性系数和色散特性
具有决定性影响。本模块提供：
  1. 基于三角形网格的光纤截面离散化
  2. 三角形区域上的高斯积分（用于计算有效模场面积）
  3. 边界节点识别（用于施加Dirichlet/Neumann边界条件）

核心物理公式：
  有效模场面积:
    A_eff = [∫∫ |E(x,y)|² dxdy]² / ∫∫ |E(x,y)|⁴ dxdy

  非线性系数:
    γ = n₂ ω₀ / (c A_eff)

  其中n₂为石英的非线性折射率 (~2.6×10⁻²⁰ m²/W)，c为光速。

  三角形高斯积分（Wandzura规则）:
    ∫_Δ f(x,y) dA ≈ |Δ| Σ_{i=1}^{N} w_i f(x_i, y_i)
"""

import numpy as np


# Wandzura 5阶三角形高斯积分节点和权重（面积坐标）
_WANDZURA05_X = np.array([
    0.33333333333333,
    0.05971587178977, 0.79742698535309, 0.14296124917414,
    0.47014206410512, 0.47014206410512, 0.05971587178977
])
_WANDZURA05_Y = np.array([
    0.33333333333333,
    0.79742698535309, 0.14296124917414, 0.05971587178977,
    0.47014206410512, 0.05971587178977, 0.47014206410512
])
_WANDZURA05_W = np.array([
    0.22500000000000,
    0.13239415278851, 0.13239415278851, 0.13239415278851,
    0.12593918054483, 0.12593918054483, 0.12593918054483
])


def triangle_area(v1, v2, v3):
    """
    计算三角形面积（有向面积的一半）。
    v1, v2, v3: ndarray shape (2,)
    """
    v1 = np.asarray(v1, dtype=float)
    v2 = np.asarray(v2, dtype=float)
    v3 = np.asarray(v3, dtype=float)
    return 0.5 * abs((v2[0] - v1[0]) * (v3[1] - v1[1]) - (v2[1] - v1[1]) * (v3[0] - v1[0]))


def triangle_integrand_gauss(f, v1, v2, v3):
    """
    在三角形(v1,v2,v3)上应用Wandzura 5阶高斯积分计算 ∫ f dA。

    参数:
        f: callable, f(x,y) -> float
        v1, v2, v3: ndarray shape (2,), 三角形顶点
    """
    v1 = np.asarray(v1, dtype=float)
    v2 = np.asarray(v2, dtype=float)
    v3 = np.asarray(v3, dtype=float)

    area = triangle_area(v1, v2, v3)
    result = 0.0
    for i in range(_WANDZURA05_X.size):
        # 面积坐标到笛卡尔坐标
        xi = _WANDZURA05_X[i]
        eta = _WANDZURA05_Y[i]
        zeta = 1.0 - xi - eta
        x = xi * v1[0] + eta * v2[0] + zeta * v3[0]
        y = xi * v1[1] + eta * v2[1] + zeta * v3[1]
        result += _WANDZURA05_W[i] * f(x, y)

    return area * result


def create_fiber_triangulation(r_core, r_cladding, n_theta=16, n_radial_core=4, n_radial_clad=6):
    """
    创建阶跃型光纤截面的三角剖分（环形网格）。

    返回:
        nodes: ndarray shape (N, 2), 节点坐标
        triangles: ndarray shape (M, 3), 三角形节点索引
        boundary_flags: ndarray shape (N,), 1表示边界节点，0表示内部节点
    """
    if r_core <= 0 or r_cladding <= r_core or n_theta < 3:
        raise ValueError("create_fiber_triangulation: invalid geometry parameters")

    nodes = []
    # 中心点
    nodes.append([0.0, 0.0])

    # 核心层节点
    for i in range(1, n_radial_core + 1):
        r = i * r_core / n_radial_core
        for j in range(n_theta):
            theta = 2.0 * np.pi * j / n_theta
            nodes.append([r * np.cos(theta), r * np.sin(theta)])

    # 包层节点
    for i in range(1, n_radial_clad + 1):
        r = r_core + i * (r_cladding - r_core) / n_radial_clad
        for j in range(n_theta):
            theta = 2.0 * np.pi * j / n_theta
            nodes.append([r * np.cos(theta), r * np.sin(theta)])

    nodes = np.array(nodes)

    triangles = []
    # 中心到第一层
    for j in range(n_theta):
        n0 = 0
        n1 = 1 + j
        n2 = 1 + (j + 1) % n_theta
        triangles.append([n0, n1, n2])

    # 核心层内部
    for i in range(n_radial_core - 1):
        offset = 1 + i * n_theta
        offset_next = 1 + (i + 1) * n_theta
        for j in range(n_theta):
            n0 = offset + j
            n1 = offset + (j + 1) % n_theta
            n2 = offset_next + j
            n3 = offset_next + (j + 1) % n_theta
            triangles.append([n0, n1, n2])
            triangles.append([n1, n3, n2])

    # 核心-包层界面
    core_offset = 1 + (n_radial_core - 1) * n_theta
    clad_offset = 1 + n_radial_core * n_theta
    for j in range(n_theta):
        n0 = core_offset + j
        n1 = core_offset + (j + 1) % n_theta
        n2 = clad_offset + j
        n3 = clad_offset + (j + 1) % n_theta
        triangles.append([n0, n1, n2])
        triangles.append([n1, n3, n2])

    # 包层内部
    for i in range(n_radial_clad - 1):
        offset = clad_offset + i * n_theta
        offset_next = clad_offset + (i + 1) * n_theta
        for j in range(n_theta):
            n0 = offset + j
            n1 = offset + (j + 1) % n_theta
            n2 = offset_next + j
            n3 = offset_next + (j + 1) % n_theta
            triangles.append([n0, n1, n2])
            triangles.append([n1, n3, n2])

    triangles = np.array(triangles, dtype=int)

    # 识别边界节点
    boundary_flags = np.zeros(nodes.shape[0], dtype=int)
    # 外边界节点
    outer_offset = clad_offset + (n_radial_clad - 1) * n_theta
    for j in range(n_theta):
        boundary_flags[outer_offset + j] = 1

    # 核心-包层界面也标记为边界（用于区分不同折射率区域）
    for j in range(n_theta):
        boundary_flags[core_offset + j] = 1
        boundary_flags[clad_offset + j] = 1

    return nodes, triangles, boundary_flags


def identify_boundary_nodes(triangles, node_num):
    """
    从三角剖分中识别边界节点（对应种子项目1333_triangulation_boundary_nodes）。

    算法：边界边只被一个三角形共享。统计每条边的出现次数，
    出现一次的边的端点即为边界节点。
    """
    triangles = np.asarray(triangles, dtype=int)
    edge_count = {}

    for tri in triangles:
        # 三角形的三条边（排序以保证无向一致性）
        edges = [
            tuple(sorted([tri[0], tri[1]])),
            tuple(sorted([tri[1], tri[2]])),
            tuple(sorted([tri[2], tri[0]]))
        ]
        for e in edges:
            edge_count[e] = edge_count.get(e, 0) + 1

    boundary_nodes = np.zeros(node_num, dtype=int)
    for e, count in edge_count.items():
        if count == 1:
            boundary_nodes[e[0]] = 1
            boundary_nodes[e[1]] = 1

    return boundary_nodes


def compute_effective_area(nodes, triangles, mode_field):
    """
    计算有效模场面积 A_eff。

    参数:
        nodes: ndarray shape (N, 2)
        triangles: ndarray shape (M, 3)
        mode_field: callable, f(x,y) -> complex, 模式场分布

    返回:
        A_eff: float
    """
    num_integrand = lambda x, y: np.abs(mode_field(x, y)) ** 2
    den_integrand = lambda x, y: np.abs(mode_field(x, y)) ** 4

    num = 0.0
    den = 0.0
    for tri in triangles:
        v1, v2, v3 = nodes[tri[0]], nodes[tri[1]], nodes[tri[2]]
        num += triangle_integrand_gauss(num_integrand, v1, v2, v3)
        den += triangle_integrand_gauss(den_integrand, v1, v2, v3)

    if den <= 0:
        return np.inf

    return (num ** 2) / den


def compute_nonlinear_coefficient(n2, omega0, A_eff):
    """
    计算光纤非线性系数 γ = n₂ ω₀ / (c A_eff)。

    参数:
        n2: float, 非线性折射率 (m²/W)
        omega0: float, 中心角频率 (rad/s)
        A_eff: float, 有效模场面积 (m²)

    返回:
        gamma: float, 非线性系数 (1/(W·m))
    """
    c = 2.99792458e8  # 光速 m/s
    if A_eff <= 0:
        raise ValueError("compute_nonlinear_coefficient: A_eff must be positive")
    return n2 * omega0 / (c * A_eff)
