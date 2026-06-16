"""
divertor_geometry.py - 偏滤器几何描述、SOL网格生成与网格拓扑

本模块融合以下种子项目的核心算法：
  - 109_boundary_word_right → 边界字表示法，多边形网格生成
  - 756_mesh_vtoe → 顶点到单元映射（VTOE）逆连接计算
  - 535_hilbert_curve → Hilbert空间填充曲线排序（缓存优化）

功能：
  1. 偏滤器靶板几何的边界字描述
  2. SOL（Scrape-Off Layer）结构化网格生成
  3. 非结构化三角网格及其连接关系
  4. Hilbert曲线排序优化内存局部性
  5. 磁面坐标到物理坐标的映射
"""

import numpy as np
from typing import Tuple, List, Dict, Optional


# =============================================================================
# 边界字表示法（来自109_boundary_word_right）
# =============================================================================
# 八方向编码：用于描述偏滤器靶板表面的分段轮廓
# 方向编码：E=东, NE=东北, N=北, NW=西北, W=西, SW=西南, S=南, SE=东南
DIRECTION_MAP = {
    'E': (1.0, 0.0),
    'NE': (0.7071, 0.7071),
    'N': (0.0, 1.0),
    'NW': (-0.7071, 0.7071),
    'W': (-1.0, 0.0),
    'SW': (-0.7071, -0.7071),
    'S': (0.0, -1.0),
    'SE': (0.7071, -0.7071),
}


def boundary_word_to_vertices(boundary_word: str,
                                step_size: float = 0.01) -> np.ndarray:
    """
    将边界字转换为顶点坐标序列

    偏滤器靶板表面形状由边界字描述：每个字母代表一个方向段

    参数:
        boundary_word: 方向字符序列（如 "EENESW"）
        step_size: 每段步长 [m]
    返回:
        vertices: shape (N, 2) 的顶点坐标数组
    """
    vertices = [(0.0, 0.0)]
    x, y = 0.0, 0.0

    for char in boundary_word.upper():
        if char in DIRECTION_MAP:
            dx, dy = DIRECTION_MAP[char]
            x += dx * step_size
            y += dy * step_size
            vertices.append((x, y))

    return np.array(vertices)


def generate_divertor_target_profile(R_major: float = 1.5,
                                      a_minor: float = 0.5,
                                      tilt_angle_deg: float = 5.0,
                                      n_points: int = 64) -> np.ndarray:
    """
    生成偏滤器靶板表面轮廓（用于计算靶板热负荷分布）

    靶板位于 tokamak 大半径 R 处，具有倾斜角以分散热负荷
    靶板径向范围: [R - a, R + a]

    参数:
        R_major: 大半径 [m]
        a_minor: 小半径 [m]
        tilt_angle_deg: 靶板倾斜角 [度]
        n_points: 离散点数
    返回:
        target_points: shape (n_points, 2) 靶板上的点 (R, Z)
    """
    # 靶板径向范围
    R_inner = R_major - 0.3 * a_minor
    R_outer = R_major + 0.3 * a_minor

    R_values = np.linspace(R_inner, R_outer, n_points)

    # 靶板倾斜（相对于水平面的角度）
    tilt_rad = np.radians(tilt_angle_deg)
    Z_base = -0.8  # 靶板基准Z位置（下偏滤器）

    # 倾斜靶板Z坐标
    dR = R_values - R_major
    Z_values = Z_base + dR * np.tan(tilt_rad)

    # 加上轻微的弧度（模拟靶板曲率）
    curvature = 0.1 / a_minor
    Z_values += curvature * dR**2

    return np.column_stack([R_values, Z_values])


# =============================================================================
# SOL结构化网格生成
# =============================================================================
def generate_sol_mesh(R_major: float = 1.5,
                       a_minor: float = 0.5,
                       n_radial: int = 32,
                       n_poloidal: int = 64,
                       psi_min: float = 1.01,
                       psi_max: float = 1.20) -> Dict[str, np.ndarray]:
    """
    生成SOL（Scrape-Off Layer）的结构化网格

    使用磁面坐标 (psi, theta) -> (R, Z) 映射
    psi: 归一化极向磁通
    theta: 极向角

    映射关系（简化圆截面近似）:
        R = R_major + (a * sqrt(psi)) * cos(theta)
        Z = (a * sqrt(psi) * kappa) * sin(theta)
    其中 kappa 是拉长比

    参数:
        R_major: 大半径 [m]
        a_minor: 小半径 [m]
        n_radial: 径向网格点数
        n_poloidal: 极向网格点数
        psi_min: 最小归一化磁通（略大于分离面）
        psi_max: 最大归一化磁通
    返回:
        dict包含 'R', 'Z', 'psi', 'theta', 'metric' 等
    """
    psi_1d = np.linspace(psi_min, psi_max, n_radial)
    theta_1d = np.linspace(0, 2 * np.pi, n_poloidal, endpoint=False)

    psi_2d, theta_2d = np.meshgrid(psi_1d, theta_1d, indexing='ij')

    # 磁面坐标到物理坐标的映射
    # R = R0 + a * sqrt(psi) * cos(theta)
    # Z = a * kappa * sqrt(psi) * sin(theta)
    kappa = 1.6  # 典型ITER拉长比

    sqrt_psi = np.sqrt(psi_2d)
    R = R_major + a_minor * sqrt_psi * np.cos(theta_2d)
    Z = a_minor * kappa * sqrt_psi * np.sin(theta_2d)

    # 计算度量张量 g_ij = (partial r / partial xi^i) . (partial r / partial xi^j)
    # 其中 xi^1 = psi, xi^2 = theta
    # g_11 = (dR/dpsi)^2 + (dZ/dpsi)^2
    # g_12 = (dR/dpsi)(dR/dtheta) + (dZ/dpsi)(dZ/dtheta)
    # g_22 = (dR/dtheta)^2 + (dZ/dtheta)^2

    dR_dpsi = 0.5 * a_minor / (sqrt_psi + 1e-30) * np.cos(theta_2d)
    dZ_dpsi = 0.5 * a_minor * kappa / (sqrt_psi + 1e-30) * np.sin(theta_2d)
    dR_dtheta = -a_minor * sqrt_psi * np.sin(theta_2d)
    dZ_dtheta = a_minor * kappa * sqrt_psi * np.cos(theta_2d)

    g11 = dR_dpsi**2 + dZ_dpsi**2
    g12 = dR_dpsi * dR_dtheta + dZ_dpsi * dZ_dtheta
    g22 = dR_dtheta**2 + dZ_dtheta**2

    # Jacobian: J = sqrt(g) = sqrt(g11*g22 - g12^2)
    J = np.sqrt(np.abs(g11 * g22 - g12**2) + 1e-30)

    # 磁场强度（简化：B = B0 * R0/R）
    B0 = 5.3  # 磁场强度 [T]（ITER级）
    B_mag = B0 * R_major / R

    # 平行连接长度（近似）
    q_safety = 1.5 + 2.0 * psi_2d  # 安全因子（简化）
    L_parallel = 2.0 * np.pi * R * q_safety  # 近似连接长度

    return {
        'R': R,
        'Z': Z,
        'psi': psi_2d,
        'theta': theta_2d,
        'g11': g11,
        'g12': g12,
        'g22': g22,
        'J': J,
        'B': B_mag,
        'L_parallel': L_parallel,
        'n_radial': n_radial,
        'n_poloidal': n_poloidal,
        'R_major': R_major,
        'a_minor': a_minor,
    }


# =============================================================================
# 非结构化三角网格（用于FEM求解器）
# =============================================================================
def generate_triangular_mesh(x_min: float, x_max: float,
                              y_min: float, y_max: float,
                              nx: int = 16, ny: int = 16) -> Dict[str, np.ndarray]:
    """
    在矩形区域上生成三角形有限元网格

    每个矩形单元被分为两个三角形（对角线切割）

    参数:
        x_min, x_max: x方向范围
        y_min, y_max: y方向范围
        nx, ny: 网格点数
    返回:
        dict包含 'nodes', 'elements', 'n_nodes', 'n_elements'
    """
    x = np.linspace(x_min, x_max, nx)
    y = np.linspace(y_min, y_max, ny)
    xx, yy = np.meshgrid(x, y)

    # 节点坐标
    nodes = np.column_stack([xx.ravel(), yy.ravel()])
    n_nodes = len(nodes)

    # 单元连接（每个矩形分两个三角形）
    elements = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            # 四个顶点编号
            n0 = j * nx + i
            n1 = n0 + 1
            n2 = n0 + nx
            n3 = n2 + 1

            # 两个三角形
            elements.append([n0, n1, n2])
            elements.append([n1, n3, n2])

    elements = np.array(elements, dtype=np.int32)
    n_elements = len(elements)

    return {
        'nodes': nodes,
        'elements': elements,
        'n_nodes': n_nodes,
        'n_elements': n_elements,
        'nx': nx,
        'ny': ny,
    }


# =============================================================================
# 顶点到单元映射 (VTOE)（来自756_mesh_vtoe）
# =============================================================================
def compute_vertex_to_element_map(elements: np.ndarray,
                                   n_nodes: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算顶点到单元的逆连接映射

    基于排序的高效算法：
    1. 创建 (vertex, element) 对的列表
    2. 按顶点编号排序
    3. 构建指针数组

    参数:
        elements: shape (n_elements, 3) 单元连接表
        n_nodes: 节点总数
    返回:
        vtoe_pointer: shape (n_nodes+1,) 每个顶点的元素起始指针
        vtoe_list: shape (n_entries,) 包含该顶点的元素编号列表
    """
    n_elements = len(elements)
    n_vertices_per_elem = elements.shape[1]

    # 步骤1：创建(vertex, element)对
    pairs = []
    for elem_idx in range(n_elements):
        for local_vert in range(n_vertices_per_elem):
            global_vert = elements[elem_idx, local_vert]
            pairs.append((global_vert, elem_idx))

    pairs = np.array(pairs)

    # 步骤2：按顶点编号排序（稳定排序保持原始顺序）
    sort_idx = np.lexsort((pairs[:, 1], pairs[:, 0]))
    pairs_sorted = pairs[sort_idx]

    # 步骤3：构建指针数组
    vtoe_pointer = np.zeros(n_nodes + 1, dtype=np.int32)
    vtoe_list = pairs_sorted[:, 1].astype(np.int32)

    # 计算每个顶点的元素数量
    for i in range(len(pairs_sorted)):
        v = pairs_sorted[i, 0]
        if v < n_nodes:
            vtoe_pointer[v + 1] += 1

    # 前缀和
    for i in range(1, n_nodes + 1):
        vtoe_pointer[i] += vtoe_pointer[i - 1]

    return vtoe_pointer, vtoe_list


# =============================================================================
# Hilbert曲线排序（来自535_hilbert_curve）
# =============================================================================
def xy_to_hilbert(x: int, y: int, order: int) -> int:
    """
    将2D坐标转换为Hilbert曲线参数d

    Hilbert空间填充曲线保持空间局部性，用于优化
    网格遍历的缓存命中率

    算法使用位操作进行象限旋转和映射

    参数:
        x, y: 网格坐标 (非负整数)
        order: Hilbert曲线阶数 (2^order x 2^order 网格)
    返回:
        d: Hilbert曲线参数值
    """
    d = 0
    n = 1 << order  # n = 2^order

    s = n >> 1
    while s > 0:
        rx = 1 if (x & s) > 0 else 0
        ry = 1 if (y & s) > 0 else 0

        d += s * s * ((3 * rx) ^ ry)

        # 旋转
        if ry == 0:
            if rx == 1:
                x = n - 1 - x
                y = n - 1 - y
            x, y = y, x

        s >>= 1

    return d


def hilbert_order_elements(elements: np.ndarray, nodes: np.ndarray,
                            grid_size: int = 16) -> np.ndarray:
    """
    使用Hilbert曲线对有限元进行排序，优化缓存局部性

    参数:
        elements: shape (n_elements, n_verts) 单元连接表
        nodes: shape (n_nodes, 2) 节点坐标
        grid_size: Hilbert网格分辨率（必须是2的幂）
    返回:
        sorted_indices: 排序后的元素索引
    """
    # 确定 Hilbert 阶数
    order = max(1, int(np.log2(grid_size)))
    n = 1 << order

    # 节点坐标映射到整数网格
    x_min, x_max = nodes[:, 0].min(), nodes[:, 0].max()
    y_min, y_max = nodes[:, 1].min(), nodes[:, 1].max()

    x_range = x_max - x_min + 1e-10
    y_range = y_max - y_min + 1e-10

    n_elements = len(elements)
    hilbert_keys = np.zeros(n_elements, dtype=np.int64)

    for e in range(n_elements):
        # 单元中心坐标
        elem_nodes = nodes[elements[e]]
        cx = np.mean(elem_nodes[:, 0])
        cy = np.mean(elem_nodes[:, 1])

        # 映射到整数网格
        ix = int((cx - x_min) / x_range * (n - 1))
        iy = int((cy - y_min) / y_range * (n - 1))
        ix = min(max(ix, 0), n - 1)
        iy = min(max(iy, 0), n - 1)

        hilbert_keys[e] = xy_to_hilbert(ix, iy, order)

    sorted_indices = np.argsort(hilbert_keys)
    return sorted_indices


# =============================================================================
# 偏滤器靶板网格（沿靶板方向的一维/二维网格）
# =============================================================================
def generate_divertor_target_mesh(n_toroidal: int = 32,
                                   n_poloidal: int = 8,
                                   R_inner: float = 1.2,
                                   R_outer: float = 1.8,
                                   Z_target: float = -0.8) -> Dict[str, np.ndarray]:
    """
    生成偏滤器靶板的二维表面网格

    用于计算靶板热负荷的空间分布

    参数:
        n_toroidal: 环向分辨率
        n_poloidal: 极向分辨率
        R_inner, R_outer: 靶板内外径 [m]
        Z_target: 靶板Z位置 [m]
    返回:
        dict包含靶板网格信息
    """
    R = np.linspace(R_inner, R_outer, n_toroidal)
    Z = np.full(n_toroidal, Z_target)

    # 靶板面积元素
    dR = (R_outer - R_inner) / (n_toroidal - 1)

    # 热负荷分布初始化为均匀
    q_profile = np.zeros(n_toroidal)

    return {
        'R_target': R,
        'Z_target': Z,
        'dR': dR,
        'q_profile': q_profile,
        'n_toroidal': n_toroidal,
        'n_poloidal': n_poloidal,
    }


# =============================================================================
# 场线追踪（沿磁力线的积分路径）
# =============================================================================
def trace_field_line(R_start: float, Z_start: float,
                      R_major: float = 1.5, B0: float = 5.3,
                      n_steps: int = 100, ds: float = 0.05) -> np.ndarray:
    """
    沿磁力线追踪（简化的圆截面tokamak）

    使用简单的 Euler 方法沿磁场方向步进

    B_R = -B0 * R0/R * (Z-Z0) / (kappa*a)  (极向分量)
    B_Z = B0 * R0/R * (R-R0) / a  (极向分量)
    B_phi = B0 * R0/R  (环向分量)

    参数:
        R_start, Z_start: 起始位置
        R_major: 大半径
        B0: 磁场强度
        n_steps: 步数
        ds: 步长 [m]
    返回:
        field_line: shape (n_steps+1, 3) 的 (R, Z, phi) 轨迹
    """
    a = 0.5
    kappa = 1.6

    trajectory = np.zeros((n_steps + 1, 3))
    trajectory[0] = [R_start, Z_start, 0.0]

    R, Z, phi = R_start, Z_start, 0.0

    for step in range(n_steps):
        # 局部磁场方向（归一化）
        B_phi = B0 * R_major / R
        B_R = -B0 * R_major * (Z - 0.0) / (kappa * a * R)
        B_Z = B0 * R_major * (R - R_major) / (a * R)

        B_mag = np.sqrt(B_R**2 + B_Z**2 + B_phi**2 + 1e-30)

        # 单位方向向量
        b_R = B_R / B_mag
        b_Z = B_Z / B_mag
        b_phi = B_phi / B_mag

        # Euler步进
        R += ds * b_R
        Z += ds * b_Z
        phi += ds * b_phi / (R + 1e-10)

        # 边界检查
        R = max(R, 0.3)
        R = min(R, 3.0)
        Z = max(Z, -2.0)
        Z = min(Z, 2.0)

        trajectory[step + 1] = [R, Z, phi]

    return trajectory


# =============================================================================
# 网格质量检查
# =============================================================================
def check_mesh_quality(nodes: np.ndarray, elements: np.ndarray) -> Dict[str, float]:
    """
    检查有限元网格质量

    参数:
        nodes: 节点坐标
        elements: 单元连接
    返回:
        dict包含网格质量指标
    """
    n_elements = len(elements)
    min_area = float('inf')
    max_aspect = 0.0
    n_degenerate = 0

    for e in range(n_elements):
        verts = nodes[elements[e]]
        if len(verts) < 3:
            continue

        # 三角形面积
        v0 = verts[0]
        v1 = verts[1]
        v2 = verts[2]

        area = 0.5 * abs((v1[0] - v0[0]) * (v2[1] - v0[1]) -
                          (v2[0] - v0[0]) * (v1[1] - v0[1]))

        if area < 1e-20:
            n_degenerate += 1
        min_area = min(min_area, area)

        # 边长比（aspect ratio）
        edges = [
            np.linalg.norm(v1 - v0),
            np.linalg.norm(v2 - v1),
            np.linalg.norm(v0 - v2),
        ]
        if min(edges) > 1e-20:
            aspect = max(edges) / min(edges)
            max_aspect = max(max_aspect, aspect)

    return {
        'n_elements': n_elements,
        'min_area': min_area,
        'max_aspect_ratio': max_aspect,
        'n_degenerate': n_degenerate,
        'quality_ok': n_degenerate == 0 and max_aspect < 10.0,
    }
