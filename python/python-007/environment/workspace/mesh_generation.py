"""
网格生成与管理模块
整合自：
  - 1320_triangle_to_fem（TRIANGLE格式转FEM格式）
  - 1305_triangle_grid（三角形网格生成）
  - 1339_triangulation_mask（三角剖分掩码）

在吸积盘模拟中用于：
  1. 生成吸积盘截面（r-z平面）的三角形网格
  2. 对网格进行掩码处理（排除喷流区域或黑洞视界内区域）
  3. 转换为FEM格式用于有限元求解
"""
import numpy as np


# ===========================
# Triangle Grid Generation
# ===========================

def triangle_grid_count(n_subdivisions):
    """
    计算三角形网格的节点数。
    对于 N 次细分，节点数为三角数：
        N_g = (N+1)(N+2) / 2
    """
    n = int(n_subdivisions)
    if n < 0:
        raise ValueError("n_subdivisions must be non-negative")
    return (n + 1) * (n + 2) // 2


def triangle_grid(n_subdivisions, vertices):
    """
    在给定三角形内生成规则网格点（重心坐标细分）。

    对于三角形的三个顶点 V1, V2, V3，网格点为：
        P = (i*V1 + j*V2 + k*V3) / N
    其中 i+j+k = N, i,j,k >= 0。

    参数:
        n_subdivisions: 细分次数 N
        vertices: (3, 2) 顶点坐标数组

    返回:
        points: (N_g, 2) 网格点坐标
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    if vertices.shape != (3, 2):
        raise ValueError("vertices must be (3, 2) array")

    N = int(n_subdivisions)
    if N < 0:
        raise ValueError("n_subdivisions must be non-negative")

    count = triangle_grid_count(N)
    points = np.zeros((count, 2), dtype=np.float64)

    idx = 0
    for i in range(N + 1):
        for j in range(N + 1 - i):
            k = N - i - j
            points[idx] = (i * vertices[0] + j * vertices[1] + k * vertices[2]) / N
            idx += 1

    return points


def generate_disk_cross_section_mesh(r_in, r_out, z_max, n_r=20, n_z=10):
    """
    生成吸积盘截面（r-z平面）的三角形网格。
    将环形区域 [r_in, r_out] x [-z_max, z_max] 划分为三角形。

    参数:
        r_in: 内半径（如 ISCO 半径）
        r_out: 外半径
        z_max: 最大半厚度
        n_r, n_z: 径向和垂直方向的网格数

    返回:
        nodes: (n_nodes, 2) 节点坐标 [r, z]
        elements: (n_elements, 3) 三角形单元（节点索引）
    """
    if r_in <= 0 or r_out <= r_in or z_max <= 0:
        raise ValueError("Invalid geometry parameters")

    # 生成矩形网格
    r_nodes = np.linspace(r_in, r_out, n_r + 1)
    z_nodes = np.linspace(-z_max, z_max, n_z + 1)

    # 构建节点列表
    nodes = []
    node_map = {}
    for i, r in enumerate(r_nodes):
        for j, z in enumerate(z_nodes):
            nodes.append([r, z])
            node_map[(i, j)] = len(nodes) - 1

    nodes = np.array(nodes, dtype=np.float64)

    # 将每个矩形划分为两个三角形
    elements = []
    for i in range(n_r):
        for j in range(n_z):
            n1 = node_map[(i, j)]
            n2 = node_map[(i + 1, j)]
            n3 = node_map[(i, j + 1)]
            n4 = node_map[(i + 1, j + 1)]

            # 两个三角形
            elements.append([n1, n2, n3])
            elements.append([n2, n4, n3])

    elements = np.array(elements, dtype=np.int64)

    return nodes, elements


# ===========================
# Triangulation Mask
# ===========================

def triangulation_mask(nodes, elements, mask_func):
    """
    对三角剖分应用掩码函数，删除满足掩码条件的三角形和孤立节点。

    参数:
        nodes: (n_nodes, dim) 节点坐标
        elements: (n_elements, 3) 三角形单元
        mask_func: callable, 输入三角形坐标，返回 True 表示删除

    返回:
        filtered_nodes: 过滤后的节点
        filtered_elements: 过滤后的单元
    """
    nodes = np.asarray(nodes, dtype=np.float64)
    elements = np.asarray(elements, dtype=np.int64)

    if elements.shape[1] != 3:
        raise ValueError("Elements must have 3 nodes per triangle")

    n_elements = elements.shape[0]
    keep = np.ones(n_elements, dtype=bool)

    for e in range(n_elements):
        tri_nodes = elements[e]
        if np.any(tri_nodes < 0) or np.any(tri_nodes >= nodes.shape[0]):
            keep[e] = False
            continue

        coords = nodes[tri_nodes]
        if mask_func(coords):
            keep[e] = False

    # 保留的单元
    filtered_elements = elements[keep].copy()

    # 找出仍在使用的节点
    used_nodes = np.unique(filtered_elements)
    if len(used_nodes) == 0:
        return np.zeros((0, nodes.shape[1])), np.zeros((0, 3), dtype=np.int64)

    # 重编号
    new_index = {old: new for new, old in enumerate(used_nodes)}
    for e in range(filtered_elements.shape[0]):
        for k in range(3):
            filtered_elements[e, k] = new_index[filtered_elements[e, k]]

    filtered_nodes = nodes[used_nodes].copy()

    return filtered_nodes, filtered_elements


# ===========================
# FEM Mesh Conversion
# ===========================

def mesh_to_fem_format(nodes, elements, attributes=None):
    """
    将网格数据转换为 FEM 格式字典。

    返回字典包含：
        - nodes: 节点坐标
        - elements: 单元连接
        - element_attributes: 单元属性（可选）
        - n_nodes: 节点数
        - n_elements: 单元数
        - node_dim: 空间维度
        - element_order: 单元阶数（3=线性三角形）
    """
    nodes = np.asarray(nodes, dtype=np.float64)
    elements = np.asarray(elements, dtype=np.int64)

    result = {
        'nodes': nodes,
        'elements': elements,
        'n_nodes': nodes.shape[0],
        'n_elements': elements.shape[0],
        'node_dim': nodes.shape[1],
        'element_order': elements.shape[1]
    }

    if attributes is not None:
        result['element_attributes'] = np.asarray(attributes, dtype=np.float64)
    else:
        result['element_attributes'] = np.zeros(elements.shape[0], dtype=np.float64)

    return result


def compute_triangle_area(nodes, element):
    """
    计算三角形的面积（2D或3D）。
    对于2D: A = 0.5*|x1(y2-y3) + x2(y3-y1) + x3(y1-y2)|
    """
    coords = nodes[element]
    if coords.shape[1] == 2:
        x1, y1 = coords[0]
        x2, y2 = coords[1]
        x3, y3 = coords[2]
        return 0.5 * abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    else:
        # 3D: 叉积的一半
        v1 = coords[1] - coords[0]
        v2 = coords[2] - coords[0]
        return 0.5 * np.linalg.norm(np.cross(v1, v2))


def mask_disk_jet_region(nodes, elements, r_jet, z_jet_threshold):
    """
    掩码掉吸积盘网格中位于喷流区域的三角形。
    喷流区域定义为：r < r_jet 且 |z| > z_jet_threshold。

    参数:
        nodes: (n, 2) [r, z]
        elements: (m, 3)
        r_jet: 喷流起始半径
        z_jet_threshold: 喷流垂直阈值

    返回:
        过滤后的 nodes, elements
    """
    def is_jet(coords):
        # 三角形重心
        centroid = np.mean(coords, axis=0)
        r_c, z_c = centroid
        return (r_c < r_jet) and (abs(z_c) > z_jet_threshold)

    return triangulation_mask(nodes, elements, is_jet)


def mask_black_hole_horizon(nodes, elements, r_isco):
    """
    掩码掉位于黑洞视界/ISCO半径内的网格单元。
    """
    def is_inside_horizon(coords):
        centroid = np.mean(coords, axis=0)
        r_c = centroid[0]
        return r_c < r_isco

    return triangulation_mask(nodes, elements, is_inside_horizon)
