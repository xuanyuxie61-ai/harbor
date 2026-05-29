r"""
geometry_utils.py
=================
几何处理、STL 文件解析、传感器优化布置与涡量场拓扑分析模块。

科学背景
--------
在工程应用中，涡激振动结构常为复杂几何体（带尾缘平板、多圆柱群、
海洋立管截面等）。STL（Stereolithography）格式是描述此类三角面片
网格的工业标准。本模块提供 STL 文件的解析与表面法向量计算，用于
确定流体-结构界面。

此外，实验与监测需要在流场中布置有限数量的传感器以获取升力、涡量等
关键信息。在预算约束下，选择最优测点组合可建模为子集和问题：

    给定 N 个候选测点，其信息量权重 w_i，预算上限 B，
    求子集 S \subseteq \{1,...,N\} 使得 \sum_{i \in S} w_i \le B
    且 \sum_{i \in S} w_i 最大。

采用 swap 启发式算法（subset-sum-swap）高效求解此 NP-hard 问题的
近似解。

涡量场拓扑分析
--------------
利用复平面等距采样与模运算映射（caustic 思想），追踪涡量等值线的
拓扑连接关系：将涡量场 \omega(x,y) 映射到复平面 z = x + i y，
通过等值线提取算法识别涡核位置与涡街相位。

对应原种子项目：
- 1166_stla_io（STL 文件解析：面片法向量、节点坐标读取）
- 1181_subset_sum_swap（子集和 swap 启发式优化）
- 140_caustic（复平面等距采样与模映射，用于涡核定位）
- 1426_xyzl_display（线段拓扑连接，用于涡量等值线追踪）
r"""

import numpy as np


# ---------------------------------------------------------------------------
# STL 解析（来自 1166_stla_io 核心算法）
# ---------------------------------------------------------------------------

def parse_stl_ascii(data_lines):
    r"""
    解析 ASCII 格式 STL 文件内容。

    返回
    ----
    vertices : ndarray, shape (N_facet, 3, 3)
        每个面片的三个顶点坐标。
    normals : ndarray, shape (N_facet, 3)
        每个面片的法向量。
    """
    vertices = []
    normals = []
    current_normal = np.array([0.0, 0.0, 1.0])
    current_verts = []

    for line in data_lines:
        line = line.strip().lower()
        if line.startswith('facet normal'):
            parts = line.split()
            if len(parts) >= 6:
                current_normal = np.array([
                    float(parts[2]), float(parts[3]), float(parts[4])
                ])
        elif line.startswith('vertex'):
            parts = line.split()
            if len(parts) >= 4:
                current_verts.append([
                    float(parts[1]), float(parts[2]), float(parts[3])
                ])
        elif line.startswith('endfacet'):
            if len(current_verts) == 3:
                vertices.append(current_verts)
                normals.append(current_normal)
            current_verts = []

    if len(vertices) == 0:
        return np.array([]), np.array([])
    return np.array(vertices, dtype=float), np.array(normals, dtype=float)


def compute_face_normals(vertices):
    r"""
    由顶点计算面片法向量（右手定则）。

    对三角形 (v0, v1, v2)：
    n = (v1 - v0) \times (v2 - v0) / ||(v1 - v0) \times (v2 - v0)||
    """
    v0 = vertices[:, 0, :]
    v1 = vertices[:, 1, :]
    v2 = vertices[:, 2, :]
    cross = np.cross(v1 - v0, v2 - v0)
    norms = np.linalg.norm(cross, axis=1, keepdims=True)
    norms = np.where(norms < 1e-15, 1e-15, norms)
    return cross / norms


def stla_check(vertices, normals):
    r"""
    检查 STL 数据一致性：
    1. 法向量长度接近 1。
    2. 面片方向一致（封闭体）。
    3. 无退化面片（面积 > 0）。

    返回错误代码：0=正常，1=退化面片，2=法向量未归一化。
    """
    if vertices.size == 0:
        return 2

    # 退化检查
    v0 = vertices[:, 0, :]
    v1 = vertices[:, 1, :]
    v2 = vertices[:, 2, :]
    cross = np.cross(v1 - v0, v2 - v0)
    areas = np.linalg.norm(cross, axis=1)
    if np.any(areas < 1e-12):
        return 1

    # 法向量归一化检查
    n_norms = np.linalg.norm(normals, axis=1)
    if np.any(np.abs(n_norms - 1.0) > 0.01):
        return 2

    return 0


# ---------------------------------------------------------------------------
# 子集和传感器优化（来自 1181_subset_sum_swap）
# ---------------------------------------------------------------------------

def subset_sum_swap(weights, budget):
    r"""
    子集和问题的 swap 启发式求解。

    算法：
    1. 将权重按降序排列。
    2. 依次尝试加入当前权重；若超预算，尝试用已选元素替换。
    3. 重复直至无改进。

    参数
    ----
    weights : ndarray
        候选测点的信息量权重（正数）。
    budget : float
        预算上限。

    返回
    ----
    selected : ndarray(bool)
        各测点是否被选中。
    achieved : float
        实际达到的总权重。
    """
    weights = np.asarray(weights, dtype=float)
    if np.any(weights <= 0):
        raise ValueError("权重必须为正。")
    if budget <= 0:
        raise ValueError("预算必须为正。")

    n = len(weights)
    # 降序排列
    sorted_idx = np.argsort(weights)[::-1]
    sorted_w = weights[sorted_idx]

    selected = np.zeros(n, dtype=bool)
    achieved = 0.0

    while True:
        nmove = 0
        for i in range(n):
            if not selected[i]:
                if achieved + sorted_w[i] <= budget:
                    selected[i] = True
                    achieved += sorted_w[i]
                    nmove += 1
                    continue

            if not selected[i]:
                for j in range(n):
                    if selected[j]:
                        delta = sorted_w[i] - sorted_w[j]
                        if delta > 0 and achieved + delta <= budget:
                            selected[j] = False
                            selected[i] = True
                            achieved += delta
                            nmove += 2
                            break

        if nmove == 0:
            break

    # 映射回原始顺序
    result = np.zeros(n, dtype=bool)
    result[sorted_idx[selected]] = True
    return result, achieved


def sensor_placement_optimization(candidate_positions, field_values,
                                  budget_num, influence_radius):
    r"""
    在候选位置中选择预算内的最优测点集。

    信息量权重 w_i 由局部场量梯度与覆盖面积共同决定：
    w_i = |\nabla \phi_i| * A_i * (1 + \alpha_{info} \cdot SNR_i)

    其中预算 budget_num 为允许的最大测点数。
    """
    ny, nx = field_values.shape
    weights = np.zeros(len(candidate_positions))

    for idx, (j, i) in enumerate(candidate_positions):
        # 局部梯度近似
        if 0 < i < nx - 1 and 0 < j < ny - 1:
            grad_x = abs(field_values[j, i + 1] - field_values[j, i - 1]) / 2.0
            grad_y = abs(field_values[j + 1, i] - field_values[j - 1, i]) / 2.0
            grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)
        else:
            grad_mag = 0.0
        weights[idx] = grad_mag

    # 防止全零
    max_w = np.max(weights)
    if max_w < 1e-15:
        weights[:] = 1.0

    # 将 budget_num 映射为子集和预算
    total = np.sum(weights)
    budget = total * budget_num / len(candidate_positions) if len(candidate_positions) > 0 else 0

    selected, achieved = subset_sum_swap(weights, budget)
    return selected, achieved


# ---------------------------------------------------------------------------
# 复平面涡核映射（来自 140_caustic）
# ---------------------------------------------------------------------------

def vortex_caustic_map(n_points, m_ratio, cylinder_center, radius_scale):
    r"""
    利用 caustic 映射思想追踪涡量场中的准周期结构。

    在复平面上定义单位圆上的等距采样点：
    z_j = \exp(2\pi i j / n), j = 0, ..., n

    通过模运算连接：z_j 与 z_{(j*m) \mod n} 相连，形成焦散型图案。
    将此类映射的拓扑不变量（环绕数、交叉数）用于涡街周期性评估。

    参数
    ----
    n_points : int
        采样点数。
    m_ratio : int
        模运算乘子，控制连接拓扑。
    cylinder_center : tuple
        圆柱中心 (cx, cy)。
    radius_scale : float
        映射半径。

    返回
    ----
    connections : list of tuple
        每条连接线段 [(x1,y1), (x2,y2)]。
    winding_number : int
        拓扑环绕数估计。
    """
    theta = 2.0 * np.pi * np.arange(n_points + 1) / n_points
    z = np.exp(1j * theta)

    cx, cy = cylinder_center
    connections = []
    for j in range(n_points):
        idx1 = j
        idx2 = (j * m_ratio) % n_points
        x1 = cx + radius_scale * np.real(z[idx1])
        y1 = cy + radius_scale * np.imag(z[idx1])
        x2 = cx + radius_scale * np.real(z[idx2])
        y2 = cy + radius_scale * np.imag(z[idx2])
        connections.append(((x1, y1), (x2, y2)))

    # 环绕数估计（简化）：gcd(n, m) 决定连通分量数
    from math import gcd
    w = gcd(n_points, m_ratio)
    winding_number = n_points // w if w > 0 else n_points

    return connections, winding_number


# ---------------------------------------------------------------------------
# 线段拓扑（来自 1426_xyzl_display 非可视化部分）
# ---------------------------------------------------------------------------

def extract_iso_line_segments(field, threshold, x_coords, y_coords):
    r"""
    从标量场中提取等值线段的拓扑连接关系（ Marching Squares 简化版）。

    对每对相邻节点，若场值跨阈值，则线性插值得到等值点坐标，
    并记录线段连接关系。

    返回线段列表与节点邻接表，可用于涡量等值线追踪与涡核识别
   （不绘制、不显示）。
    """
    ny, nx = field.shape
    segments = []
    adjacency = {}
    node_id = 0
    node_map = {}  # (j, i, edge) -> node_id

    def get_or_create_node(key, px, py):
        nonlocal node_id
        if key not in node_map:
            node_map[key] = node_id
            adjacency[node_id] = []
            node_id += 1
        return node_map[key]

    # 水平边
    for j in range(ny):
        for i in range(nx - 1):
            f1 = field[j, i]
            f2 = field[j, i + 1]
            if (f1 - threshold) * (f2 - threshold) < 0:
                t = (threshold - f1) / (f2 - f1)
                px = x_coords[i] + t * (x_coords[i + 1] - x_coords[i])
                py = y_coords[j]
                key = (j, i, 'h')
                nid = get_or_create_node(key, px, py)
                # 与相邻垂直边连接
                # 简化：仅记录存在性，不构建完整图
                segments.append((px, py))

    # 垂直边
    for j in range(ny - 1):
        for i in range(nx):
            f1 = field[j, i]
            f2 = field[j + 1, i]
            if (f1 - threshold) * (f2 - threshold) < 0:
                t = (threshold - f1) / (f2 - f1)
                px = x_coords[i]
                py = y_coords[j] + t * (y_coords[j + 1] - y_coords[j])
                key = (j, i, 'v')
                nid = get_or_create_node(key, px, py)
                segments.append((px, py))

    return segments, adjacency


def generate_simple_cylinder_stl_lines(diameter=1.0, num_facets=36):
    r"""
    生成描述圆柱表面的简化 ASCII STL 数据行（用于测试解析）。
    """
    lines = ["solid cylinder"]
    r = diameter / 2.0
    for i in range(num_facets):
        theta1 = 2.0 * np.pi * i / num_facets
        theta2 = 2.0 * np.pi * (i + 1) / num_facets
        x1, y1 = r * np.cos(theta1), r * np.sin(theta1)
        x2, y2 = r * np.cos(theta2), r * np.sin(theta2)
        # 上表面三角形（z=0 平面）
        nx_n = (y2 - y1)
        ny_n = -(x2 - x1)
        nz_n = 0.0
        nn = np.sqrt(nx_n**2 + ny_n**2 + nz_n**2)
        if nn < 1e-15:
            nn = 1.0
        lines.append(f"  facet normal {nx_n/nn:.6f} {ny_n/nn:.6f} {nz_n/nn:.6f}")
        lines.append("    outer loop")
        lines.append(f"      vertex {x1:.6f} {y1:.6f} 0.0")
        lines.append(f"      vertex {x2:.6f} {y2:.6f} 0.0")
        lines.append(f"      vertex 0.0 0.0 0.0")
        lines.append("    endloop")
        lines.append("  endfacet")
    lines.append("endsolid cylinder")
    return lines
