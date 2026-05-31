"""
tumor_geometry.py

肿瘤几何建模与边界网格生成模块

本模块融合以下种子项目的核心算法：
  - 078_bernstein_polynomial: Bernstein 多项式参数化肿瘤边界
  - 1333_triangulation_boundary_nodes: 三角网格边界节点识别
  - 1168_stla_to_tri_surface_fast: 表面三角剖分快速生成

科学背景：
  肿瘤边界（Tumor Boundary）的精确几何描述是微环境建模的基础。
  我们采用 Bernstein 多项式参数化闭合曲线：

    B_{n,i}(t) = C(n,i) * t^i * (1-t)^{n-i},   t in [0,1]

  肿瘤轮廓由控制点 {P_i} 加权 Bernstein 基函数生成：

    C(t) = sum_{i=0}^{n} P_i * B_{n,i}(t)

  进而对肿瘤域进行 Delaunay 三角剖分，并识别边界节点以施加
  Dirichlet/Neumann 边界条件。
"""

import numpy as np
from typing import Tuple, List


def bernstein_poly_01(n: int, x: float) -> np.ndarray:
    """
    计算定义在 [0,1] 上的 n 次 Bernstein 多项式在点 x 处的值。

    数学公式：
        B(n,i)(x) = n! / (i! * (n-i)!) * (1-x)^{n-i} * x^i

    性质：
        - partition of unity: sum_i B(n,i)(x) = 1
        - 非负性: B(n,i)(x) >= 0
        - 端点插值: B(n,0)(0)=1, B(n,n)(1)=1

    参数:
        n: 多项式次数
        x: 计算点，必须在 [0,1] 区间内

    返回:
        bern: 长度为 n+1 的数组， bern[i] = B(n,i)(x)
    """
    if not (0.0 <= x <= 1.0):
        raise ValueError(f"bernstein_poly_01: x={x} 超出 [0,1] 区间")
    if n < 0:
        raise ValueError("bernstein_poly_01: n 必须非负")

    bern = np.zeros(n + 1)
    if n == 0:
        bern[0] = 1.0
        return bern

    # 递推初始化
    bern[0] = 1.0 - x
    bern[1] = x

    # 递推公式:
    #   对 i = 2..n:
    #     bern[i] = x * bern[i-1]
    #     对 j = i-1 .. 1:
    #       bern[j] = x * bern[j-1] + (1-x) * bern[j]
    #     bern[0] = (1-x) * bern[0]
    for i in range(2, n + 1):
        bern[i] = x * bern[i - 1]
        for j in range(i - 1, 0, -1):
            bern[j] = x * bern[j - 1] + (1.0 - x) * bern[j]
        bern[0] = (1.0 - x) * bern[0]

    return bern


def bernstein_tumor_boundary(
    control_points: np.ndarray, num_samples: int = 256
) -> np.ndarray:
    """
    使用 Bernstein 多项式参数化生成肿瘤闭合边界。

    参数:
        control_points: 形状为 (n+1, 2) 的控制点坐标数组
        num_samples: 边界采样点数

    返回:
        boundary_points: 形状为 (num_samples, 2) 的边界点坐标
    """
    n = control_points.shape[0] - 1
    if n < 1:
        raise ValueError("bernstein_tumor_boundary: 至少需要 2 个控制点")
    if control_points.shape[1] != 2:
        raise ValueError("bernstein_tumor_boundary: 控制点必须是二维坐标")

    t_vals = np.linspace(0.0, 1.0, num_samples)
    boundary_points = np.zeros((num_samples, 2))

    for idx, t in enumerate(t_vals):
        # 边界检查
        t_clamped = float(np.clip(t, 0.0, 1.0))
        bern = bernstein_poly_01(n, t_clamped)
        boundary_points[idx, :] = np.dot(bern, control_points)

    return boundary_points


def generate_delaunay_triangulation(
    boundary_points: np.ndarray, interior_points: np.ndarray = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    对肿瘤域进行约束 Delaunay 三角剖分。

    采用基于外接圆准则的简化实现：
      对点集 P，三角形 T 是 Delaunay 的当且仅当：
        对任意三角形 t in T，其外接圆内不包含 P 的其他点。

    参数:
        boundary_points: (B, 2) 边界点
        interior_points: (I, 2) 内部点，可为 None

    返回:
        nodes: (N, 2) 所有节点坐标
        triangles: (T, 3) 三角形节点索引（0-based）
    """
    if boundary_points.ndim != 2 or boundary_points.shape[1] != 2:
        raise ValueError("generate_delaunay_triangulation: 边界点必须为 (B,2)")

    if interior_points is None:
        interior_points = np.zeros((0, 2))

    nodes = np.vstack([boundary_points, interior_points])
    n_nodes = nodes.shape[0]
    n_boundary = boundary_points.shape[0]

    # 简化的 Bowyer-Watson 风格三角剖分
    # 首先构建超级三角形包围所有点
    min_xy = np.min(nodes, axis=0)
    max_xy = np.max(nodes, axis=0)
    dx, dy = max_xy - min_xy
    margin = max(dx, dy) * 2.0
    if margin <= 0:
        margin = 1.0

    super_tri = np.array([
        [min_xy[0] - margin, min_xy[1] - margin],
        [max_xy[0] + margin, min_xy[1] - margin],
        [min_xy[0] + 0.5 * dx, max_xy[1] + margin],
    ])

    # 初始三角形列表
    tri_list = [(n_nodes, n_nodes + 1, n_nodes + 2)]
    nodes_extended = np.vstack([nodes, super_tri])

    def _circumcircle_contains(tri, p):
        a, b, c = tri
        pts = nodes_extended[[a, b, c], :]
        ax, ay = pts[0]
        bx, by = pts[1]
        cx, cy = pts[2]

        d = 2.0 * ((ax - cx) * (by - cy) - (ay - cy) * (bx - cx))
        if abs(d) < 1e-14:
            return False

        ux = (((ax - cx) ** 2 + (ay - cy) ** 2) * (by - cy) -
              ((bx - cx) ** 2 + (by - cy) ** 2) * (ay - cy)) / d
        uy = (((bx - cx) ** 2 + (by - cy) ** 2) * (ax - cx) -
              ((ax - cx) ** 2 + (ay - cy) ** 2) * (bx - cx)) / d
        cx_c = cx + ux
        cy_c = cy + uy
        r2 = (cx_c - cx) ** 2 + (cy_c - cy) ** 2
        dist2 = (p[0] - cx_c) ** 2 + (p[1] - cy_c) ** 2
        return dist2 < r2 - 1e-12

    for p_idx in range(n_nodes):
        p = nodes_extended[p_idx]
        bad_tris = []
        for tri in tri_list:
            if _circumcircle_contains(tri, p):
                bad_tris.append(tri)

        # 收集 bad triangles 的边
        edges = []
        for tri in bad_tris:
            e = [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])]
            for edge in e:
                # 检查该边是否被其他 bad triangle 共享（反向顺序）
                is_shared = False
                for other in bad_tris:
                    if other is tri:
                        continue
                    if (edge[1], edge[0]) in [(other[0], other[1]),
                                               (other[1], other[2]),
                                               (other[2], other[0])]:
                        is_shared = True
                        break
                if not is_shared:
                    edges.append(edge)

        # 移除 bad triangles
        for bt in bad_tris:
            tri_list.remove(bt)

        # 用 p_idx 与每条边构成新三角形
        for edge in edges:
            tri_list.append((edge[0], edge[1], p_idx))

    # 移除包含超级三角形顶点的三角形
    filtered = []
    for tri in tri_list:
        if n_nodes not in tri and n_nodes + 1 not in tri and n_nodes + 2 not in tri:
            filtered.append(tri)

    triangles = np.array(filtered, dtype=int)
    return nodes, triangles


def detect_boundary_nodes(
    node_num: int, triangle_num: int, triangle_node: np.ndarray
) -> np.ndarray:
    """
    识别三角网格中的边界节点。

    算法原理：
      对 3 节点线性元，边界边只被一个三角形使用。
      统计所有边的出现次数，出现一次的边为边界边，其端点为边界节点。

    参数:
        node_num: 节点总数
        triangle_num: 三角形总数
        triangle_node: (3, T) 或 (T, 3) 三角形节点索引（1-based 或 0-based）

    返回:
        node_boundary: (node_num,) bool 数组，True 表示边界节点
    """
    if triangle_node.size == 0:
        return np.zeros(node_num, dtype=bool)

    # 统一为 (T, 3) 和 0-based
    if triangle_node.shape[0] == 3 and triangle_node.shape[1] == triangle_num:
        tri = triangle_node.T.copy()
    else:
        tri = triangle_node.copy()

    node_min = tri.min()
    node_max = tri.max()

    # 自动检测并纠正 0-based / 1-based 索引
    if node_min == 0 and node_max == node_num - 1:
        pass  # 已经是 0-based
    elif node_min == 1 and node_max == node_num:
        tri = tri - 1
    else:
        raise ValueError(
            f"detect_boundary_nodes: 节点索引范围 [{node_min}, {node_max}] "
            f"与节点总数 {node_num} 不匹配"
        )

    edge_count = {}
    for t in range(triangle_num):
        verts = tri[t, :]
        edges = [(verts[0], verts[1]),
                 (verts[1], verts[2]),
                 (verts[2], verts[0])]
        for e in edges:
            a, b = e
            if a > b:
                a, b = b, a
            edge_count[(a, b)] = edge_count.get((a, b), 0) + 1

    node_boundary = np.zeros(node_num, dtype=bool)
    for (a, b), count in edge_count.items():
        if count == 1:
            node_boundary[a] = True
            node_boundary[b] = True

    return node_boundary


def tumor_surface_triangulation(
    boundary_points: np.ndarray, interior_density: int = 16
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    生成肿瘤表面三角剖分（2D 域的平面三角网格）。

    基于 1168_stla_to_tri_surface_fast 思想，快速构建三角表面：
      1. 生成边界点
      2. 在内部填充均匀采样点
      3. Delaunay 三角剖分
      4. 识别并标记边界节点

    参数:
        boundary_points: (B, 2) 边界采样点
        interior_density: 内部采样密度参数

    返回:
        nodes: (N, 2) 节点坐标
        triangles: (T, 3) 三角形索引
        is_boundary: (N,) bool 边界标记
    """
    if boundary_points.shape[0] < 3:
        raise ValueError("tumor_surface_triangulation: 至少需要 3 个边界点")

    # 在内部生成采样点（简化实现：均匀网格 + 内部筛选）
    min_xy = np.min(boundary_points, axis=0)
    max_xy = np.max(boundary_points, axis=0)
    x_vals = np.linspace(min_xy[0], max_xy[0], interior_density)
    y_vals = np.linspace(min_xy[1], max_xy[1], interior_density)
    xx, yy = np.meshgrid(x_vals, y_vals)
    candidates = np.column_stack([xx.ravel(), yy.ravel()])

    # 使用绕数法判断点是否在多边形内部
    def _point_in_polygon(pt, poly):
        x, y = pt
        n = poly.shape[0]
        inside = False
        j = n - 1
        for i in range(n):
            xi, yi = poly[i]
            xj, yj = poly[j]
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-15) + xi):
                inside = not inside
            j = i
        return inside

    interior_points = []
    for pt in candidates:
        if _point_in_polygon(pt, boundary_points):
            # 距离边界足够远
            dists = np.linalg.norm(boundary_points - pt, axis=1)
            if np.min(dists) > 1e-3 * np.linalg.norm(max_xy - min_xy):
                interior_points.append(pt)

    interior = np.array(interior_points) if interior_points else np.zeros((0, 2))

    nodes, triangles = generate_delaunay_triangulation(boundary_points, interior)
    is_boundary = detect_boundary_nodes(nodes.shape[0], triangles.shape[0], triangles)
    return nodes, triangles, is_boundary


def compute_tumor_area(nodes: np.ndarray, triangles: np.ndarray) -> float:
    """
    通过三角网格计算肿瘤面积。

    对每个三角形 (a,b,c)，面积公式：
        A_t = 0.5 * | (b-a) x (c-a) |
    """
    area = 0.0
    for t in range(triangles.shape[0]):
        a, b, c = triangles[t, :]
        pa, pb, pc = nodes[a], nodes[b], nodes[c]
        cross = (pb[0] - pa[0]) * (pc[1] - pa[1]) - (pb[1] - pa[1]) * (pc[0] - pa[0])
        area += 0.5 * abs(cross)
    return area


def compute_tumor_perimeter(boundary_points: np.ndarray) -> float:
    """
    计算肿瘤边界周长。
    """
    n = boundary_points.shape[0]
    if n < 2:
        return 0.0
    perim = 0.0
    for i in range(n):
        j = (i + 1) % n
        perim += np.linalg.norm(boundary_points[j] - boundary_points[i])
    return perim
