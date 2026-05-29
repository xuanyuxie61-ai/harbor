"""
field_interp.py — 三角化网格上的标量场插值与分析

融合以下种子项目：
- 1342_triangulation_order3_contour : 三阶三角化上的等高线/标量场分析
- 1310_triangle_io : 三角网格数据读取

功能：
1. 在三角化球面/球壳网格上进行线性插值
2. 计算标量场的梯度、拉普拉斯
3. 在网格上计算面积分与体积分
4. 标量场的极值搜索与临界点检测

核心数学模型：
-------------
四面体内线性插值：
  给定四面体顶点 p₀,p₁,p₂,p₃ 和对应值 v₀,v₁,v₂,v₃，
  对任意点 p = p₀ + ξ(p₁-p₀) + η(p₂-p₀) + ζ(p₃-p₀)，
  有 v(p) = v₀ + ξ(v₁-v₀) + η(v₂-v₀) + ζ(v₃-v₀)。

重心坐标：通过求解线性方程组
  [p₁-p₀, p₂-p₀, p₃-p₀] [ξ, η, ζ]^T = p - p₀

体积积分：
  ∫_Ω f dV ≈ Σ_{单元 e} (f₀+f₁+f₂+f₃)/4 · Vol(e)

面积分（球面上）：
  ∫_S f dS ≈ Σ_{三角形 t} (f₀+f₁+f₂)/3 · Area(t)
"""

import numpy as np


def barycentric_coordinates_tetrahedron(p, p0, p1, p2, p3):
    """
    计算点 p 在四面体 (p0,p1,p2,p3) 中的重心坐标 (λ0,λ1,λ2,λ3)。
    """
    v0 = p1 - p0
    v1 = p2 - p0
    v2 = p3 - p0
    vp = p - p0

    mat = np.column_stack((v0, v1, v2))
    try:
        coords = np.linalg.solve(mat, vp)
    except np.linalg.LinAlgError:
        return None

    lam1, lam2, lam3 = coords
    lam0 = 1.0 - lam1 - lam2 - lam3
    return np.array([lam0, lam1, lam2, lam3])


def interpolate_in_tetrahedron(p, p0, p1, p2, p3, v0, v1, v2, v3):
    """
    在四面体内进行线性插值。
    """
    lams = barycentric_coordinates_tetrahedron(p, p0, p1, p2, p3)
    if lams is None:
        return None
    if np.any(lams < -0.01) or np.any(lams > 1.01):
        return None
    return lams[0] * v0 + lams[1] * v1 + lams[2] * v2 + lams[3] * v3


def find_containing_tetrahedron(point, nodes, elements, search_radius=0.5):
    """
    查找包含给定点的四面体单元索引。
    采用空间预筛选 + 重心坐标检测。
    """
    # 预筛选：只检查与点距离较近的单元中心
    candidates = []
    for idx, elem in enumerate(elements):
        pts = nodes[elem]
        center = np.mean(pts, axis=0)
        if np.linalg.norm(center - point) < search_radius:
            candidates.append(idx)

    for idx in candidates:
        elem = elements[idx]
        p0, p1, p2, p3 = nodes[elem[0]], nodes[elem[1]], nodes[elem[2]], nodes[elem[3]]
        lams = barycentric_coordinates_tetrahedron(point, p0, p1, p2, p3)
        if lams is not None and np.all(lams >= -1e-6) and np.all(lams <= 1 + 1e-6):
            return idx, lams

    # 回退：最近单元
    if len(candidates) > 0:
        return candidates[0], None
    return None, None


def scalar_field_interpolator(nodes, elements, values):
    """
    构建标量场插值函数。
    返回 func(point) -> interpolated_value。
    """
    def interp(point):
        idx, lams = find_containing_tetrahedron(point, nodes, elements)
        if idx is None:
            # 回退到最近节点
            dists = np.linalg.norm(nodes - point, axis=1)
            nearest = np.argmin(dists)
            return values[nearest]

        if lams is not None:
            elem = elements[idx]
            return lams[0] * values[elem[0]] + lams[1] * values[elem[1]] + \
                   lams[2] * values[elem[2]] + lams[3] * values[elem[3]]

        elem = elements[idx]
        pts = nodes[elem]
        dists = np.linalg.norm(pts - point, axis=1)
        weights = 1.0 / (dists + 1e-15)
        weights /= np.sum(weights)
        return np.dot(weights, values[elem])

    return interp


def compute_volume_integral(nodes, elements, values):
    """
    在四面体网格上计算体积分 ∫ f dV。
    """
    if elements.size == 0:
        return np.mean(values) * len(nodes) * 0.001

    total = 0.0
    total_vol = 0.0
    for elem in elements:
        p0, p1, p2, p3 = nodes[elem[0]], nodes[elem[1]], nodes[elem[2]], nodes[elem[3]]
        v0 = p1 - p0
        v1 = p2 - p0
        v2 = p3 - p0
        vol = abs(np.dot(v0, np.cross(v1, v2))) / 6.0
        f_avg = np.mean(values[elem])
        total += f_avg * vol
        total_vol += vol

    return total, total_vol


def compute_surface_integral_sphere(nodes, elements_tri, values, r_target=1.0):
    """
    在三角形网格上计算球面积分 ∫ f dS。
    elements_tri: (M,3) 三角形索引。
    """
    if elements_tri.size == 0:
        return 0.0, 0.0

    total = 0.0
    total_area = 0.0
    for tri in elements_tri:
        p0, p1, p2 = nodes[tri[0]], nodes[tri[1]], nodes[tri[2]]
        area = 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0))
        f_avg = np.mean(values[tri])
        total += f_avg * area
        total_area += area

    return total, total_area


def extract_surface_triangles(elements):
    """
    从四面体单元中提取表面三角形（边界面）。
    采用面频统计法：仅出现一次的四面体表面为外表面。
    """
    if elements.size == 0:
        return np.array([])

    face_count = {}
    face_to_tet = {}

    for ei, elem in enumerate(elements):
        faces = [
            tuple(sorted([elem[0], elem[1], elem[2]])),
            tuple(sorted([elem[0], elem[1], elem[3]])),
            tuple(sorted([elem[0], elem[2], elem[3]])),
            tuple(sorted([elem[1], elem[2], elem[3]])),
        ]
        for face in faces:
            face_count[face] = face_count.get(face, 0) + 1

    surface_faces = [list(face) for face, count in face_count.items() if count == 1]
    return np.array(surface_faces, dtype=int)


def gradient_recovery_superconvergent(nodes, elements, values):
    """
    超收敛梯度恢复（Zienkiewicz-Zhu 型）。
    在每个节点处，用周围单元梯度加权平均得到更光滑的梯度场。
    """
    n_nodes = len(nodes)
    grad_sum = np.zeros((n_nodes, 3))
    weight_sum = np.zeros(n_nodes)

    if elements.size == 0:
        return grad_sum

    for elem in elements:
        p0, p1, p2, p3 = nodes[elem[0]], nodes[elem[1]], nodes[elem[2]], nodes[elem[3]]
        vals = values[elem]

        # 四面体内的常梯度
        mat = np.column_stack((p1 - p0, p2 - p0, p3 - p0))
        rhs = np.array([vals[1] - vals[0], vals[2] - vals[0], vals[3] - vals[0]])
        try:
            grad_elem = np.linalg.solve(mat, rhs)
        except np.linalg.LinAlgError:
            grad_elem = np.zeros(3)

        # 体积权重
        vol = abs(np.dot(p1 - p0, np.cross(p2 - p0, p3 - p0))) / 6.0
        for idx in elem:
            grad_sum[idx] += grad_elem * vol
            weight_sum[idx] += vol

    for i in range(n_nodes):
        if weight_sum[i] > 0:
            grad_sum[i] /= weight_sum[i]

    return grad_sum


def find_critical_points(nodes, elements, values):
    """
    在三角化网格上搜索标量场的临界点（极大值、极小值、鞍点近似）。
    返回临界点列表 [(node_index, type, value), ...]。
    """
    n_nodes = len(nodes)
    critical = []

    for i in range(n_nodes):
        # 寻找相邻节点
        neighbors = set()
        if elements.size > 0:
            for elem in elements:
                if i in elem:
                    for idx in elem:
                        if idx != i:
                            neighbors.add(idx)

        if len(neighbors) == 0:
            continue

        vals_neighbor = [values[j] for j in neighbors]
        val_i = values[i]

        if val_i > max(vals_neighbor):
            critical.append((i, 'maximum', val_i))
        elif val_i < min(vals_neighbor):
            critical.append((i, 'minimum', val_i))
        else:
            # 检查是否为鞍点：比某些邻居大，比另一些邻居小
            has_higher = any(val_i > v for v in vals_neighbor)
            has_lower = any(val_i < v for v in vals_neighbor)
            if has_higher and has_lower:
                critical.append((i, 'saddle', val_i))

    return critical
