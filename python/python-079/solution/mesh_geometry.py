"""
海洋平台网格生成与计算几何模块

基于种子项目：
  - 676_line_cvt_lloyd：一维 Lloyd 算法优化节点分布
  - 882_polygon：多边形面积、形心、惯性矩与三角剖分
  - 1335_triangulation_delaunay_discrepancy：Delaunay 三角剖分质量检查
  - 1080_simplex_integrals：单纯形上的精确积分

核心功能：
  1. 使用 Lloyd 算法在海洋平台水线面周围生成最优分布的观测/计算节点。
  2. 多边形几何计算（水线面面积、形心、惯性矩、固角），用于水静力学分析。
  3. Delaunay 三角剖分质量评估，确保 CFD/FEM 网格的数值稳定性。
  4. 单纯形（三角形/四面体）上的精确矩积分，用于有限元质量矩阵组装。
"""

import numpy as np
from typing import Tuple, List, Optional
from utils import arc_cosine_safe, gamma_func


# ======================================================================
# 1. Lloyd 算法 / CVT 节点优化 (源自 676_line_cvt_lloyd)
# ======================================================================

def line_cvt_lloyd_step(
    n: int, a: float, b: float, x: np.ndarray, constrained: bool = True
) -> np.ndarray:
    """
    一维 Lloyd 松弛单步。
    将每个生成点移动到其 Voronoi 单元的质心。
    对于均匀密度，质心等于相邻中点的平均：
        x*_j = (x_{j-1} + 2 x_j + x_{j+1}) / 4
    constrained=True 时固定端点于 [a, b]。
    """
    if n < 2:
        raise ValueError("节点数 n 至少为 2")
    x = np.asarray(x, dtype=float).copy()
    x = np.sort(x)
    if constrained:
        x[0] = a
        x[-1] = b
    x_new = x.copy()
    for j in range(1, n - 1):
        x_new[j] = 0.25 * (x[j - 1] + 2.0 * x[j] + x[j + 1])
    if constrained:
        x_new[0] = a
        x_new[-1] = b
    return x_new


def line_cvt_energy(n: int, a: float, b: float, x: np.ndarray) -> float:
    """
    计算一维 CVT 能量泛函：
        E = Σ_j ∫_{m_{j-1}}^{m_j} (t - x_j)^2 dt
    其中 m_j = (x_j + x_{j+1}) / 2 为 Voronoi 边界。
    解析积分得：E = Σ_j [ (x_j - m_{j-1})^3 + (m_j - x_j)^3 ] / 3
    """
    x = np.asarray(x, dtype=float)
    x = np.sort(x)
    if x[0] < a - 1e-12 or x[-1] > b + 1e-12:
        raise ValueError("节点超出区间 [a, b]")
    energy = 0.0
    for j in range(n):
        if j == 0:
            ml = a
        else:
            ml = 0.5 * (x[j - 1] + x[j])
        if j == n - 1:
            mr = b
        else:
            mr = 0.5 * (x[j] + x[j + 1])
        energy += ((x[j] - ml) ** 3.0 + (mr - x[j]) ** 3.0) / 3.0
    return energy


def generate_cvt_nodes_1d(
    n: int,
    a: float,
    b: float,
    n_iter: int = 200,
    constrained: bool = True,
) -> np.ndarray:
    """
    使用 Lloyd 迭代在 [a, b] 上生成 CVT 最优分布的 n 个节点。
    返回排序后的节点坐标数组。
    """
    if n < 2:
        raise ValueError("节点数 n 至少为 2")
    if a >= b:
        raise ValueError("区间左端点 a 必须小于右端点 b")
    x = np.linspace(a, b, n)
    for _ in range(n_iter):
        x_new = line_cvt_lloyd_step(n, a, b, x, constrained=constrained)
        if np.max(np.abs(x_new - x)) < 1e-12:
            break
        x = x_new
    return x


# ======================================================================
# 2. 多边形计算几何 (源自 882_polygon)
# ======================================================================

def polygon_area_2d(v: np.ndarray) -> float:
    """
    计算二维多边形有向面积（Shoelace 公式）：
        A = 0.5 * Σ_{i=0}^{n-1} (x_i y_{i+1} - x_{i+1} y_i)
    顶点按逆时针顺序给出时面积为正。
    """
    v = np.asarray(v, dtype=float)
    if v.ndim != 2 or v.shape[1] != 2:
        raise ValueError("v 必须是 n×2 的顶点坐标数组")
    n = v.shape[0]
    if n < 3:
        return 0.0
    x = v[:, 0]
    y = v[:, 1]
    area = 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)
    return float(area)


def polygon_centroid_2d(v: np.ndarray) -> np.ndarray:
    """
    计算二维多边形形心（面积加权平均）。
    使用 Green 定理将面积分转化为边界积分：
        Cx = (1/6A) Σ (x_i + x_{i+1})(x_i y_{i+1} - x_{i+1} y_i)
        Cy = (1/6A) Σ (y_i + y_{i+1})(x_i y_{i+1} - x_{i+1} y_i)
    """
    v = np.asarray(v, dtype=float)
    area = polygon_area_2d(v)
    if abs(area) < 1e-15:
        return np.zeros(2)
    n = v.shape[0]
    x = v[:, 0]
    y = v[:, 1]
    cross = x * np.roll(y, -1) - np.roll(x, -1) * y
    cx = np.sum((x + np.roll(x, -1)) * cross) / (6.0 * area)
    cy = np.sum((y + np.roll(y, -1)) * cross) / (6.0 * area)
    return np.array([cx, cy], dtype=float)


def polygon_moment_integral(v: np.ndarray, p: int, q: int) -> float:
    """
    计算多边形上的精确单矩积分：
        I_{pq} = ∫_Ω x^p y^q dA
    使用 Steger 的边界求和公式（源自 882_polygon 的 monomial_integral）。
    """
    v = np.asarray(v, dtype=float)
    n = v.shape[0]
    if n < 3:
        return 0.0
    integral = 0.0
    for i in range(n):
        x1, y1 = v[i]
        x2, y2 = v[(i + 1) % n]
        dx = x2 - x1
        dy = y2 - y1
        # 沿边 (x1,y1)→(x2,y2) 参数化积分
        # ∫_0^1 (x1 + t dx)^p (y1 + t dy)^q (x1 dy - y1 dx) dt
        # 使用二项式展开
        edge_int = 0.0
        for k in range(p + 1):
            for l in range(q + 1):
                coeff = (
                    _binomial(p, k)
                    * _binomial(q, l)
                    * (x1 ** (p - k))
                    * (dx ** k)
                    * (y1 ** (q - l))
                    * (dy ** l)
                    * (x1 * dy - y1 * dx)
                    / (k + l + 1)
                )
                edge_int += coeff
        integral += edge_int
    return abs(integral)


def _binomial(n: int, k: int) -> int:
    """计算二项式系数 C(n,k)。"""
    if k < 0 or k > n:
        return 0
    if k == 0 or k == n:
        return 1
    k = min(k, n - k)
    res = 1
    for i in range(k):
        res = res * (n - i) // (i + 1)
    return res


def polygon_second_moments(v: np.ndarray) -> Tuple[float, float, float]:
    """
    计算水线面的二阶矩（用于水静力学恢复力矩）：
        I_xx = ∫ y^2 dA   (绕 x 轴的惯性矩)
        I_yy = ∫ x^2 dA   (绕 y 轴的惯性矩)
        I_xy = ∫ x y dA   (惯性积)
    返回 (I_xx, I_yy, I_xy)。
    """
    I_xx = polygon_moment_integral(v, 0, 2)
    I_yy = polygon_moment_integral(v, 2, 0)
    I_xy = polygon_moment_integral(v, 1, 1)
    return I_xx, I_yy, I_xy


def polygon_solid_angle_3d(
    poly: np.ndarray, p: np.ndarray
) -> float:
    """
    计算三维多边形在点 p 处所张的带符号立体角（源自 882_polygon）。
    使用 Gauss-Bonnet 定理：
        Ω = Σ_i α_i - (n-2)π
    其中 α_i 为投影到以 p 为中心的单位球面上的多边形的内角。
    用于面板法中的源点影响系数计算。
    """
    poly = np.asarray(poly, dtype=float)
    p = np.asarray(p, dtype=float)
    n = poly.shape[0]
    if n < 3:
        return 0.0
    # 投影到单位球面
    u = poly - p
    norms = np.linalg.norm(u, axis=1)
    if np.any(norms < 1e-15):
        return 0.0
    u = u / norms[:, np.newaxis]
    # 计算球面多边形内角
    angle_sum = 0.0
    for i in range(n):
        v1 = u[(i - 1) % n]
        v2 = u[i]
        v3 = u[(i + 1) % n]
        # 球面角：由大圆弧 v1→v2 和 v2→v3 的夹角
        n1 = np.cross(v2, v1)
        n2 = np.cross(v2, v3)
        n1_norm = np.linalg.norm(n1)
        n2_norm = np.linalg.norm(n2)
        if n1_norm < 1e-15 or n2_norm < 1e-15:
            continue
        cos_alpha = np.dot(n1, n2) / (n1_norm * n2_norm)
        cos_alpha = max(-1.0, min(1.0, cos_alpha))
        angle_sum += np.arccos(cos_alpha)
    solid_angle = angle_sum - (n - 2) * np.pi
    return solid_angle


def polygon_triangulate_earclip(
    x: np.ndarray, y: np.ndarray
) -> List[Tuple[int, int, int]]:
    """
    耳切法（Ear-clipping）三角剖分简单多边形。
    返回三角形索引列表 [(i0,i1,i2), ...]。
    """
    n = len(x)
    if n < 3:
        return []
    indices = list(range(n))
    triangles = []
    while len(indices) > 3:
        ear_found = False
        m = len(indices)
        for i in range(m):
            i_prev = indices[(i - 1) % m]
            i_curr = indices[i]
            i_next = indices[(i + 1) % m]
            if _is_convex(x, y, i_prev, i_curr, i_next, indices):
                if _is_ear(x, y, i_prev, i_curr, i_next, indices):
                    triangles.append((i_prev, i_curr, i_next))
                    indices.pop(i)
                    ear_found = True
                    break
        if not ear_found:
            # 退化情况：强制取第一个凸顶点
            triangles.append((indices[0], indices[1], indices[2]))
            indices.pop(1)
    if len(indices) == 3:
        triangles.append((indices[0], indices[1], indices[2]))
    return triangles


def _is_convex(
    x: np.ndarray, y: np.ndarray, i: int, j: int, k: int, active: List[int]
) -> bool:
    """判断顶点 j 是否为凸顶点（逆时针多边形中 cross > 0）。"""
    cross = (x[j] - x[i]) * (y[k] - y[j]) - (y[j] - y[i]) * (x[k] - x[j])
    return cross > 1e-12


def _is_ear(
    x: np.ndarray, y: np.ndarray, i: int, j: int, k: int, active: List[int]
) -> bool:
    """判断三角形 (i,j,k) 是否为耳（内部不含其他顶点）。"""
    for idx in active:
        if idx in (i, j, k):
            continue
        if _point_in_triangle(x[idx], y[idx], x[i], y[i], x[j], y[j], x[k], y[k]):
            return False
    return True


def _point_in_triangle(
    px: float, py: float,
    ax: float, ay: float,
    bx: float, by: float,
    cx: float, cy: float,
) -> bool:
    """使用重心坐标判断点是否在三角形内。"""
    denom = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
    if abs(denom) < 1e-15:
        return False
    a = ((by - cy) * (px - cx) + (cx - bx) * (py - cy)) / denom
    b = ((cy - ay) * (px - cx) + (ax - cx) * (py - cy)) / denom
    c = 1.0 - a - b
    return (a >= -1e-12) and (b >= -1e-12) and (c >= -1e-12)


# ======================================================================
# 3. Delaunay 质量检查 (源自 1335_triangulation_delaunay_discrepancy)
# ======================================================================

def triangle_angles_2d(
    t: np.ndarray,
) -> np.ndarray:
    """
    计算三角形三个内角（弧度）。
    使用余弦定理：cos(A) = (b² + c² - a²) / (2bc)
    """
    t = np.asarray(t, dtype=float)
    a = np.linalg.norm(t[1] - t[2])
    b = np.linalg.norm(t[0] - t[2])
    c = np.linalg.norm(t[0] - t[1])
    angles = np.zeros(3)
    # 防止数值误差
    eps = 1e-12
    if a < eps or b < eps or c < eps:
        return angles
    cos_a = max(-1.0, min(1.0, (b * b + c * c - a * a) / (2.0 * b * c)))
    cos_b = max(-1.0, min(1.0, (a * a + c * c - b * b) / (2.0 * a * c)))
    angles[0] = arc_cosine_safe(cos_a)
    angles[1] = arc_cosine_safe(cos_b)
    angles[2] = np.pi - angles[0] - angles[1]
    return angles


def triangulation_delaunay_discrepancy(
    nodes: np.ndarray, triangles: List[Tuple[int, int, int]]
) -> float:
    """
    计算三角剖分的 Delaunay 不一致度。
    对每对共享边的相邻三角形，比较当前对角线配置与替代配置的
    最小内角。不一致度 = max(替代最小角 - 当前最小角)。
    若不一致度 ≤ 0，则剖分是 Delaunay 的。
    """
    if len(triangles) < 2:
        return 0.0
    # 构建边到三角形的邻接关系
    edge_to_tri = {}
    for ti, tri in enumerate(triangles):
        edges = [(min(tri[0], tri[1]), max(tri[0], tri[1])),
                 (min(tri[1], tri[2]), max(tri[1], tri[2])),
                 (min(tri[2], tri[0]), max(tri[2], tri[0]))]
        for e in edges:
            edge_to_tri.setdefault(e, []).append(ti)

    max_disc = 0.0
    checked = set()
    for e, tris in edge_to_tri.items():
        if len(tris) != 2:
            continue
        t1_idx, t2_idx = tris[0], tris[1]
        key = (min(t1_idx, t2_idx), max(t1_idx, t2_idx))
        if key in checked:
            continue
        checked.add(key)
        tri1 = triangles[t1_idx]
        tri2 = triangles[t2_idx]
        # 找到共享边的对顶点
        v1 = [v for v in tri1 if v not in e][0]
        v2 = [v for v in tri2 if v not in e][0]
        # 当前配置的四边形
        quad = [nodes[v1], nodes[e[0]], nodes[v2], nodes[e[1]]]
        # 当前对角线 (v1, 共享边端点)
        # 替代对角线 (v1, v2)
        cur_min = min(
            np.min(triangle_angles_2d(np.array([nodes[v1], nodes[e[0]], nodes[e[1]]]))),
            np.min(triangle_angles_2d(np.array([nodes[v2], nodes[e[0]], nodes[e[1]]]))),
        )
        alt_min = min(
            np.min(triangle_angles_2d(np.array([nodes[v1], nodes[v2], nodes[e[0]]]))),
            np.min(triangle_angles_2d(np.array([nodes[v1], nodes[v2], nodes[e[1]]]))),
        )
        max_disc = max(max_disc, alt_min - cur_min)
    return max_disc


# ======================================================================
# 4. 单纯形精确积分 (源自 1080_simplex_integrals)
# ======================================================================

def simplex01_volume(m: int) -> float:
    """
    计算 m 维单位单纯形的体积（测度）：
        V = 1 / m!
    """
    if m < 1:
        return 1.0
    vol = 1.0
    for k in range(1, m + 1):
        vol /= float(k)
    return vol


def simplex01_monomial_integral(m: int, e: np.ndarray) -> float:
    """
    计算单位单纯形上的精确单矩积分：
        I = ∫_{Δ} ∏_{i=1}^m x_i^{e_i} dV
    解析公式：
        I = ∏_{i=1}^m e_i! / ( (m + Σ e_i)! )
    即 I = Γ(e_1+1) ... Γ(e_m+1) / Γ(m + Σ e_i + 1)
    """
    e = np.asarray(e, dtype=int)
    if len(e) != m:
        raise ValueError("指数数组长度必须与维度 m 一致")
    if np.any(e < 0):
        raise ValueError("指数必须非负")
    numerator = 1.0
    for val in e:
        numerator *= gamma_func(float(val + 1))
    den_arg = float(m + np.sum(e) + 1)
    denominator = gamma_func(den_arg)
    return numerator / denominator


def triangle_exact_integral_fem(
    nodes: np.ndarray, exponents: List[Tuple[int, int]]
) -> List[float]:
    """
    对参考三角形上的基函数矩进行精确积分，用于 FEM 质量矩阵。
    参考三角形：顶点 (0,0), (1,0), (0,1)。
    返回每个指数对的积分值列表。
    """
    results = []
    for p, q in exponents:
        val = simplex01_monomial_integral(2, np.array([p, q]))
        results.append(val)
    return results


# ======================================================================
# 5. 海洋平台水线面水静力学计算
# ======================================================================

def compute_waterplane_properties(
    waterline_vertices: np.ndarray,
) -> dict:
    """
    计算海洋平台水线面的水静力学特性。
    返回字典包含：
      - area: 水线面面积 (m²)
      - centroid: 形心坐标 (m, m)
      - I_xx: 绕 x 轴惯性矩 (m⁴)
      - I_yy: 绕 y 轴惯性矩 (m⁴)
      - I_xy: 惯性积 (m⁴)
      - BM_t: 横稳心半径 (m)
      - BM_l: 纵稳心半径 (m)
    """
    v = np.asarray(waterline_vertices, dtype=float)
    area = abs(polygon_area_2d(v))
    centroid = polygon_centroid_2d(v)
    I_xx, I_yy, I_xy = polygon_second_moments(v)
    # 稳心半径 BM = I / V_disp，这里用水线面惯性矩近似
    # 假设排水体积由水线面面积 × 平均吃水估算
    draft = 20.0  # 典型半潜平台吃水
    V_disp = area * draft
    BM_t = I_xx / V_disp if V_disp > 1e-12 else 0.0
    BM_l = I_yy / V_disp if V_disp > 1e-12 else 0.0
    return {
        "area": area,
        "centroid": centroid,
        "I_xx": I_xx,
        "I_yy": I_yy,
        "I_xy": I_xy,
        "BM_t": BM_t,
        "BM_l": BM_l,
    }


def generate_platform_waterline(
    platform_type: str = "semi-submersible",
) -> np.ndarray:
    """
    生成典型海洋平台的水线面多边形顶点。
    semi-submersible：矩形立柱 + 连接撑杆的简化轮廓。
    """
    if platform_type == "semi-submersible":
        # 简化半潜平台：四个立柱，外轮廓近似
        vertices = np.array(
            [
                [-40.0, -30.0],
                [-40.0, 30.0],
                [-15.0, 30.0],
                [-15.0, 10.0],
                [15.0, 10.0],
                [15.0, 30.0],
                [40.0, 30.0],
                [40.0, -30.0],
                [15.0, -30.0],
                [15.0, -10.0],
                [-15.0, -10.0],
                [-15.0, -30.0],
            ],
            dtype=float,
        )
    elif platform_type == "spar":
        # Spar 平台：近似圆形（用多边形逼近）
        theta = np.linspace(0, 2 * np.pi, 33, endpoint=False)
        r = 20.0
        vertices = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
    elif platform_type == "tension-leg":
        # TLP：方形平台
        vertices = np.array(
            [[-35, -35], [35, -35], [35, 35], [-35, 35]], dtype=float
        )
    else:
        raise ValueError(f"未知的平台类型: {platform_type}")
    return vertices
