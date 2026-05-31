"""
retina_geometry.py
视网膜几何离散化与网格质量评估

基于以下种子项目合成：
- 469_geompack: 2D Delaunay三角化、凸包、Voronoi图
- 753_mesh_boundary: 多边形网格边界提取
- 1348_triangulation_quality: 三角网格质量评估
- 527_hexagon_integrals: 单位六边形上的单项式积分

科学背景：
视网膜表面需要被离散化为计算网格以进行后续PDE求解。
视锥细胞在视网膜中央凹（fovea）呈六边形密排阵列，
本模块提供：
1. 六边形感光细胞阵列生成
2. Delaunay三角化构建计算网格
3. 边界提取与网格质量评估
4. 多边形矩计算用于空间积分
"""

import numpy as np
import math
from typing import Tuple, List


# =============================================================================
# 六边形感光细胞阵列生成（基于527_hexagon_integrals）
# =============================================================================

def generate_hexagonal_photoreceptor_array(radius: float, n_rings: int) -> np.ndarray:
    """
    生成六边形密排的光感受器阵列，模拟视网膜中央凹的视锥细胞排列。
    
    六边形晶格的基矢量为：
        a1 = (sqrt(3)/2,  1/2) * d
        a2 = (sqrt(3)/2, -1/2) * d
    其中 d = 2*radius 为相邻细胞中心间距。
    
    对于第n层环上的细胞，其位置由以下公式给出：
        r_{n,k} = n * d * (cos(k*pi/3), sin(k*pi/3)),  k=0,...,5
    加上环上的内部点。
    
    参数:
        radius: 单个感光细胞的有效半径 (微米)
        n_rings: 环数（中心为第0环）
    
    返回:
        points: (N, 2) 的numpy数组，每行为一个感光细胞的中心坐标
    """
    d = 2.0 * radius
    points = []
    
    # 中心点
    points.append([0.0, 0.0])
    
    # 逐层生成环
    for n in range(1, n_rings + 1):
        # 六边形环上的6n个点
        for k in range(6 * n):
            # 每个边上的点数为 n，共6条边
            # 使用角度参数化
            edge = k // n
            pos_in_edge = k % n
            
            # 六边形顶点方向
            theta0 = edge * np.pi / 3.0
            theta1 = ((edge + 1) % 6) * np.pi / 3.0
            
            # 边上插值
            t = pos_in_edge / n if n > 0 else 0.0
            x = n * d * ((1 - t) * np.cos(theta0) + t * np.cos(theta1))
            y = n * d * ((1 - t) * np.sin(theta0) + t * np.sin(theta1))
            points.append([x, y])
    
    return np.array(points, dtype=np.float64)


def hexagon_moment_integral(p: int, q: int, vertices: np.ndarray) -> float:
    """
    计算多边形区域上单项式 x^p * y^q 的定积分。
    
    基于Green公式和Carsten Steger的多边形矩计算方法：
    对于多边形P，其顶点为 (x_i, y_i), i=0,...,n-1（逆时针排列），有：
    
        I(p,q) = ∫∫_P x^p y^q dx dy
                = Σ_{i=0}^{n-1} ∫_{边i} x^p y^{q+1} / (q+1) dx
    
    其中边i从 (x_i, y_i) 到 (x_{i+1}, y_{i+1})。
    
    利用参数化 x = x_i + t*(x_{i+1}-x_i), t∈[0,1]，可展开为：
        I_edge = Σ_{k=0}^p Σ_{m=0}^{q+1} C(p,k) C(q+1,m) 
                 * x_i^{p-k} (Δx)^k * y_i^{q+1-m} (Δy)^m 
                 / [(q+1) * (k+m+1)]
    
    参数:
        p, q: 矩的阶数
        vertices: (n, 2) 多边形顶点坐标，逆时针排列
    
    返回:
        value: 积分值
    """
    n = vertices.shape[0]
    value = 0.0
    
    for i in range(n):
        x0, y0 = vertices[i]
        x1, y1 = vertices[(i + 1) % n]
        dx = x1 - x0
        dy = y1 - y0
        
        # 利用二项式展开计算边上的积分
        for k in range(p + 1):
            for m in range(q + 2):
                binom_pk = math.comb(p, k)
                binom_q1m = math.comb(q + 1, m)
                coeff = binom_pk * binom_q1m
                x_term = (x0 ** (p - k)) * (dx ** k) if p - k >= 0 and k >= 0 else 0.0
                y_term = (y0 ** (q + 1 - m)) * (dy ** m) if q + 1 - m >= 0 and m >= 0 else 0.0
                denom = (q + 1) * (k + m + 1)
                if denom != 0:
                    value += coeff * x_term * y_term / denom
    
    return float(value)


# =============================================================================
# Delaunay三角化（基于469_geompack）
# =============================================================================

def _orient2d(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """
    计算二维定向面积（有符号面积的两倍）：
        orient2d(a,b,c) = (b-a) × (c-a) 
                        = (b_x - a_x)(c_y - a_y) - (b_y - a_y)(c_x - a_x)
    
    正值：a,b,c 逆时针排列
    零值：三点共线
    负值：a,b,c 顺时针排列
    """
    return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))


def _in_circumcircle(a: np.ndarray, b: np.ndarray, c: np.ndarray, p: np.ndarray) -> bool:
    """
    判断点p是否在三角形abc的外接圆内部。
    
    使用行列式判定（基于外接圆幂）：
    
        | a_x  a_y  a_x^2+a_y^2  1 |
        | b_x  b_y  b_x^2+b_y^2  1 |
    D = | c_x  c_y  c_x^2+c_y^2  1 |
        | p_x  p_y  p_x^2+p_y^2  1 |
    
    若D * orient2d(a,b,c) > 0，则p在abc的外接圆内部。
    """
    ax, ay = a[0], a[1]
    bx, by = b[0], b[1]
    cx, cy = c[0], c[1]
    px, py = p[0], p[1]
    
    det = (
        (ax - px) * ((by - py) * (cx**2 + cy**2 - px**2 - py**2) 
                     - (cy - py) * (bx**2 + by**2 - px**2 - py**2))
        - (ay - py) * ((bx - px) * (cx**2 + cy**2 - px**2 - py**2)
                       - (cx - px) * (bx**2 + by**2 - px**2 - py**2))
        + (ax**2 + ay**2 - px**2 - py**2) * ((bx - px) * (cy - py) - (by - py) * (cx - px))
    )
    
    orient = _orient2d(a, b, c)
    return det * orient > 1e-12


def delaunay_triangulation_2d(points: np.ndarray) -> np.ndarray:
    """
    2D点集的Delaunay三角化（朴素O(N^4)算法，适合小规模精细网格）。
    
    Delaunay条件：对于三角化中的任意三角形，其外接圆内部不包含任何其他输入点。
    该条件等价于最大化所有三角形的最小角。
    
    算法步骤：
    1. 遍历所有三点组合 (i,j,k)
    2. 检查定向面积是否为正（确保逆时针）
    3. 检查外接圆是否包含其他点
    4. 若不包含，则该三角形为Delaunay三角形
    
    参数:
        points: (N, 2) 点集坐标
    
    返回:
        triangles: (M, 3) 每个三角形的三个顶点索引
    """
    n = points.shape[0]
    triangles = []
    
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a, b, c = points[i], points[j], points[k]
                
                # 检查定向面积
                orient = _orient2d(a, b, c)
                if abs(orient) < 1e-12:
                    continue  # 三点共线，跳过
                
                # 确保逆时针
                if orient < 0:
                    b, c = c, b
                    j_tmp, k_tmp = k, j
                else:
                    j_tmp, k_tmp = j, k
                
                # 检查Delaunay条件：外接圆不包含其他点
                is_delaunay = True
                for p_idx in range(n):
                    if p_idx in (i, j_tmp, k_tmp):
                        continue
                    if _in_circumcircle(a, c, b, points[p_idx]):
                        is_delaunay = False
                        break
                
                if is_delaunay:
                    triangles.append([i, j_tmp, k_tmp])
    
    return np.array(triangles, dtype=np.int64)


# =============================================================================
# 网格边界提取（基于753_mesh_boundary）
# =============================================================================

def extract_mesh_boundary(triangles: np.ndarray) -> np.ndarray:
    """
    从三角网格中提取边界边序列（基于有向边对消法）。
    
    算法原理：
    1. 对每个三角形的每条边，生成有向边 (a→b)，其中a,b按三角形顶点逆时针顺序
    2. 若有向边 (a→b) 的反向 (b→a) 不存在于其他三角形中，则 (a→b) 为边界边
    3. 将边界边重新排序，形成连续的逆时针闭合环
    
    数学基础：对于一致（conformal）三角网格，内部边恰好被两个相邻三角形以相反方向共享。
    
    参数:
        triangles: (M, 3) 三角形顶点索引，每个三角形顶点按逆时针排列
    
    返回:
        boundary: (B, 2) 边界边列表，每条边为两个顶点索引
    """
    edges = []
    for tri in triangles:
        # 三角形三条边，按逆时针方向
        edges.append((tri[0], tri[1]))
        edges.append((tri[1], tri[2]))
        edges.append((tri[2], tri[0]))
    
    # 统计每条有向边的出现次数及其反向边是否存在
    edge_set = set(edges)
    boundary_edges = []
    
    for e in edges:
        reverse = (e[1], e[0])
        if reverse not in edge_set:
            boundary_edges.append(e)
    
    if not boundary_edges:
        return np.array([]).reshape(0, 2)
    
    # 排序形成连续环
    boundary_edges = list(dict.fromkeys(boundary_edges))  # 去重保序
    
    # 构建邻接表排序
    ordered = [boundary_edges[0]]
    remaining = boundary_edges[1:]
    
    while remaining:
        current_end = ordered[-1][1]
        found = False
        for idx, e in enumerate(remaining):
            if e[0] == current_end:
                ordered.append(e)
                remaining.pop(idx)
                found = True
                break
        if not found:
            break
    
    return np.array(ordered, dtype=np.int64)


# =============================================================================
# 网格质量评估（基于1348_triangulation_quality）
# =============================================================================

def triangle_quality_alpha(p: np.ndarray, q: np.ndarray, r: np.ndarray) -> float:
    """
    计算三角形的ALPHA质量度量。
    
    ALPHA = min(θ_p, θ_q, θ_r) / 60°
    
    其中θ_p, θ_q, θ_r为三角形的三个内角。归一化到[0,1]，越接近1质量越好。
    
    内角计算使用余弦定理：
        cos(θ_p) = (|pq|² + |pr|² - |qr|²) / (2 * |pq| * |pr|)
    
    参数:
        p, q, r: 三角形三个顶点坐标，每个为(2,)数组
    
    返回:
        alpha: ALPHA质量度量
    """
    # 边长平方
    a2 = np.sum((q - r) ** 2)
    b2 = np.sum((p - r) ** 2)
    c2 = np.sum((p - q) ** 2)
    
    # 使用余弦定理计算角度（弧度）
    eps = 1e-14
    a = np.sqrt(a2)
    b = np.sqrt(b2)
    c = np.sqrt(c2)
    
    # 避免数值误差导致arccos域外
    cos_p = np.clip((b2 + c2 - a2) / (2.0 * b * c + eps), -1.0, 1.0)
    cos_q = np.clip((a2 + c2 - b2) / (2.0 * a * c + eps), -1.0, 1.0)
    cos_r = np.clip((a2 + b2 - c2) / (2.0 * a * b + eps), -1.0, 1.0)
    
    angle_p = np.arccos(cos_p)
    angle_q = np.arccos(cos_q)
    angle_r = np.arccos(cos_r)
    
    min_angle = min(angle_p, angle_q, angle_r)
    alpha = min_angle / (np.pi / 3.0)  # 60度 = pi/3
    
    return float(alpha)


def triangle_quality_q(p: np.ndarray, q: np.ndarray, r: np.ndarray) -> float:
    """
    计算三角形的Q质量度量。
    
    Q = 2 * r_in / R_out
    
    其中 r_in 为内切圆半径，R_out 为外接圆半径。
    Q = 1 表示正三角形，Q → 0 表示退化三角形。
    
    公式：
        面积 A = 0.5 * | (q-p) × (r-p) |
        r_in = A / s,  s = (a+b+c)/2 为半周长
        R_out = abc / (4A)
        Q = 2 * r_in / R_out = 4A² / (abc * s)
    
    参数:
        p, q, r: 三角形三个顶点坐标
    
    返回:
        q_measure: Q质量度量
    """
    a = np.linalg.norm(q - r)
    b = np.linalg.norm(p - r)
    c = np.linalg.norm(p - q)
    
    # 面积（有符号面积的两倍除以2）
    area = 0.5 * abs(_orient2d(p, q, r))
    
    s = 0.5 * (a + b + c)
    eps = 1e-14
    
    if s < eps or a < eps or b < eps or c < eps or area < eps:
        return 0.0
    
    r_in = area / s
    R_out = a * b * c / (4.0 * area + eps)
    q_measure = 2.0 * r_in / (R_out + eps)
    
    return float(q_measure)


def evaluate_mesh_quality(points: np.ndarray, triangles: np.ndarray) -> dict:
    """
    综合评估三角网格质量。
    
    返回字典包含：
    - alpha_min, alpha_ave: ALPHA度量的最小值和平均值
    - q_min, q_ave: Q度量的最小值和平均值
    - area_min, area_max, area_ave, area_std: 面积统计
    - boundary_segments: 边界边列表
    """
    n_tri = triangles.shape[0]
    alphas = []
    qs = []
    areas = []
    
    for tri in triangles:
        p, q, r = points[tri[0]], points[tri[1]], points[tri[2]]
        alphas.append(triangle_quality_alpha(p, q, r))
        qs.append(triangle_quality_q(p, q, r))
        areas.append(0.5 * abs(_orient2d(p, q, r)))
    
    alphas = np.array(alphas)
    qs = np.array(qs)
    areas = np.array(areas)
    
    boundary = extract_mesh_boundary(triangles)
    
    quality = {
        'alpha_min': float(np.min(alphas)),
        'alpha_ave': float(np.mean(alphas)),
        'q_min': float(np.min(qs)),
        'q_ave': float(np.mean(qs)),
        'area_min': float(np.min(areas)),
        'area_max': float(np.max(areas)),
        'area_ave': float(np.mean(areas)),
        'area_std': float(np.std(areas)),
        'boundary_segments': boundary,
        'num_triangles': n_tri,
        'num_boundary_edges': boundary.shape[0] if boundary.size > 0 else 0,
    }
    
    return quality


# =============================================================================
# 凸包计算（基于469_geompack的Jarvis march）
# =============================================================================

def convex_hull_2d(points: np.ndarray) -> np.ndarray:
    """
    2D点集的凸包计算（Gift Wrapping / Jarvis March算法）。
    
    算法复杂度：O(n*h)，其中h为凸包上的点数。
    
    步骤：
    1. 找到x坐标最小的点作为起始点
    2. 从当前点出发，寻找使转向角最大的下一个点
    3. 重复直到回到起始点
    
    转向角判定使用叉积：
        cross(p_current, p_candidate, p_next) > 0 表示左转
    
    参数:
        points: (N, 2) 点集
    
    返回:
        hull_indices: 凸包上的点索引列表（逆时针）
    """
    n = points.shape[0]
    if n <= 3:
        return np.arange(n)
    
    # 找到最左下点
    start = 0
    for i in range(1, n):
        if points[i, 0] < points[start, 0] or \
           (abs(points[i, 0] - points[start, 0]) < 1e-12 and points[i, 1] < points[start, 1]):
            start = i
    
    hull = []
    current = start
    
    while True:
        hull.append(current)
        next_point = (current + 1) % n
        
        for i in range(n):
            if i == current:
                continue
            # 如果i在current->next_point的左侧，则更新next_point
            cross = _orient2d(points[current], points[next_point], points[i])
            if cross > 1e-12:
                next_point = i
            elif abs(cross) < 1e-12:
                # 共线时取更远的点
                d_next = np.sum((points[next_point] - points[current]) ** 2)
                d_i = np.sum((points[i] - points[current]) ** 2)
                if d_i > d_next:
                    next_point = i
        
        current = next_point
        if current == start:
            break
    
    return np.array(hull, dtype=np.int64)
