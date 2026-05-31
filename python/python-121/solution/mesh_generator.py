"""
mesh_generator.py
心脏组织网格生成与边界处理模块

融入原项目:
- 239_cvt_1_movie: Centroidal Voronoi Tessellation迭代
- 106_boundary_word_drafter: 多边形边界处理
- 928_pwl_interp_2d_scattered: 散乱数据三角剖分与插值

功能:
1. 基于CVT的心脏组织非结构化网格生成
2. 多边形边界定义与点在多边形内测试
3. 散乱电势数据插值
"""

import numpy as np
from math import sqrt


# ============================================================================
# 点在多边形内测试（源自 106_boundary_word_drafter/polygon_contains_point）
# ============================================================================

def polygon_contains_point(polygon, q):
    """
    判断点 q 是否在简单多边形 polygon 内部
    
    算法: 射线投射法（Ray Casting）
    从点 q 向右发射水平射线，计算与多边形边界的交点数
    - 奇数个交点: 点在内部
    - 偶数个交点: 点在外部
    
    参数:
        polygon: (N,2) 多边形顶点数组
        q: (2,) 测试点坐标
    返回:
        inside: 是否在内部
    """
    polygon = np.asarray(polygon)
    q = np.asarray(q)
    
    n = len(polygon)
    if n < 3:
        return False
    
    inside = False
    x1, y1 = polygon[n - 1]
    
    for i in range(n):
        x2, y2 = polygon[i]
        
        # 检查射线是否与边相交
        if ((y1 < q[1] <= y2) or (q[1] <= y1 and y2 < q[1])):
            # 计算交点x坐标
            if y2 != y1:
                x_intersect = x1 + (q[1] - y1) * (x2 - x1) / (y2 - y1)
                if q[0] <= x_intersect:
                    inside = not inside
        
        x1, y1 = x2, y2
    
    # 边界点检测
    x1, y1 = polygon[n - 1]
    for i in range(n):
        x2, y2 = polygon[i]
        
        # 点是否在边线段上
        cross = (q[1] - y1) * (x2 - x1) - (y2 - y1) * (q[0] - x1)
        if abs(cross) < 1e-10:
            # 检查是否在端点之间
            dot = (q[0] - x1) * (x2 - x1) + (q[1] - y1) * (y2 - y1)
            sq_len = (x2 - x1) ** 2 + (y2 - y1) ** 2
            if 0 <= dot <= sq_len:
                return True
        
        x1, y1 = x2, y2
    
    return inside


def compute_polygon_bbox(polygon):
    """计算多边形包围盒"""
    polygon = np.asarray(polygon)
    if len(polygon) == 0:
        return (0.0, 0.0, 0.0, 0.0)
    xmin = np.min(polygon[:, 0])
    xmax = np.max(polygon[:, 0])
    ymin = np.min(polygon[:, 1])
    ymax = np.max(polygon[:, 1])
    return (xmin, xmax, ymin, ymax)


# ============================================================================
# Centroidal Voronoi Tessellation（源自 239_cvt_1_movie）
# ============================================================================

def find_closest(ndim, n_generators, n_samples, samples, generators):
    """
    对每个样本点找到最近的生成器索引
    
    参数:
        ndim: 空间维度
        n_generators: 生成器数量
        n_samples: 样本点数量
        samples: (ndim, n_samples) 样本点
        generators: (ndim, n_generators) 生成器
    返回:
        nearest: (n_samples,) 最近生成器索引
    """
    nearest = np.zeros(n_samples, dtype=int)
    for j in range(n_samples):
        min_dist = float('inf')
        min_idx = 0
        for i in range(n_generators):
            d = 0.0
            for k in range(ndim):
                diff = generators[k, i] - samples[k, j]
                d += diff * diff
            if d < min_dist:
                min_dist = d
                min_idx = i
        nearest[j] = min_idx
    return nearest


def cvt_iterate(n, r, ratio):
    """
    CVT迭代: 将生成器移动到Voronoi细胞的质心
    
    数学原理:
    对于区域 Ω 和生成器 {z_i}_{i=1}^n，CVT最小化能量泛函:
    
    E(z_1, ..., z_n) = sum_{i=1}^n ∫_{V_i} ρ(x) ||x - z_i||² dx
    
    其中 V_i = {x ∈ Ω : ||x - z_i|| ≤ ||x - z_j||, ∀j ≠ i} 是Voronoi细胞
    
    最优条件: z_i = (∫_{V_i} ρ(x) x dx) / (∫_{V_i} ρ(x) dx)
    
    参数:
        n: 生成器/Voronoi细胞数量
        r: (2, n) 当前生成器位置
        ratio: 每个生成器的采样点数
    返回:
        r: 更新后的生成器位置
        diff: 生成器移动的总距离
        energy: 离散能量估计
    """
    ndim = 2
    sample_num = ratio * n
    
    # 在 [0,1]^2 内均匀采样
    s = np.random.rand(ndim, sample_num)
    
    # 找到每个样本点最近的生成器
    nearest = find_closest(ndim, n, sample_num, s, r)
    
    # 累积每个Voronoi细胞的质心
    r2 = np.zeros((ndim, n))
    energy = 0.0
    count = np.zeros(n)
    
    for j in range(sample_num):
        idx = nearest[j]
        r2[:, idx] += s[:, j]
        dx = r[0, idx] - s[0, j]
        dy = r[1, idx] - s[1, j]
        energy += dx * dx + dy * dy
        count[idx] += 1
    
    energy = energy / sample_num
    
    # 计算质心
    for j in range(n):
        if count[j] > 0:
            r2[:, j] /= count[j]
    
    # 计算移动距离
    diff = 0.0
    for j in range(n):
        dx = r2[0, j] - r[0, j]
        dy = r2[1, j] - r[1, j]
        diff += sqrt(dx * dx + dy * dy)
    
    # 更新生成器
    r[:, :] = r2[:, :]
    
    return r, diff, energy


def generate_cvt_mesh(n_generators, n_iterations=50, ratio=1000, domain=None):
    """
    生成CVT网格
    
    参数:
        n_generators: 生成器数量
        n_iterations: CVT迭代次数
        ratio: 采样比例
        domain: ((xmin,xmax),(ymin,ymax)) 定义域
    返回:
        generators: (2, n_generators) 生成器位置
        diff_history: 每次迭代的移动距离
        energy_history: 每次迭代的能量
    """
    if domain is None:
        domain = ((0.0, 1.0), (0.0, 1.0))
    
    (xmin, xmax), (ymin, ymax) = domain
    
    # 随机初始化生成器
    r = np.zeros((2, n_generators))
    r[0, :] = np.random.uniform(xmin, xmax, n_generators)
    r[1, :] = np.random.uniform(ymin, ymax, n_generators)
    
    diff_history = []
    energy_history = []
    
    for _ in range(n_iterations):
        r, diff, energy = cvt_iterate(n_generators, r, ratio)
        diff_history.append(diff)
        energy_history.append(energy)
    
    return r, diff_history, energy_history


# ============================================================================
# 心脏几何定义
# ============================================================================

def define_cardiac_boundary(model='ventricle'):
    """
    定义心脏组织边界
    
    返回多边形顶点，用于定义模拟区域
    """
    if model == 'ventricle':
        # 简化的左心室截面轮廓
        t = np.linspace(0, 2 * np.pi, 100)
        # 使用椭圆近似心室形状
        a, b = 1.0, 1.4
        x = a * np.cos(t)
        y = b * np.sin(t) * (1.0 + 0.1 * np.cos(3 * t))
        # 添加心尖变形
        y = y - 0.3 * np.sin(t) ** 2
        polygon = np.column_stack((x, y))
        return polygon
    elif model == 'atrium':
        # 心房近似形状
        t = np.linspace(0, 2 * np.pi, 80)
        x = 1.2 * np.cos(t) + 0.2 * np.cos(3 * t)
        y = 0.8 * np.sin(t) + 0.1 * np.sin(4 * t)
        polygon = np.column_stack((x, y))
        return polygon
    else:
        # 默认正方形
        return np.array([[0, 0], [1, 0], [1, 1], [0, 1]])


def filter_points_in_polygon(points, polygon):
    """
    过滤出位于多边形内部的点
    
    参数:
        points: (N, 2) 点集
        polygon: (M, 2) 多边形顶点
    返回:
        inside_points: 内部点
        inside_mask: 布尔掩码
    """
    points = np.asarray(points)
    n = len(points)
    mask = np.zeros(n, dtype=bool)
    
    for i in range(n):
        mask[i] = polygon_contains_point(polygon, points[i])
    
    return points[mask], mask


def generate_cardiac_mesh(n_points, model='ventricle', n_cvt_iter=30):
    """
    生成心脏组织网格
    
    结合CVT和边界裁剪，在心脏几何内部生成均匀分布的节点
    
    参数:
        n_points: 目标节点数
        model: 心脏模型类型
        n_cvt_iter: CVT迭代次数
    返回:
        nodes: (N, 2) 网格节点坐标
        polygon: 边界多边形
    """
    polygon = define_cardiac_boundary(model)
    bbox = compute_polygon_bbox(polygon)
    
    # 估算需要的生成器数量（考虑边界裁剪后大约保留60%）
    n_generators = int(n_points / 0.6)
    
    # 生成CVT
    domain = ((bbox[0], bbox[1]), (bbox[2], bbox[3]))
    generators, _, _ = generate_cvt_mesh(n_generators, n_cvt_iter, ratio=500,
                                          domain=domain)
    
    # 裁剪到多边形内部
    nodes, mask = filter_points_in_polygon(generators.T, polygon)
    
    # 如果节点数不足，补充随机点
    while len(nodes) < n_points:
        extra = np.zeros((n_points - len(nodes), 2))
        extra[:, 0] = np.random.uniform(bbox[0], bbox[1], len(extra))
        extra[:, 1] = np.random.uniform(bbox[2], bbox[3], len(extra))
        extra_inside, _ = filter_points_in_polygon(extra, polygon)
        if len(extra_inside) > 0:
            nodes = np.vstack([nodes, extra_inside])
        else:
            break
    
    # 截取目标数量
    if len(nodes) > n_points:
        nodes = nodes[:n_points]
    
    return nodes, polygon


# ============================================================================
# 散乱数据插值（源自 928_pwl_interp_2d_scattered）
# ============================================================================

def scattered_interpolation_2d(data_points, data_values, query_points):
    """
    二维散乱数据的分段线性插值（Shepard方法简化版）
    
    给定散乱数据点 {x_i, y_i} 和值 {z_i}，在查询点 (xq, yq) 处插值
    
    Shepard插值:
    z(x) = sum_i w_i(x) * z_i / sum_i w_i(x)
    w_i(x) = 1 / ||x - x_i||^p
    
    参数:
        data_points: (N, 2) 数据点坐标
        data_values: (N,) 数据值
        query_points: (M, 2) 查询点坐标
    返回:
        interpolated: (M,) 插值结果
    """
    data_points = np.asarray(data_points)
    data_values = np.asarray(data_values)
    query_points = np.asarray(query_points)
    
    n_query = len(query_points)
    n_data = len(data_points)
    interpolated = np.zeros(n_query)
    
    p = 2.0  # Shepard指数
    
    for j in range(n_query):
        qx, qy = query_points[j]
        weights = np.zeros(n_data)
        
        for i in range(n_data):
            dx = qx - data_points[i, 0]
            dy = qy - data_points[i, 1]
            dist_sq = dx * dx + dy * dy
            if dist_sq < 1e-20:
                interpolated[j] = data_values[i]
                weights = None
                break
            weights[i] = 1.0 / (dist_sq ** (p / 2.0))
        
        if weights is not None:
            w_sum = np.sum(weights)
            if w_sum > 0:
                interpolated[j] = np.sum(weights * data_values) / w_sum
            else:
                interpolated[j] = np.mean(data_values)
    
    return interpolated
