"""
基于 Centroidal Voronoi Tessellation (CVT) 的自适应网格生成与优化模块

融合自:
- 238_cvt: CVT 迭代算法、能量最小化
- 1398_voronoi_plot: Voronoi 图生成与最近邻搜索

CVT 能量泛函:
    F({z_i}_{i=1}^n) = Σ_i ∫_{V_i} ρ(x) ||x - z_i||^2 dx
其中 V_i 是生成元 z_i 对应的 Voronoi 单元，ρ(x) 是密度函数。

Lloyd 迭代:
    z_i^{(k+1)} = centroid(V_i^{(k)})
    = ∫_{V_i} ρ(x) x dx / ∫_{V_i} ρ(x) dx

收敛性分析:
    F^{(k+1)} <= F^{(k)}，且等号成立当且仅当 z_i = centroid(V_i)。
"""

import numpy as np


def compute_voronoi_cells(sample_points, generators):
    """
    计算每个采样点最近的发电机（生成元）索引。
    
    对应原 find_closest.m 的核心功能。
    
    Parameters
    ----------
    sample_points : ndarray, shape (n_samples, dim)
        采样点
    generators : ndarray, shape (n_gen, dim)
        生成元坐标
    
    Returns
    -------
    nearest : ndarray, shape (n_samples,)
        每个采样点最近的生成元索引
    """
    if sample_points.ndim == 1:
        sample_points = sample_points.reshape(-1, 1)
    if generators.ndim == 1:
        generators = generators.reshape(-1, 1)

    n_samples = sample_points.shape[0]
    n_gen = generators.shape[0]

    # 使用广播计算所有点对之间的距离平方
    diff = sample_points[:, np.newaxis, :] - generators[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    nearest = np.argmin(dist_sq, axis=1)
    return nearest


def cvt_iterate(generators, sample_points, density_values=None):
    """
    执行一次 Lloyd CVT 迭代。
    
    对应原 cvt_iterate.m 的核心功能。
    
    Parameters
    ----------
    generators : ndarray, shape (n_gen, dim)
        当前生成元坐标
    sample_points : ndarray, shape (n_samples, dim)
        采样点
    density_values : ndarray, shape (n_samples,), optional
        采样点处的密度函数值 ρ(x)。若 None 则假设均匀密度。
    
    Returns
    -------
    new_generators : ndarray, shape (n_gen, dim)
        更新后的生成元坐标
    it_diff : float
        生成元位移的 L2 范数之和
    energy : float
        离散 CVT 能量
    """
    n_gen = generators.shape[0]
    dim = generators.shape[1]

    if density_values is None:
        density_values = np.ones(sample_points.shape[0])

    nearest = compute_voronoi_cells(sample_points, generators)

    new_generators = np.zeros_like(generators)
    counts = np.zeros(n_gen)
    energy = 0.0

    for j in range(n_gen):
        mask = (nearest == j)
        count = np.sum(mask)
        if count > 0:
            weights = density_values[mask]
            weighted_sum = np.sum(
                sample_points[mask] * weights[:, np.newaxis], axis=0
            )
            total_weight = np.sum(weights)
            new_generators[j] = weighted_sum / total_weight
            counts[j] = total_weight

            # 计算能量贡献
            diff = sample_points[mask] - generators[j]
            energy += np.sum(weights[:, np.newaxis] * (diff ** 2))
        else:
            # 空单元：保持原位置或随机扰动
            new_generators[j] = generators[j]

    energy = energy / sample_points.shape[0]

    it_diff = np.sum(np.sqrt(np.sum((new_generators - generators) ** 2, axis=1)))

    return new_generators, it_diff, energy


def cvt_energy(generators, sample_points, density_values=None):
    """
    计算离散 CVT 能量。
    
    对应原 cvt_energy.m 的核心功能。
    
    Parameters
    ----------
    generators : ndarray, shape (n_gen, dim)
        生成元坐标
    sample_points : ndarray, shape (n_samples, dim)
        采样点
    density_values : ndarray, shape (n_samples,), optional
        密度函数值
    
    Returns
    -------
    energy : float
        离散 CVT 能量
    """
    if density_values is None:
        density_values = np.ones(sample_points.shape[0])

    nearest = compute_voronoi_cells(sample_points, generators)
    energy = 0.0
    for j in range(generators.shape[0]):
        mask = (nearest == j)
        if np.sum(mask) > 0:
            diff = sample_points[mask] - generators[j]
            energy += np.sum(
                density_values[mask][:, np.newaxis] * (diff ** 2)
            )

    return energy / sample_points.shape[0]


def generate_cvt_mesh(
    n_cells,
    domain_bounds,
    density_func=None,
    it_max=50,
    sample_multiplier=100,
    tol=1e-5
):
    """
    生成 CVT 优化的二维节点分布。
    
    Parameters
    ----------
    n_cells : int
        目标 Voronoi 单元数（即节点数）
    domain_bounds : tuple
        ((xmin, xmax), (ymin, ymax))
    density_func : callable, optional
        density_func(x, y) -> float，返回点 (x,y) 处的密度值
    it_max : int
        最大迭代次数
    sample_multiplier : int
        每个单元的采样点数倍数
    tol : float
        收敛容差
    
    Returns
    -------
    generators : ndarray, shape (n_cells, 2)
        CVT 优化后的节点坐标
    energy_history : list
        能量迭代历史
    """
    dim = 2
    sample_num = n_cells * sample_multiplier

    # 均匀随机初始化
    rng = np.random.default_rng(seed=42)
    generators = rng.random((n_cells, dim))
    generators[:, 0] = generators[:, 0] * (domain_bounds[0][1] - domain_bounds[0][0]) + domain_bounds[0][0]
    generators[:, 1] = generators[:, 1] * (domain_bounds[1][1] - domain_bounds[1][0]) + domain_bounds[1][0]

    energy_history = []

    for it in range(it_max):
        sample_points = rng.random((sample_num, dim))
        sample_points[:, 0] = sample_points[:, 0] * (domain_bounds[0][1] - domain_bounds[0][0]) + domain_bounds[0][0]
        sample_points[:, 1] = sample_points[:, 1] * (domain_bounds[1][1] - domain_bounds[1][0]) + domain_bounds[1][0]

        if density_func is not None:
            density_values = np.array([
                density_func(p[0], p[1]) for p in sample_points
            ])
            # 确保密度为正
            density_values = np.maximum(density_values, 1e-10)
        else:
            density_values = None

        generators, it_diff, energy = cvt_iterate(
            generators, sample_points, density_values
        )
        energy_history.append(energy)

        # 边界投影
        generators[:, 0] = np.clip(generators[:, 0], domain_bounds[0][0], domain_bounds[0][1])
        generators[:, 1] = np.clip(generators[:, 1], domain_bounds[1][0], domain_bounds[1][1])

        if it_diff < tol:
            break

    return generators, energy_history


def compute_delaunay_triangulation(points):
    """
    基于 Bowyer-Watson 算法的简化 Delaunay 三角剖分。
    将 CVT 节点剖分为三角形网格。
    
    对于计算数学 AMR 应用，我们采用基于超平面的简单剖分策略：
    先对点进行排序，然后生成三角形。
    
    Parameters
    ----------
    points : ndarray, shape (n, 2)
        二维点集
    
    Returns
    -------
    triangles : ndarray, shape (m, 3)
        每个三角形由三个顶点索引组成
    nodes : ndarray, shape (n, 2)
        去重后的节点坐标
    """
    # 去除重复点
    tol = 1e-10
    unique_points = []
    for p in points:
        is_duplicate = False
        for up in unique_points:
            if np.linalg.norm(p - up) < tol:
                is_duplicate = True
                break
        if not is_duplicate:
            unique_points.append(p)
    nodes = np.array(unique_points)
    n = len(nodes)

    if n < 3:
        raise ValueError("compute_delaunay_triangulation: 至少需要3个节点")

    # 使用 scipy 的 Delaunay 如果可用，否则用简单方法
    try:
        from scipy.spatial import Delaunay
        tri = Delaunay(nodes)
        triangles = tri.simplices.astype(int)
        return triangles, nodes
    except ImportError:
        pass

    # 简单网格化：对矩形域进行三角剖分
    # 假设点在矩形域内，先排序然后生成三角形
    xmin, ymin = nodes.min(axis=0)
    xmax, ymax = nodes.max(axis=0)

    # 使用最近邻方法生成三角形
    # 对每个点，找最近的两个邻居构成三角形
    triangles = []
    used = set()

    for i in range(n):
        dists = np.linalg.norm(nodes - nodes[i], axis=1)
        dists[i] = np.inf
        neighbors = np.argsort(dists)[:min(6, n - 1)]
        for j in range(len(neighbors)):
            for k in range(j + 1, len(neighbors)):
                a, b, c = i, neighbors[j], neighbors[k]
                key = tuple(sorted([a, b, c]))
                if key not in used:
                    # 检查三角形方向（逆时针）
                    x1, y1 = nodes[a]
                    x2, y2 = nodes[b]
                    x3, y3 = nodes[c]
                    area = 0.5 * ((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
                    if area > 1e-12:
                        triangles.append([a, b, c])
                        used.add(key)
                    elif area < -1e-12:
                        triangles.append([a, c, b])
                        used.add(key)

    if len(triangles) == 0:
        # 兜底：生成一个简单的三角形覆盖
        triangles = [[0, 1, 2]] if n >= 3 else []

    return np.array(triangles, dtype=int), nodes


def triangle_area(nodes, triangle):
    """
    计算三角形的有向面积。
    
    Parameters
    ----------
    nodes : ndarray, shape (n, 2)
        节点坐标
    triangle : array-like, shape (3,)
        三个顶点索引
    
    Returns
    -------
    area : float
        面积（正值）
    """
    p1 = nodes[triangle[0]]
    p2 = nodes[triangle[1]]
    p3 = nodes[triangle[2]]
    area = 0.5 * abs(
        (p2[0] - p1[0]) * (p3[1] - p1[1]) -
        (p3[0] - p1[0]) * (p2[1] - p1[1])
    )
    return area
