"""
mesh_generator.py
自适应网格生成与空间统计模块

融合以下种子项目的核心算法：
  - 1396_voronoi_mountains：Voronoi最近邻距离计算
  - 258_cvt_metric：变度量空间中的重心Voronoi镶嵌
  - 1306_triangle_histogram：三角形区域点的统计直方图

物理背景：
  重力异常数据在空间分布上往往不均匀，需要在数据密集区加密网格、
  在稀疏区粗化网格以提高计算效率。本模块提供：
  
  1. Voronoi图驱动的空间分解（用于重力测站数据的区域划分）
  2. 变度量CVT（根据重力梯度自适应调整网格密度）
  3. 球面三角形直方图（评估全球重力数据的空间覆盖均匀性）
"""

import numpy as np


def voronoi_nearest_neighbor(points, query_points):
    """
    计算查询点到给定点集的Voronoi最近邻。
    
    融合 1396_voronoi_mountains 的核心算法。
    
    对于每个查询点 q，找到生成点集 P 中最近的点：
        k = argmin_{p in P} ||q - p||_2
    
    返回距离和最近邻索引。
    
    参数：
        points: (N, d) 生成点集
        query_points: (M, d) 查询点
    返回：
        distances: (M,) 最近距离
        indices: (M,) 最近邻索引
    """
    points = np.asarray(points, dtype=float)
    query_points = np.asarray(query_points, dtype=float)
    
    if points.ndim == 1:
        points = points.reshape(-1, 1)
    if query_points.ndim == 1:
        query_points = query_points.reshape(-1, 1)
    
    N, d = points.shape
    M = query_points.shape[0]
    
    distances = np.zeros(M)
    indices = np.zeros(M, dtype=int)
    
    for m in range(M):
        q = query_points[m]
        diff = points - q
        dists = np.sum(diff**2, axis=1)
        idx = np.argmin(dists)
        distances[m] = np.sqrt(dists[idx])
        indices[m] = idx
    
    return distances, indices


def voronoi_region_area_2d(points, n_samples=10000):
    """
    使用蒙特卡洛采样估算二维Voronoi区域的面积。
    
    对每个生成点 p_i，在包围盒内随机采样 n_samples 个点，
    统计最近邻为 p_i 的点的比例，乘以总面积即得区域面积估计。
    """
    points = np.asarray(points, dtype=float)
    if points.ndim == 1:
        points = points.reshape(-1, 1)
    
    N = points.shape[0]
    
    # 包围盒
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    ranges = maxs - mins
    # 扩展10%
    mins -= 0.1 * ranges
    maxs += 0.1 * ranges
    ranges = maxs - mins
    
    d = points.shape[1]
    total_area = np.prod(ranges)
    
    # 随机采样
    samples = np.random.rand(n_samples, d) * ranges + mins
    _, indices = voronoi_nearest_neighbor(points, samples)
    
    areas = np.zeros(N)
    for i in range(N):
        count = np.sum(indices == i)
        areas[i] = count / n_samples * total_area
    
    return areas


def cvt_lloyd_iteration(points, metric_func, n_samples=5000, n_iter=20):
    """
    变度量空间中的Lloyd迭代（重心Voronoi镶嵌）。
    
    融合 258_cvt_metric 的核心算法。
    
    标准欧氏距离：d(x,y)^2 = ||x-y||^2
    变度量距离：d(x,y)^2 = (x-y)^T * A((x+y)/2) * (x-y)
    
    其中 A(p) 是点 p 处的对称正定度量矩阵，反映重力异常梯度信息。
    
    算法：
      1. 在区域内采样大量点
      2. 对每个采样点，找到变度量距离下最近的生成点
      3. 将生成点更新为其Voronoi区域的质心
      4. 重复直到收敛
    
    参数：
        points: (N, d) 初始生成点
        metric_func: callable(p) -> (d, d) 对称正定矩阵
        n_samples: 每步采样数
        n_iter: Lloyd迭代次数
    返回：
        points: 优化后的生成点
        energies: 每步的CVT能量
    """
    points = np.asarray(points, dtype=float)
    if points.ndim == 1:
        points = points.reshape(-1, 1)
    
    N, d = points.shape
    energies = []
    
    # 包围盒
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    ranges = maxs - mins
    mins -= 0.2 * ranges
    maxs += 0.2 * ranges
    ranges = maxs - mins
    
    for it in range(n_iter):
        # 采样
        samples = np.random.rand(n_samples, d) * ranges + mins
        
        # 为每个采样点找到最近生成点（变度量距离）
        nearest = np.zeros(n_samples, dtype=int)
        min_dists = np.full(n_samples, np.inf)
        
        for g in range(N):
            pg = points[g]
            A = metric_func(pg)
            # 确保正定
            A = _ensure_spd(A)
            
            diff = samples - pg
            # d^2 = diff^T A diff，对多个点批量计算
            dists = np.sum(diff @ A * diff, axis=1)
            mask = dists < min_dists
            min_dists[mask] = dists[mask]
            nearest[mask] = g
        
        # 计算CVT能量
        energy = np.mean(min_dists)
        energies.append(energy)
        
        # 更新生成点为区域质心
        new_points = np.zeros_like(points)
        counts = np.zeros(N)
        
        for g in range(N):
            mask = nearest == g
            count = np.sum(mask)
            counts[g] = count
            if count > 0:
                new_points[g] = np.mean(samples[mask], axis=0)
            else:
                new_points[g] = points[g]
        
        points = new_points.copy()
    
    return points, energies


def _ensure_spd(A):
    """确保矩阵对称正定。"""
    A = np.asarray(A, dtype=float)
    # 对称化
    A = 0.5 * (A + A.T)
    # 特征值截断
    eigvals, eigvecs = np.linalg.eigh(A)
    eigvals = np.maximum(eigvals, 1e-8)
    A = eigvecs @ np.diag(eigvals) @ eigvecs.T
    return A


def adaptive_metric_from_gravity(gravity_grad, base_scale=1.0):
    """
    由重力梯度构造变度量矩阵。
    
    在重力梯度大的区域（异常变化剧烈），缩小度量尺度以加密网格：
        A(p) = I * (1 + ||grad(g)|| / g_ref) / scale_factor
    
    参数：
        gravity_grad: (d,) 重力梯度向量 [mGal/m]
        base_scale: 基础尺度
    返回：
        A: (d, d) 度量矩阵
    """
    grad = np.asarray(gravity_grad, dtype=float)
    g_norm = np.linalg.norm(grad)
    factor = (1.0 + g_norm / 1e-3) * base_scale  # 1e-3 mGal/m 为参考梯度
    d = len(grad)
    A = np.eye(d) * factor
    return A


def spherical_triangle_histogram(lat, lon, n_divisions=6):
    """
    球面三角形网格直方图统计。
    
    融合 1306_triangle_histogram 的核心统计思想。
    
    将单位球面投影到八面体的8个球面三角形面上，
    每个面再细分为 n_divisions^2 个小球面三角形，
    统计重力测站在各小三角形内的分布数量。
    
    参数：
        lat: (N,) 纬度 [deg]
        lon: (N,) 经度 [deg]
        n_divisions: 每边细分数
    返回：
        histo: (8 * n_divisions^2,) 每个小三角形的计数
        uniformity_index: 均匀性指数 (0=完全均匀, 1=完全集中)
        expected_count: 期望计数
    """
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    N = len(lat)
    
    # 转为球坐标 (theta=colat, phi=lon)
    theta = np.radians(90.0 - lat)  # 余纬 [rad]
    phi = np.radians(lon)
    
    # 转为笛卡尔坐标
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    
    # 确定每个点属于八面体的哪个面（8个球面三角形）
    # 面的划分基于坐标符号：
    # 面0: x>0, y>0, z>0  (第一卦限)
    # 面1: x<0, y>0, z>0
    # ...
    # 实际用更简单的策略：按 (|x|,|y|,|z|) 的最大分量索引
    
    abs_coords = np.column_stack([np.abs(x), np.abs(y), np.abs(z)])
    max_dim = np.argmax(abs_coords, axis=1)  # 0=x, 1=y, 2=z
    sign = np.sign([x, y, z]).T
    
    # 8个面的编码
    face_id = np.zeros(N, dtype=int)
    for i in range(N):
        sx = 0 if sign[i, 0] > 0 else 1
        sy = 0 if sign[i, 1] > 0 else 1
        sz = 0 if sign[i, 2] > 0 else 1
        face_id[i] = sx * 4 + sy * 2 + sz
    
    # 在每个面内做二维重心坐标细分
    n_tri = n_divisions * n_divisions
    histo = np.zeros(8 * n_tri, dtype=int)
    
    for i in range(N):
        fid = face_id[i]
        # 简化：使用 (u,v) 参数化
        # 将面的两个非主维度归一化到 [0,1]
        md = max_dim[i]
        if md == 0:  # x 主导
            u = abs(y[i]) / (abs(y[i]) + abs(z[i]) + 1e-15)
            v = abs(z[i]) / (abs(y[i]) + abs(z[i]) + 1e-15)
        elif md == 1:  # y 主导
            u = abs(x[i]) / (abs(x[i]) + abs(z[i]) + 1e-15)
            v = abs(z[i]) / (abs(x[i]) + abs(z[i]) + 1e-15)
        else:  # z 主导
            u = abs(x[i]) / (abs(x[i]) + abs(y[i]) + 1e-15)
            v = abs(y[i]) / (abs(x[i]) + abs(y[i]) + 1e-15)
        
        u = np.clip(u, 0.0, 1.0)
        v = np.clip(v, 0.0, 1.0)
        
        # 确定子三角形索引（简化网格）
        iu = min(int(u * n_divisions), n_divisions - 1)
        iv = min(int(v * n_divisions), n_divisions - 1)
        tri_idx = iu * n_divisions + iv
        global_idx = fid * n_tri + tri_idx
        if global_idx < len(histo):
            histo[global_idx] += 1
    
    expected_count = N / (8.0 * n_tri)
    if expected_count > 0:
        variance = np.var(histo)
        max_dev = np.max(np.abs(histo - expected_count))
        uniformity_index = max_dev / (N + 1e-15)
    else:
        uniformity_index = 0.0
    
    return histo, uniformity_index, expected_count


def adaptive_gravity_mesh(obs_lat, obs_lon, obs_gravity,
                           base_nx=20, base_ny=20,
                           cvt_samples=2000, cvt_iter=10):
    """
    生成重力数据驱动的自适应计算网格。
    
    步骤：
      1. 由重力数据估算空间梯度
      2. 构造变度量矩阵
      3. 使用CVT优化网格节点位置
      4. 统计网格均匀性
    
    参数：
        obs_lat, obs_lon: 观测点经纬度 [deg]
        obs_gravity: 观测重力异常 [mGal]
        base_nx, base_ny: 基础网格维度
    返回：
        grid_points: (N, 2) 优化的网格节点 (lat, lon)
        uniformity: 均匀性指数
    """
    # 初始规则网格
    lat_min, lat_max = np.min(obs_lat), np.max(obs_lat)
    lon_min, lon_max = np.min(obs_lon), np.max(obs_lon)
    
    lat_grid = np.linspace(lat_min, lat_max, base_nx)
    lon_grid = np.linspace(lon_min, lon_max, base_ny)
    LAT, LON = np.meshgrid(lat_grid, lon_grid)
    points = np.column_stack([LAT.flatten(), LON.flatten()])
    
    # 估算重力梯度（简化）
    def metric_func(p):
        # 找到最近的观测点
        diffs = obs_lat - p[0] + obs_lon - p[1]  # 简化距离
        idx = np.argmin(np.abs(diffs))
        # 用附近点的重力差异估计梯度
        grad_lat = 0.0
        grad_lon = 0.0
        if idx > 0 and idx < len(obs_gravity) - 1:
            grad_lat = (obs_gravity[idx + 1] - obs_gravity[idx - 1]) / (obs_lat[idx + 1] - obs_lat[idx - 1] + 1e-10)
            grad_lon = (obs_gravity[idx + 1] - obs_gravity[idx - 1]) / (obs_lon[idx + 1] - obs_lon[idx - 1] + 1e-10)
        grad = np.array([grad_lat, grad_lon])
        return adaptive_metric_from_gravity(grad, base_scale=1.0)
    
    # CVT优化
    opt_points, energies = cvt_lloyd_iteration(points, metric_func,
                                                n_samples=cvt_samples,
                                                n_iter=cvt_iter)
    
    # 均匀性统计
    _, uniformity, _ = spherical_triangle_histogram(opt_points[:, 0], opt_points[:, 1], n_divisions=4)
    
    return opt_points, uniformity, energies
