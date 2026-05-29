"""
sampling_utils.py
准随机采样与空间统计模块

融合以下种子项目的核心算法：
  - 498_hammersley：低差异准随机序列
  - 236_cube_surface_distance：立方体/球面表面距离统计

物理背景：
  在重力正演计算中，蒙特卡洛积分效率依赖于采样点的空间分布均匀性。
  低差异序列（如Hammersley、Halton、Sobol）具有 O((log N)^d / N) 的
  星差异收敛率，远优于纯随机采样的 O(1/sqrt(N))。
  
  本模块提供：
  1. 多维Hammersley和Halton序列
  2. 球面均匀采样（用于全球重力测站分布模拟）
  3. 球面距离统计（评估数据覆盖质量）
"""

import numpy as np


def radical_inverse(n, base):
    """
    计算整数 n 在素数 base 下的radical inverse（van der Corput序列）。
    
    公式：
        phi_b(n) = sum_{k=0}^{\infty} d_k * b^{-(k+1)}
    其中 n = sum d_k * b^k。
    """
    result = 0.0
    inv_base = 1.0 / base
    f = inv_base
    while n > 0:
        d = n % base
        result += d * f
        f *= inv_base
        n //= base
    return result


def halton_sequence(dim, n_points, offset=0):
    """
    生成Halton低差异序列。
    
    前 dim 个素数作为基：
        x_i(j) = phi_{p_j}(i)
    
    参数：
        dim: 维度
        n_points: 点数
        offset: 偏移
    返回：
        seq: (n_points, dim) 序列
    """
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    if dim > len(primes):
        raise ValueError("dim {} exceeds available primes".format(dim))
    
    seq = np.zeros((n_points, dim))
    for i in range(n_points):
        idx = i + offset
        for j in range(dim):
            seq[i, j] = radical_inverse(idx, primes[j])
    
    return seq


def hammersley_sequence_nd(dim, n_points, offset=0):
    """
    生成多维Hammersley序列。
    
    第一维：i / N
    其余维：phi_{p_j}(i)
    """
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    if dim > len(primes) + 1:
        raise ValueError("dim too large")
    
    seq = np.zeros((n_points, dim))
    for i in range(n_points):
        idx = i + offset
        if n_points > 0:
            seq[i, 0] = (idx % (n_points + 1)) / n_points
        else:
            seq[i, 0] = 0.0
        for j in range(1, dim):
            seq[i, j] = radical_inverse(idx, primes[j - 1])
    
    return seq


def sphere_surface_uniform(n_points, method='fibonacci'):
    """
    生成球面均匀分布的点。
    
    方法：
      'fibonacci': Fibonacci螺旋网格
      'hammersley': Hammersley序列投影
      'random': 均匀随机
    
    Fibonacci螺旋公式：
        theta = arccos(1 - 2*i/N)
        phi = 2 * pi * i / phi_golden
    其中 phi_golden = (1 + sqrt(5)) / 2。
    """
    if method == 'fibonacci':
        phi_golden = (1.0 + np.sqrt(5.0)) / 2.0
        indices = np.arange(n_points, dtype=float)
        theta = np.arccos(1.0 - 2.0 * (indices + 0.5) / n_points)
        phi = 2.0 * np.pi * indices / phi_golden
        
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        
        return np.column_stack([x, y, z])
    
    elif method == 'hammersley':
        seq = hammersley_sequence_nd(2, n_points)
        theta = np.arccos(1.0 - 2.0 * seq[:, 0])
        phi = 2.0 * np.pi * seq[:, 1]
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        return np.column_stack([x, y, z])
    
    elif method == 'random':
        phi = 2.0 * np.pi * np.random.rand(n_points)
        cos_theta = 2.0 * np.random.rand(n_points) - 1.0
        theta = np.arccos(cos_theta)
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        return np.column_stack([x, y, z])
    
    else:
        raise ValueError("Unknown method: {}".format(method))


def great_circle_distance(p1, p2, radius=6371e3):
    """
    计算球面上两点的大圆距离（Haversine公式）。
    
    公式：
        a = sin^2(dphi/2) + cos(phi1)*cos(phi2)*sin^2(dlambda/2)
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        d = R * c
    """
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    
    if p1.ndim == 1 and p2.ndim == 1:
        lat1, lon1 = np.radians(p1[0]), np.radians(p1[1])
        lat2, lon2 = np.radians(p2[0]), np.radians(p2[1])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
        a = min(1.0, max(0.0, a))
        c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
        return radius * c
    else:
        # 批量计算
        lat1 = np.radians(p1[:, 0])
        lon1 = np.radians(p1[:, 1])
        lat2 = np.radians(p2[:, 0])
        lon2 = np.radians(p2[:, 1])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
        a = np.clip(a, 0.0, 1.0)
        c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
        return radius * c


def sphere_distance_statistics(points, n_pairs=None):
    """
    计算球面上点集的距离统计。
    
    融合 236_cube_surface_distance 的统计思想。
    
    参数：
        points: (N, 3) 单位球面上的点，或 (N, 2) 经纬度 [deg]
        n_pairs: 随机采样对数（None表示全计算）
    返回：
        mean_dist: 平均弦长距离
        std_dist: 标准差
        min_dist: 最小距离
        max_dist: 最大距离
    """
    points = np.asarray(points, dtype=float)
    
    if points.shape[1] == 2:
        # 经纬度转笛卡尔
        lat, lon = np.radians(points[:, 0]), np.radians(points[:, 1])
        theta = np.pi / 2.0 - lat
        x = np.sin(theta) * np.cos(lon)
        y = np.sin(theta) * np.sin(lon)
        z = np.cos(theta)
        cart = np.column_stack([x, y, z])
    else:
        cart = points / np.linalg.norm(points, axis=1, keepdims=True)
    
    N = cart.shape[0]
    
    if n_pairs is None or n_pairs >= N * (N - 1) // 2:
        # 全计算
        dists = []
        for i in range(N):
            for j in range(i + 1, N):
                d = np.linalg.norm(cart[i] - cart[j])
                dists.append(d)
        dists = np.array(dists)
    else:
        # 随机采样
        dists = np.zeros(n_pairs)
        for k in range(n_pairs):
            i, j = np.random.choice(N, 2, replace=False)
            dists[k] = np.linalg.norm(cart[i] - cart[j])
    
    if len(dists) == 0:
        return 0.0, 0.0, 0.0, 0.0
    
    return np.mean(dists), np.std(dists), np.min(dists), np.max(dists)


def generate_gravity_station_network(n_stations, method='fibonacci',
                                      lat_range=(-60, 60), lon_range=(0, 360)):
    """
    生成重力测站网络的空间分布。
    
    参数：
        n_stations: 测站数
        method: 分布方法
        lat_range, lon_range: 纬度/经度范围 [deg]
    返回：
        stations: (n_stations, 2) [lat, lon]
    """
    if method == 'uniform_grid':
        n_lat = int(np.sqrt(n_stations))
        n_lon = int(np.ceil(n_stations / n_lat))
        lats = np.linspace(lat_range[0], lat_range[1], n_lat)
        lons = np.linspace(lon_range[0], lon_range[1], n_lon)
        LAT, LON = np.meshgrid(lats, lons)
        stations = np.column_stack([LAT.flatten(), LON.flatten()])[:n_stations]
    elif method == 'random':
        lats = np.random.uniform(lat_range[0], lat_range[1], n_stations)
        lons = np.random.uniform(lon_range[0], lon_range[1], n_stations)
        stations = np.column_stack([lats, lons])
    else:
        # Fibonacci球面采样后投影到lat/lon
        sphere_pts = sphere_surface_uniform(n_stations, method='fibonacci')
        # 转为lat/lon
        x, y, z = sphere_pts[:, 0], sphere_pts[:, 1], sphere_pts[:, 2]
        lat = 90.0 - np.degrees(np.arccos(np.clip(z, -1.0, 1.0)))
        lon = np.degrees(np.arctan2(y, x))
        lon = np.mod(lon, 360.0)
        # 筛选范围
        mask = (lat >= lat_range[0]) & (lat <= lat_range[1]) & \
               (lon >= lon_range[0]) & (lon <= lon_range[1])
        stations = np.column_stack([lat, lon])
        if not np.all(mask):
            # 补足数量
            n_missing = n_stations - np.sum(mask)
            extra_lats = np.random.uniform(lat_range[0], lat_range[1], n_missing)
            extra_lons = np.random.uniform(lon_range[0], lon_range[1], n_missing)
            stations = np.vstack([stations[mask], np.column_stack([extra_lats, extra_lons])])
    
    return stations[:n_stations]
