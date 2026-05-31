"""
sparse_interpolation.py
================================================================================
稀疏海洋观测数据的 N 维 Shepard 插值

融合项目：
    - 1073_shepard_interp_nd : N 维 Shepard 插值

核心科学问题：
    海洋观测数据在空间上极度稀疏（ARGO 浮标、航次站位）。Shepard 插值
    是一种基于反距离加权的无网格插值方法，适用于不规则分布数据点，
    可用于将离散的 pH、DIC、TA 测量插值到整个三维海洋网格。

科学背景：
    给定 N 个数据点 {x_i ∈ ℝ^m} 及其观测值 {z_i}，Shepard 插值在查询点
    x 处的值为：
    
        z(x) = Σ_{i=1}^{N} w_i(x) · z_i
    
    其中权重为：
        w_i(x) = ||x - x_i||^{-p} / Σ_{j=1}^{N} ||x - x_j||^{-p}
    
    参数 p 控制平滑度：
        p = 0 : 等权重平均（全局常数）
        p = 2 : 经典反距离平方（常用）
        p > 2 : 更强调近邻点的影响
    
    当 x 恰好与某数据点 x_i 重合时，令 w_i = 1，其余为 0。
    
    数值稳定性：当 ||x - x_i|| < ε 时，直接返回 z_i。

================================================================================
"""

import numpy as np


def shepard_interp_nd(data_coords, data_values, p, query_coords):
    """
    N 维 Shepard 反距离插值。
    
    参数:
        data_coords  : ndarray, shape (m, nd), m 维空间中 nd 个数据点坐标
        data_values  : ndarray, shape (nd,), 数据值
        p            : float, 距离幂次 (通常取 2.0)
        query_coords : ndarray, shape (m, ni), 查询点坐标
    
    返回:
        interp_values : ndarray, shape (ni,), 插值结果
    """
    m, nd = data_coords.shape
    _, ni = query_coords.shape
    
    if nd == 0:
        return np.zeros(ni)
    
    interp_values = np.zeros(ni)
    
    for q in range(ni):
        xq = query_coords[:, q]
        
        # 计算到所有数据点的欧氏距离
        diff = data_coords - xq[:, np.newaxis]
        dists = np.sqrt(np.sum(diff**2, axis=0))
        
        # 检查是否恰好落在数据点上
        min_dist = np.min(dists)
        if min_dist < 1e-12:
            idx = np.argmin(dists)
            interp_values[q] = data_values[idx]
            continue
        
        # 计算权重
        if p == 0.0:
            weights = np.ones(nd) / nd
        else:
            weights = dists**(-p)
            # 处理可能的数值溢出
            w_max = np.max(weights)
            if w_max > 1e30:
                weights = np.where(weights > 1e-30, weights, 0.0)
            w_sum = np.sum(weights)
            if w_sum < 1e-30:
                weights = np.ones(nd) / nd
            else:
                weights = weights / w_sum
        
        interp_values[q] = np.dot(weights, data_values)
    
    return interp_values


def shepard_interp_3d_ocean(data_lons, data_lats, data_depths, data_values,
                            query_lons, query_lats, query_depths, p=2.0,
                            lon_scale=1.0, lat_scale=1.0, depth_scale=1.0):
    """
    专门为三维海洋观测设计的 Shepard 插值。
    
    对经度、纬度、深度进行尺度变换以平衡各维度的贡献：
        x' = lon / lon_scale
        y' = lat / lat_scale
        z' = depth / depth_scale
    
    参数:
        data_lons, data_lats, data_depths : ndarray, 观测点坐标
        data_values                        : ndarray, 观测值
        query_lons, query_lats, query_depths : ndarray, 查询点坐标
        p                                  : float, 幂次
        *_scale                            : float, 各维度尺度因子
    
    返回:
        ndarray, 插值结果
    """
    nd = len(data_lons)
    ni = len(query_lons)
    
    data_coords = np.zeros((3, nd))
    data_coords[0, :] = data_lons / lon_scale
    data_coords[1, :] = data_lats / lat_scale
    data_coords[2, :] = data_depths / depth_scale
    
    query_coords = np.zeros((3, ni))
    query_coords[0, :] = query_lons / lon_scale
    query_coords[1, :] = query_lats / lat_scale
    query_coords[2, :] = query_depths / depth_scale
    
    return shepard_interp_nd(data_coords, data_values, p, query_coords)


def kriging_like_shepard_residual(data_coords, data_values, trend_func, p=2.0):
    """
    类似 Kriging 的两步插值：先拟合趋势，再对残差做 Shepard 插值。
    
    z(x) = trend(x) + Shepard_residual(x)
    
    参数:
        data_coords  : ndarray, shape (m, nd)
        data_values  : ndarray, shape (nd,)
        trend_func   : callable, trend(x) -> float, x shape (m,)
        p            : float
    
    返回:
        callable: interp_func(query_coords) -> values
    """
    # 计算趋势和残差
    nd = data_coords.shape[1]
    trend_vals = np.zeros(nd)
    for i in range(nd):
        trend_vals[i] = trend_func(data_coords[:, i])
    
    residuals = data_values - trend_vals
    
    def interp_func(query_coords):
        ni = query_coords.shape[1]
        trend_at_query = np.zeros(ni)
        for q in range(ni):
            trend_at_query[q] = trend_func(query_coords[:, q])
        
        residual_interp = shepard_interp_nd(data_coords, residuals, p, query_coords)
        return trend_at_query + residual_interp
    
    return interp_func


def cross_validate_shepard(data_coords, data_values, p_values=[1.0, 2.0, 3.0, 4.0]):
    """
    留一交叉验证选择最优 Shepard 幂次 p。
    
    返回:
        dict: {'best_p': 最优 p, 'rmse_by_p': 各 p 的 RMSE}
    """
    nd = data_coords.shape[1]
    if nd < 3:
        return {'best_p': 2.0, 'rmse_by_p': {2.0: 0.0}}
    
    rmse_by_p = {}
    
    for p in p_values:
        errors = []
        for i in range(nd):
            # 留一
            mask = np.ones(nd, dtype=bool)
            mask[i] = False
            train_coords = data_coords[:, mask]
            train_vals = data_values[mask]
            
            query = data_coords[:, i:i+1]
            pred = shepard_interp_nd(train_coords, train_vals, p, query)
            errors.append((pred[0] - data_values[i])**2)
        
        rmse = np.sqrt(np.mean(errors))
        rmse_by_p[p] = rmse
    
    best_p = min(rmse_by_p, key=rmse_by_p.get)
    return {'best_p': best_p, 'rmse_by_p': rmse_by_p}


def generate_sparse_ocean_observations(n_points=50, depth_range=(0, 4000),
                                        lat_range=(20, 60), lon_range=(-80, -20),
                                        seed=None):
    """
    生成模拟的稀疏海洋观测数据。
    
    生成符合物理规律的 DIC 场：
        DIC(z, lat) ≈ DIC_surf + ΔDIC·(1 - exp(-z/z_scale))
                      + lat_perturb·sin(2π·lat/30)
    
    返回:
        dict: 包含 lons, lats, depths, DIC, TA, T, S
    """
    if seed is not None:
        np.random.seed(seed)
    
    lons = np.random.uniform(lon_range[0], lon_range[1], n_points)
    lats = np.random.uniform(lat_range[0], lat_range[1], n_points)
    depths = np.random.uniform(depth_range[0], depth_range[1], n_points)
    
    # 物理上合理的 DIC 剖面
    DIC_surf = 2000.0  # μmol/kg
    delta_DIC = 300.0
    z_scale = 800.0
    DIC = DIC_surf + delta_DIC * (1.0 - np.exp(-depths / z_scale))
    DIC += np.random.normal(0, 20, n_points)
    
    # TA 与 DIC 相关但更高
    TA = DIC + 100.0 + np.random.normal(0, 15, n_points)
    
    # 温度随深度递减
    T_surf = 20.0
    T = T_surf * np.exp(-depths / 200.0) + 2.0 + np.random.normal(0, 0.5, n_points)
    T = np.clip(T, -2.0, 35.0)
    
    # 盐度
    S = 34.5 + 0.5 * np.sin(np.radians(lats)) + np.random.normal(0, 0.2, n_points)
    S = np.clip(S, 30.0, 38.0)
    
    return {
        'lons': lons,
        'lats': lats,
        'depths': depths,
        'DIC': DIC,
        'TA': TA,
        'T': T,
        'S': S,
    }
