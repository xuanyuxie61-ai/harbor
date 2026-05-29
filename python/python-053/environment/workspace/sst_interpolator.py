"""
sst_interpolator.py
===================
基于 lagrange_interp_2d (636_lagrange_interp_2d) 的二维 Lagrange 插值算法，
用于海表温度 (SST) 场在经度-纬度非均匀网格上的高精度空间重构。

科学背景
--------
卫星遥感 SST 数据（如 OISST、HadISST）通常存在于规则的经度-纬度网格上，
而海洋环流模式的计算网格可能是曲线坐标或非均匀分布的。为了将观测 SST
驱动耦合模式，或在模式网格间传递信息，需要高阶空间插值。

Lagrange 张量积插值在双多项式空间 Π_m ⊗ Π_n 中是唯一确定的，
对于解析光滑的 SST 场可达到谱精度收敛。

核心公式
--------
1. 一维 Lagrange 基函数：
   对于节点 {x_j}_{j=0}^{m}，第 i 个基函数为
   
   L_i(x) = Π_{j≠i} (x - x_j) / (x_i - x_j)

2. 二维张量积插值：
   给定矩形网格数据 {z_{ij}}，在点 (x, y) 处的插值为
   
   P(x, y) = Σ_{i=0}^{m} Σ_{j=0}^{n} z_{ij} * L_i(x) * L_j(y)

3. 插值误差估计（解析函数，Jackson 定理）：
   若 f ∈ C^{m+n+2}，则
   
   |f(x, y) - P(x, y)| ≤ C * h^{m+1} + D * k^{n+1}

   其中 h, k 分别为 x, y 方向的最大节点间距。

4. SST 距平 (Anomaly) 计算：
   
   SST_A(x, y, t) = SST(x, y, t) - climatology(x, y, month(t))

   Niño 3.4 指数：
   
   Niño3.4(t) = <SST_A>_{region(170°W–120°W, 5°S–5°N)}
"""

import numpy as np
from typing import Tuple


def lagrange_basis_1d(m: int, xd: np.ndarray, i: int, x: float) -> float:
    """
    计算一维 Lagrange 基函数 L_i(x)。

    公式：
    L_i(x) = Π_{j=0, j≠i}^{m} (x - x_j) / (x_i - x_j)

    参数
    ----
    m : int
        多项式次数（节点数 = m+1）。
    xd : np.ndarray, shape (m+1,)
        插值节点。
    i : int
        基函数索引，0 ≤ i ≤ m。
    x : float
        求值点。

    返回
    ----
    value : float
        基函数值。
    """
    if i < 0 or i > m:
        raise ValueError("Index i out of range")

    # 边界处理：若 x 恰好等于某节点，返回 Kronecker delta
    if np.isclose(x, xd[i]):
        return 1.0

    # 检查是否与其他节点重合（病态条件）
    for j in range(m + 1):
        if j != i and np.isclose(xd[i], xd[j]):
            raise ValueError(f"Duplicate nodes detected: xd[{i}] == xd[{j}]")

    value = 1.0
    for j in range(m + 1):
        if j != i:
            denom = xd[i] - xd[j]
            if abs(denom) < 1e-14:
                raise ValueError("Denominator too small in Lagrange basis")
            value *= (x - xd[j]) / denom
    return float(value)


def lagrange_interp_2d(mx: int, my: int,
                       xd_1d: np.ndarray, yd_1d: np.ndarray,
                       zd: np.ndarray,
                       xi: np.ndarray, yi: np.ndarray) -> np.ndarray:
    """
    二维张量积 Lagrange 插值。

    参数
    ----
    mx, my : int
        x, y 方向多项式次数。
    xd_1d : np.ndarray, shape (mx+1,)
        x 方向节点。
    yd_1d : np.ndarray, shape (my+1,)
        y 方向节点。
    zd : np.ndarray, shape ((mx+1)*(my+1),)
        网格点数据，按行优先存储：
        zd[l] = z_{i,j}, 其中 l = (j-1)*(mx+1) + (i-1)。
    xi, yi : np.ndarray, shape (ni,)
        待插值点坐标。

    返回
    ----
    zi : np.ndarray, shape (ni,)
        插值结果。
    """
    if xd_1d.shape[0] != mx + 1:
        raise ValueError("xd_1d length must be mx+1")
    if yd_1d.shape[0] != my + 1:
        raise ValueError("yd_1d length must be my+1")
    if zd.shape[0] != (mx + 1) * (my + 1):
        raise ValueError("zd length must be (mx+1)*(my+1)")
    if xi.shape != yi.shape:
        raise ValueError("xi and yi must have the same shape")

    ni = xi.shape[0]
    zi = np.zeros(ni, dtype=float)

    for k in range(ni):
        # 检查插值点是否在节点凸包内（鲁棒性）
        x_min, x_max = xd_1d.min(), xd_1d.max()
        y_min, y_max = yd_1d.min(), yd_1d.max()
        if xi[k] < x_min - 1e-10 or xi[k] > x_max + 1e-10:
            # 外推警告：衰减处理
            scale = 0.0
        elif yi[k] < y_min - 1e-10 or yi[k] > y_max + 1e-10:
            scale = 0.0
        else:
            scale = 1.0

        val = 0.0
        l = 0
        for i_idx in range(mx + 1):
            lx = lagrange_basis_1d(mx, xd_1d, i_idx, xi[k])
            for j_idx in range(my + 1):
                ly = lagrange_basis_1d(my, yd_1d, j_idx, yi[k])
                val += zd[l] * lx * ly
                l += 1
        zi[k] = val * scale

    return zi


def chebyshev_nodes_1d(n: int, a: float, b: float) -> np.ndarray:
    """
    生成区间 [a, b] 上的 Chebyshev 节点。

    公式：
    x_k = (a+b)/2 + (b-a)/2 * cos(π * (2k+1) / (2n)),  k=0,...,n-1

    Chebyshev 节点可最小化 Runge 现象，提高插值稳定性。
    """
    if n < 1:
        raise ValueError("n must be at least 1")
    if b <= a:
        raise ValueError("b must be greater than a")
    k = np.arange(n)
    nodes = 0.5 * (a + b) + 0.5 * (b - a) * np.cos(np.pi * (2.0 * k + 1.0) / (2.0 * n))
    return nodes


def interpolate_sst_field(lon_grid: np.ndarray, lat_grid: np.ndarray,
                          sst_data: np.ndarray,
                          lon_target: np.ndarray, lat_target: np.ndarray,
                          degree_lon: int = 5, degree_lat: int = 5) -> np.ndarray:
    """
    对 SST 场进行二维 Lagrange 插值重构。

    参数
    ----
    lon_grid : np.ndarray, shape (nx,)
        原始经度网格（度）。
    lat_grid : np.ndarray, shape (ny,)
        原始纬度网格（度）。
    sst_data : np.ndarray, shape (nx, ny)
        原始 SST 数据（℃）。
    lon_target, lat_target : np.ndarray, shape (ni,)
        目标插值点。
    degree_lon, degree_lat : int
        每个子区域插值多项式次数。

    返回
    ----
    sst_interp : np.ndarray, shape (ni,)
        插值后的 SST。
    """
    if lon_grid.ndim != 1 or lat_grid.ndim != 1:
        raise ValueError("lon_grid and lat_grid must be 1D")
    if sst_data.shape != (lon_grid.shape[0], lat_grid.shape[0]):
        raise ValueError("sst_data shape mismatch with grid dimensions")

    ni = lon_target.shape[0]
    sst_interp = np.zeros(ni, dtype=float)

    # 分块插值：将大区域划分为若干子块，每块独立插值
    nx, ny = lon_grid.shape[0], lat_grid.shape[0]
    n_block_x = max(1, nx // (degree_lon + 1))
    n_block_y = max(1, ny // (degree_lat + 1))

    for k in range(ni):
        x, y = lon_target[k], lat_target[k]

        # 找到包含 (x, y) 的子块
        ix = min(n_block_x - 1, max(0, int((x - lon_grid[0]) / (lon_grid[-1] - lon_grid[0]) * n_block_x)))
        iy = min(n_block_y - 1, max(0, int((y - lat_grid[0]) / (lat_grid[-1] - lat_grid[0]) * n_block_y)))

        i0 = ix * (degree_lon + 1)
        i1 = min(nx, i0 + degree_lon + 1)
        j0 = iy * (degree_lat + 1)
        j1 = min(ny, j0 + degree_lat + 1)

        if i1 - i0 < 2 or j1 - j0 < 2:
            # 退化情况：最近邻
            i_near = np.argmin(np.abs(lon_grid - x))
            j_near = np.argmin(np.abs(lat_grid - y))
            sst_interp[k] = sst_data[i_near, j_near]
            continue

        # 选取子块节点
        xd = lon_grid[i0:i1]
        yd = lat_grid[j0:j1]
        mx = xd.shape[0] - 1
        my = yd.shape[0] - 1

        # 展平数据
        zd = np.zeros((mx + 1) * (my + 1), dtype=float)
        l = 0
        for jj in range(j0, j1):
            for ii in range(i0, i1):
                zd[l] = sst_data[ii, jj]
                l += 1

        # 单点插值
        result = lagrange_interp_2d(mx, my, xd, yd, zd,
                                     np.array([x]), np.array([y]))
        sst_interp[k] = result[0]

    return sst_interp


def nino34_index(sst_anomaly: np.ndarray, lon: np.ndarray, lat: np.ndarray) -> float:
    """
    计算 Niño 3.4 指数。

    区域定义：170°W – 120°W, 5°S – 5°N。

    公式：
    Niño3.4 = (1/A) ∬_{region} SST_A(x, y) dx dy

    参数
    ----
    sst_anomaly : np.ndarray, shape (nx, ny)
        SST 距平场。
    lon : np.ndarray, shape (nx,)
        经度网格。
    lat : np.ndarray, shape (ny,)
        纬度网格。

    返回
    ----
    index : float
        Niño 3.4 指数（℃）。
    """
    if sst_anomaly.shape != (lon.shape[0], lat.shape[0]):
        raise ValueError("Array shape mismatch")

    # 区域掩码
    mask = ((lon >= -170.0) & (lon <= -120.0))[:, None] & \
           ((lat >= -5.0) & (lat <= 5.0))[None, :]

    if not np.any(mask):
        return 0.0

    # 考虑纬度余弦权重（等面积近似）
    cos_lat = np.cos(np.radians(lat))[None, :]
    weights = mask.astype(float) * cos_lat
    total_weight = np.sum(weights)

    if total_weight < 1e-14:
        return 0.0

    index = np.sum(sst_anomaly * weights) / total_weight
    return float(index)
