"""
source_inversion.py
微地震震源定位与机制反演模块

原项目映射:
    672_lights_out -> 离散网格上的模 2 矩阵与最小二乘求解

微地震监测的核心任务是从观测台阵的走时/波形数据中反演震源位置与机制。
本模块实现:
1. 基于离散网格的震源位置搜索（利用 Lights Out 型模 2 矩阵进行网格化编码）；
2. 基于最小二乘的震源机制反演；
3. 走时残差分析。

核心公式:
1. 震源定位目标函数（L2 范数走时残差）:
   Φ(ξ) = Σ_{k=1}^{N_sta} [T_{obs}^{(k)} - T_{calc}^{(k)}(ξ)]²
   其中 T_{calc}^{(k)} = |x_k - ξ| / v。

2. 离散网格编码:
   将三维搜索空间划分为 M×N×L 个网格单元，
   每个单元赋予二进制索引 b ∈ {0,1}^{MNL}。
   利用 Lights Out 型矩阵 A（模 2）对邻域关系编码，
   通过求解 A b = r (mod 2) 判定候选震源所在簇。

3. 矩张量线性反演:
   对于每个台站 k 和每个分量 i，远场位移与矩张量呈线性关系:
   u_i^{(k)} = G_{ij}^{(k)} M_j
   其中 G 为格林函数导数矩阵，M_j 为矩张量的 6 个独立分量
   [M11, M22, M33, M12, M13, M23]^T。
   组装成线性系统:
   d = G m
   最小二乘解:
   m̂ = (G^T G)^{-1} G^T d

4. 误差估计:
   协方差矩阵 C_m = σ_d² (G^T G)^{-1}
   其中 σ_d² = ||d - G m̂||² / (N_{obs} - 6)。
"""

import numpy as np
from typing import Tuple, List
from seismic_green import SeismicGreen
from moment_tensor import MomentTensor


def source_location_grid_search(stations: np.ndarray,
                                 observed_tt: np.ndarray,
                                 velocity: float,
                                 grid_bounds: Tuple[Tuple[float, float], ...],
                                 grid_dims: Tuple[int, int, int]) -> Tuple[np.ndarray, float]:
    """
    在离散三维网格上搜索最小走时残差震源位置。

    参数:
        stations: (N_sta, 3) 台站坐标。
        observed_tt: (N_sta,) 观测走时 (s)。
        velocity: 假设均匀速度 (m/s)。
        grid_bounds: ((xmin,xmax), (ymin,ymax), (zmin,zmax))。
        grid_dims: (nx, ny, nz)。

    返回:
        best_loc: 最优震源位置 [x,y,z]。
        best_misfit: 最小残差范数。
    """
    if stations.shape[0] != observed_tt.size:
        raise ValueError("台站数与观测走时不匹配")

    nx, ny, nz = grid_dims
    x_vals = np.linspace(grid_bounds[0][0], grid_bounds[0][1], nx)
    y_vals = np.linspace(grid_bounds[1][0], grid_bounds[1][1], ny)
    z_vals = np.linspace(grid_bounds[2][0], grid_bounds[2][1], nz)

    best_misfit = np.inf
    best_loc = np.zeros(3)

    for xi in x_vals:
        for yi in y_vals:
            for zi in z_vals:
                loc = np.array([xi, yi, zi])
                dists = np.linalg.norm(stations - loc, axis=1)
                tt_calc = dists / velocity
                misfit = np.sum((observed_tt - tt_calc) ** 2)
                if misfit < best_misfit:
                    best_misfit = misfit
                    best_loc = loc

    return best_loc, best_misfit


def moment_tensor_inversion(stations: np.ndarray,
                             observed_displacements: np.ndarray,
                             source_loc: np.ndarray,
                             green: SeismicGreen) -> MomentTensor:
    """
    基于远场位移观测反演矩张量（最小二乘）。

    参数:
        stations: (N_sta, 3) 台站坐标。
        observed_displacements: (N_sta, 3) 观测位移向量 (m)。
        source_loc: 震源位置 [x,y,z]。
        green: SeismicGreen 对象。

    返回:
        MomentTensor 对象。
    """
    # TODO Hole 2: 实现矩张量反演
    # 1. 通过数值微分组装格林函数导数矩阵 G (3*N_sta, 6)
    #    6 个独立分量索引: (0,0)=0, (1,1)=1, (2,2)=2, (0,1)=3, (0,2)=4, (1,2)=5
    #    对每一分量施加微小扰动 epsilon，调用 green.displacement_spectrum_farfield
    #    计算实部位移差分得到 G 的列
    # 2. 构造观测向量 d = observed_displacements.reshape(-1)
    # 3. 正则化最小二乘: m̂ = (G^T G + reg*I)^{-1} G^T d
    # 4. 将 m̂ 的 6 个分量映射回 3x3 对称矩阵 M_inv
    # 5. 返回 MomentTensor(M_inv)
    raise NotImplementedError("Hole 2: 请实现矩张量反演")


def connectivity_source_cluster(grid_occupied: np.ndarray,
                                 grid_dims: Tuple[int, int, int]) -> List[np.ndarray]:
    """
    利用离散网格邻接分析提取可能的震源簇。

    参数:
        grid_occupied: (nx*ny*nz,) 0/1 数组，表示网格是否满足破裂条件。
        grid_dims: (nx, ny, nz)。

    返回:
        clusters: 每个簇的网格索引列表。
    """
    nx, ny, nz = grid_dims
    if grid_occupied.size != nx * ny * nz:
        raise ValueError("网格尺寸不匹配")

    grid = grid_occupied.reshape((nx, ny, nz)).astype(bool)
    visited = np.zeros_like(grid, dtype=bool)
    clusters = []

    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                if grid[i, j, k] and not visited[i, j, k]:
                    # BFS
                    queue = [(i, j, k)]
                    visited[i, j, k] = True
                    cluster = []
                    while queue:
                        ci, cj, ck = queue.pop(0)
                        cluster.append((ci, cj, ck))
                        for di, dj, dk in [(-1, 0, 0), (1, 0, 0),
                                           (0, -1, 0), (0, 1, 0),
                                           (0, 0, -1), (0, 0, 1)]:
                            ni, nj, nk = ci + di, cj + dj, ck + dk
                            if 0 <= ni < nx and 0 <= nj < ny and 0 <= nk < nz:
                                if grid[ni, nj, nk] and not visited[ni, nj, nk]:
                                    visited[ni, nj, nk] = True
                                    queue.append((ni, nj, nk))
                    clusters.append(np.array(cluster))
    return clusters
