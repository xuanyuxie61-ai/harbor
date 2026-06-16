"""
网格生成模块

融合以下种子项目:
1. ellipsoid_grid: 椭球体网格生成
2. circle_distance: 球面距离统计

应用于 PIC 中的:
- 非均匀映射网格生成 (用于聚焦束模拟)
- 速度空间椭球网格 (相对论粒子)
- 相空间网格质量检测
"""

import numpy as np
from typing import Tuple, List, Callable
try:
    from plasma_constants import PI
except ImportError:
    from plasma_constants import PI


# ============================================================================
# 椭球网格 (from ellipsoid_grid)
# ============================================================================
def ellipsoid_grid_count(a: float, b: float, c: float,
                           spacing: float) -> int:
    """
    估计椭球 (x/a)^2 + (y/b)^2 + (z/c)^2 <= 1 内间距 spacing 的网格点数

    近似公式: N ≈ (4/3)*pi*a*b*c / spacing^3
    """
    volume = (4.0 / 3.0) * PI * a * b * c
    return max(1, int(volume / spacing**3))


def ellipsoid_grid_generate(a: float, b: float, c: float,
                               nx: int, ny: int, nz: int) -> np.ndarray:
    """
    生成椭球体内的网格点

    (x/a)^2 + (y/b)^2 + (z/c)^2 <= 1

    Parameters:
        a, b, c  : 半轴
        nx,ny,nz : 各维网格点数

    Returns:
        points : (N, 3) 数组
    """
    x = np.linspace(-a, a, nx)
    y = np.linspace(-b, b, ny)
    z = np.linspace(-c, c, nz)

    points = []
    for xi in x:
        for yi in y:
            for zi in z:
                if (xi / a)**2 + (yi / b)**2 + (zi / c)**2 <= 1.0:
                    points.append([xi, yi, zi])

    return np.array(points) if points else np.zeros((0, 3))


def ellipsoid_surface_grid(a: float, b: float, c: float,
                              n_theta: int, n_phi: int) -> np.ndarray:
    """
    椭球表面网格 (参数化):
        x = a * sin(theta) * cos(phi)
        y = b * sin(theta) * sin(phi)
        z = c * cos(theta)
    """
    theta = np.linspace(0, PI, n_theta)
    phi = np.linspace(0, 2 * PI, n_phi, endpoint=False)

    points = []
    for t in theta:
        for p in phi:
            x = a * np.sin(t) * np.cos(p)
            y = b * np.sin(t) * np.sin(p)
            z = c * np.cos(t)
            points.append([x, y, z])

    return np.array(points)


def ellipsoid_volume(a: float, b: float, c: float) -> float:
    """椭球体积: V = (4/3)*pi*a*b*c"""
    return (4.0 / 3.0) * PI * a * b * c


# ============================================================================
# 映射网格 (用于 PIC)
# ============================================================================
def uniform_grid_1d(L: float, N: int) -> Tuple[np.ndarray, float]:
    """
    一维均匀网格:
        x_i = i * dx,  i = 0, ..., N-1
        dx = L / N
    """
    dx = L / N
    x = np.arange(N) * dx
    return x, dx


def mapped_grid_1d(L: float, N: int,
                     mapping: Callable[[float], float] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    一维映射网格: 通过 xi -> x(xi) 映射生成非均匀网格

    典型映射:
        拉伸: x = L/2 + (L/2) * tanh(alpha * (2*xi/L - 1)) / tanh(alpha)
        压缩: x = L/2 * (1 + sin(pi*(2*xi/L - 1)/2))
    """
    xi = np.linspace(0, L, N)
    if mapping is None:
        return xi, np.full(N, L / N)

    x = np.array([mapping(xii) for xii in xi])
    dx = np.diff(x)
    dx = np.append(dx, dx[-1])
    return x, dx


def tanh_stretching_grid(L: float, N: int, alpha: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    双曲正切拉伸网格 (两端密集):
        x(xi) = (L/2) * [1 + tanh(alpha * (2*xi/L - 1)) / tanh(alpha)]
    """
    def mapping(xi):
        return (L / 2.0) * (1.0 + np.tanh(alpha * (2.0 * xi / L - 1.0)) / np.tanh(alpha))
    return mapped_grid_1d(L, N, mapping)


def chebyshev_grid_1d(L: float, N: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Chebyshev 网格 (两端密集, 谱方法常用):
        x_j = (L/2) * (1 - cos(pi * j / (N-1)))
    """
    j = np.arange(N)
    x = (L / 2.0) * (1.0 - np.cos(PI * j / (N - 1)))
    dx = np.diff(x)
    dx = np.append(dx, dx[-1])
    return x, dx


def velocity_ellipsoid_grid(v_th_x: float, v_th_y: float, v_th_z: float,
                               n_v: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    速度空间椭球网格 (用于速度分布函数初始化):
        (v_x/v_th_x)^2 + (v_y/v_th_y)^2 + (v_z/v_th_z)^2 <= R_max^2

    返回网格点与权重
    """
    R_max = 4.0  # 截断: 4 sigma
    nv_each = max(3, n_v)

    vx = np.linspace(-R_max * v_th_x, R_max * v_th_x, nv_each)
    vy = np.linspace(-R_max * v_th_y, R_max * v_th_y, nv_each)
    vz = np.linspace(-R_max * v_th_z, R_max * v_th_z, nv_each)

    points = []
    weights = []
    dvx = vx[1] - vx[0] if nv_each > 1 else 1.0
    dvy = vy[1] - vy[0] if nv_each > 1 else 1.0
    dvz = vz[1] - vz[0] if nv_each > 1 else 1.0

    for vx_i in vx:
        for vy_i in vy:
            for vz_i in vz:
                if ((vx_i / v_th_x)**2 + (vy_i / v_th_y)**2 +
                        (vz_i / v_th_z)**2 <= R_max**2):
                    points.append([vx_i, vy_i, vz_i])
                    # 权重 = 高斯权重
                    w = np.exp(-(vx_i**2 / (2 * v_th_x**2) +
                                  vy_i**2 / (2 * v_th_y**2) +
                                  vz_i**2 / (2 * v_th_z**2)))
                    weights.append(w * dvx * dvy * dvz)

    if not points:
        return np.zeros((0, 3)), np.array([])
    return np.array(points), np.array(weights)


def phase_space_grid(x_min: float, x_max: float, N_x: int,
                       v_min: float, v_max: float, N_v: int
                       ) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    (x, v) 相空间网格

    Returns:
        (x_grid, v_grid, dx, dv)
    """
    x, dx = uniform_grid_1d(x_max - x_min, N_x)
    x += x_min
    v, dv = uniform_grid_1d(v_max - v_min, N_v)
    v += v_min
    return x, v, dx, dv


# ============================================================================
# 网格质量评估
# ============================================================================
def grid_quality_metrics(x: np.ndarray, dx: np.ndarray) -> dict:
    """
    计算网格质量指标:
    - 最大/最小间距比
    - 间距变化率
    - 网格 Reynolds 数
    """
    if len(dx) < 2:
        return {'ratio': 1.0, 'variation': 0.0}

    dx_min = np.min(dx)
    dx_max = np.max(dx)
    ratio = dx_max / (dx_min + 1e-300)

    # 间距变化率
    variation = np.std(dx) / (np.mean(dx) + 1e-300)

    return {
        'dx_min': dx_min,
        'dx_max': dx_max,
        'ratio': ratio,
        'variation': variation,
        'n_points': len(x)
    }


def minimum_grid_resolution(lambda_D: float, n_points_per_debye: int = 10) -> float:
    """
    PIC 最小网格分辨率要求:
        dx <= lambda_D / n_points_per_debye

    避免数值 Cherenkov 辐射和有限网格不稳定性
    """
    return lambda_D / n_points_per_debye


def aliasing_check(k_max: float, dx: float) -> bool:
    """
    混叠检查: 最大波数是否超过 Nyquist 波数
        k_max <= pi / dx
    """
    k_nyquist = PI / dx
    return k_max <= k_nyquist
