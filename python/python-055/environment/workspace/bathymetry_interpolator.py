"""
bathymetry_interpolator.py
基于种子项目 1071_shepard_interp_1d（Shepard 插值），
扩展构建二维海底地形插值与曲面重建模块。

科学背景：多波束声纳测深得到的是沿航迹分布的离散深度点
{(x_i, y_i, z_i)}。为生成连续的海底数字高程模型（DEM），
需对这些稀疏采样进行空间插值。

Shepard 方法（又称反距离权重法，IDW）是一种局部插值方法，
其权重函数为：

    w_i(x, y) = d_i^{-p} / Σ_j d_j^{-p}

其中 d_i = ||(x, y) - (x_i, y_i)|| 为欧氏距离，p 为幂次参数。
当 p → ∞ 时退化为 Voronoi 最近邻插值；p = 0 时为全局平均。
Shepard 方法具有 C^∞ 光滑性（除数据点外），适合地形重建。

本模块同时实现了基于 Shepard 权重的梯度估计与曲面曲率计算，
用于评估海底地形复杂度对反演精度的影响。
"""

import numpy as np
from signal_processor import shepard_interp_1d


def shepard_interp_2d(
    xd: np.ndarray,
    yd: np.ndarray,
    zd: np.ndarray,
    p: float,
    xi: np.ndarray,
    yi: np.ndarray,
    radius: float = None
) -> np.ndarray:
    """
    二维 Shepard 反距离权重插值。

    参数:
        xd, yd: 数据点坐标，形状 (nd,)
        zd:     数据点值，形状 (nd,)
        p:      距离幂次（通常 2.0~4.0）
        xi, yi: 插值点网格坐标，形状 (ni,)
        radius: 局部搜索半径（None 表示全局）
    返回:
        zi: 插值结果，形状 (len(yi), len(xi)) 的二维数组
    """
    xd = np.asarray(xd, dtype=np.float64)
    yd = np.asarray(yd, dtype=np.float64)
    zd = np.asarray(zd, dtype=np.float64)
    xi = np.asarray(xi, dtype=np.float64)
    yi = np.asarray(yi, dtype=np.float64)

    nx = len(xi)
    ny = len(yi)
    zi = np.zeros((ny, nx), dtype=np.float64)

    for iy in range(ny):
        for ix in range(nx):
            x = xi[ix]
            y = yi[iy]
            dx = x - xd
            dy = y - yd
            d = np.sqrt(dx ** 2 + dy ** 2)

            # 精确匹配数据点
            exact = np.where(d < 1e-12)[0]
            if len(exact) > 0:
                zi[iy, ix] = zd[exact[0]]
                continue

            # 距离权重
            if p == 0.0:
                w = np.ones_like(d) / len(d)
            else:
                if radius is not None:
                    mask = d <= radius
                    if not np.any(mask):
                        # 无局部点，回退到全局最近邻
                        idx = np.argmin(d)
                        zi[iy, ix] = zd[idx]
                        continue
                    d = d[mask]
                    z_local = zd[mask]
                else:
                    z_local = zd

                w = 1.0 / (d ** p)
                s = np.sum(w)
                if s > 0:
                    w = w / s
                else:
                    w = np.ones_like(d) / len(d)

            zi[iy, ix] = np.dot(w, z_local)

    return zi


class BathymetryInterpolator:
    """
    海底地形插值器。
    """

    def __init__(self, x: np.ndarray, y: np.ndarray, z: np.ndarray, p: float = 2.5):
        """
        参数:
            x, y, z: 测深点坐标与深度（z 向下为正）
            p: Shepard 幂次
        """
        self.x = np.asarray(x, dtype=np.float64).ravel()
        self.y = np.asarray(y, dtype=np.float64).ravel()
        self.z = np.asarray(z, dtype=np.float64).ravel()
        self.p = float(p)
        self.n_points = len(self.x)

        if self.n_points < 3:
            raise ValueError("至少需要 3 个数据点")

    def interpolate_grid(
        self,
        x_range: tuple,
        y_range: tuple,
        nx: int = 100,
        ny: int = 100,
        radius: float = None
    ) -> tuple:
        """
        在矩形区域内生成插值网格。

        参数:
            x_range: (x_min, x_max)
            y_range: (y_min, y_max)
            nx, ny: 网格点数
            radius: 局部搜索半径
        返回:
            (X, Y, Z) 二维网格数组
        """
        xi = np.linspace(x_range[0], x_range[1], nx)
        yi = np.linspace(y_range[0], y_range[1], ny)
        Z = shepard_interp_2d(self.x, self.y, self.z, self.p, xi, yi, radius)
        X, Y = np.meshgrid(xi, yi)
        return X, Y, Z

    def estimate_gradient(self, xq: float, yq: float, h: float = 1.0) -> np.ndarray:
        """
        利用中心差分估计地形梯度 ∇z = [∂z/∂x, ∂z/∂y]。

        公式:
            ∂z/∂x ≈ (z(x+h, y) - z(x-h, y)) / (2h)
            ∂z/∂y ≈ (z(x, y+h) - z(x, y-h)) / (2h)
        """
        # 在查询点附近进行 Shepard 插值
        z_px = shepard_interp_2d(self.x, self.y, self.z, self.p,
                                 np.array([xq + h]), np.array([yq]))[0, 0]
        z_mx = shepard_interp_2d(self.x, self.y, self.z, self.p,
                                 np.array([xq - h]), np.array([yq]))[0, 0]
        z_py = shepard_interp_2d(self.x, self.y, self.z, self.p,
                                 np.array([xq]), np.array([yq + h]))[0, 0]
        z_my = shepard_interp_2d(self.x, self.y, self.z, self.p,
                                 np.array([xq]), np.array([yq - h]))[0, 0]

        dz_dx = (z_px - z_mx) / (2.0 * h)
        dz_dy = (z_py - z_my) / (2.0 * h)
        return np.array([dz_dx, dz_dy])

    def estimate_curvature(self, xq: float, yq: float, h: float = 5.0) -> float:
        """
        估计地形平均曲率（拉普拉斯算子近似）。

        公式:
            κ ≈ ∇²z = ∂²z/∂x² + ∂²z/∂y²
                  ≈ (z(x+h,y) + z(x-h,y) + z(x,y+h) + z(x,y-h) - 4z(x,y)) / h²
        """
        z0 = shepard_interp_2d(self.x, self.y, self.z, self.p,
                               np.array([xq]), np.array([yq]))[0, 0]
        z_px = shepard_interp_2d(self.x, self.y, self.z, self.p,
                                 np.array([xq + h]), np.array([yq]))[0, 0]
        z_mx = shepard_interp_2d(self.x, self.y, self.z, self.p,
                                 np.array([xq - h]), np.array([yq]))[0, 0]
        z_py = shepard_interp_2d(self.x, self.y, self.z, self.p,
                                 np.array([xq]), np.array([yq + h]))[0, 0]
        z_my = shepard_interp_2d(self.x, self.y, self.z, self.p,
                                 np.array([xq]), np.array([yq - h]))[0, 0]

        curvature = (z_px + z_mx + z_py + z_my - 4.0 * z0) / (h ** 2)
        return float(curvature)

    def cross_section_profile(self, x_start: float, y_start: float,
                              x_end: float, y_end: float, n_samples: int = 200) -> tuple:
        """
        提取两点之间的地形剖面线。

        参数:
            x_start, y_start: 起点
            x_end, y_end:     终点
            n_samples:        采样点数
        返回:
            (s, z) — s 为沿剖面距离，z 为深度
        """
        t = np.linspace(0.0, 1.0, n_samples)
        x_line = x_start + t * (x_end - x_start)
        y_line = y_start + t * (y_end - y_start)
        z_line = shepard_interp_1d(self.x, self.z, self.p, x_line)
        # 沿剖面的弧长参数
        dx = x_line[1:] - x_line[:-1]
        dy = y_line[1:] - y_line[:-1]
        ds = np.sqrt(dx ** 2 + dy ** 2)
        s = np.concatenate([[0.0], np.cumsum(ds)])
        return s, z_line
