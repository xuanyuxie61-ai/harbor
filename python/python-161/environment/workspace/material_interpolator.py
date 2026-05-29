"""
material_interpolator.py
基于种子项目 927_pwl_interp_2d (piecewise linear interpolation on 2D grid)
改造为钙钛矿太阳能电池材料参数随温度和化学组分的二维分段线性插值器。

钙钛矿材料 MAPbI3 的关键参数（带隙 Eg、吸收系数 α、载流子迁移率 μ）
强烈依赖于：
  1. 温度 T [K]（热膨胀导致晶格常数变化）
  2. 卤素混合比 x（MA(PbI_{3-x}Br_x)_3）

核心公式：
  1. 带隙温度依赖：Eg(T, x) = Eg(0, x) - S * T^2 / (T + Θ_D)
     其中 Θ_D 为 Debye 温度，S 为 Varshni 参数。
  2. 迁移率温度依赖：μ(T) ∝ T^{-3/2}（声学声子散射主导）
  3. 吸收系数边缘：α(E, T) ∝ sqrt(E - Eg(T))（Tauc 定律，直接带隙）
  4. 二维分段线性插值（三角剖分）：
     给定矩形网格上的数据 Z_{ij} = f(X_i, Y_j)，
     对任意查询点 (xq, yq) 找到所在矩形，再判断所在三角形，
     使用重心坐标进行线性插值：
       f(q) = α f(P_a) + β f(P_b) + γ f(P_c)
     其中 α+β+γ = 1。
"""

import numpy as np
from typing import Tuple


def r8vec_bracket5(n: int, x: np.ndarray, xval: float) -> int:
    """
    在已排序数组 x 中找到包含 xval 的区间索引 i，使得 x[i] <= xval <= x[i+1]。
    若 xval 越界，返回 -1。
    对应原项目中的 r8vec_bracket5。
    """
    if n < 2 or xval < x[0] or xval > x[-1]:
        return -1
    # 二分查找
    lo, hi = 0, n - 1
    while hi > lo + 1:
        mid = (lo + hi) // 2
        if xval < x[mid]:
            hi = mid
        else:
            lo = mid
    return lo


def pwl_interp_2d_scalar(
    nxd: int, nyd: int,
    xd: np.ndarray, yd: np.ndarray, zd: np.ndarray,
    xi: float, yi: float
) -> float:
    """
    对单个查询点进行二维分段线性插值。
    对应原项目 pwl_interp_2d 的核心算法。
    """
    i = r8vec_bracket5(nxd, xd, xi)
    if i == -1:
        return np.inf

    j = r8vec_bracket5(nyd, yd, yi)
    if j == -1:
        return np.inf

    # 判断查询点位于矩形的哪个三角形中
    # 对角线：从 (i, j) 到 (i+1, j+1)
    # 三角形1：(i,j), (i+1,j), (i,j+1)
    # 三角形2：(i+1,j+1), (i,j+1), (i+1,j)
    y_diag = yd[j + 1] + (yd[j] - yd[j + 1]) * (xi - xd[i]) / (xd[i + 1] - xd[i])

    if yi < y_diag:
        # 三角形1
        dxa = xd[i + 1] - xd[i]
        dya = yd[j] - yd[j]
        dxb = xd[i] - xd[i]
        dyb = yd[j + 1] - yd[j]
        dxi = xi - xd[i]
        dyi = yi - yd[j]
        det = dxa * dyb - dya * dxb
        if abs(det) < 1e-14:
            return np.inf
        alpha = (dxi * dyb - dyi * dxb) / det
        beta = (dxa * dyi - dya * dxi) / det
        gamma = 1.0 - alpha - beta
        return alpha * zd[i + 1, j] + beta * zd[i, j + 1] + gamma * zd[i, j]
    else:
        # 三角形2
        dxa = xd[i] - xd[i + 1]
        dya = yd[j + 1] - yd[j + 1]
        dxb = xd[i + 1] - xd[i + 1]
        dyb = yd[j] - yd[j + 1]
        dxi = xi - xd[i + 1]
        dyi = yi - yd[j + 1]
        det = dxa * dyb - dya * dxb
        if abs(det) < 1e-14:
            return np.inf
        alpha = (dxi * dyb - dyi * dxb) / det
        beta = (dxa * dyi - dya * dxi) / det
        gamma = 1.0 - alpha - beta
        return alpha * zd[i, j + 1] + beta * zd[i + 1, j + 1] + gamma * zd[i + 1, j]


def pwl_interp_2d_vector(
    xd: np.ndarray, yd: np.ndarray, zd: np.ndarray,
    xi: np.ndarray, yi: np.ndarray
) -> np.ndarray:
    """
    向量化二维分段线性插值。
    """
    nxd, nyd = len(xd), len(yd)
    ni = len(xi)
    zi = np.full(ni, np.inf)
    for k in range(ni):
        zi[k] = pwl_interp_2d_scalar(nxd, nyd, xd, yd, zd, xi[k], yi[k])
    return zi


class PerovskiteMaterial:
    """
    钙钛矿材料参数插值器。
    在 (温度 T, 卤素配比 x) 的二维网格上存储材料参数，
    通过分段线性插值获取任意 (T, x) 处的参数。
    """

    def __init__(self):
        # 构建网格数据
        self.T_grid = np.linspace(200.0, 400.0, 21)   # K
        self.x_grid = np.linspace(0.0, 1.0, 11)       # Br 比例

        # 生成模拟数据：带隙 Eg [eV]
        self.Eg_grid = self._build_eg_table()
        # 电子迁移率 μ_n [cm^2/(V·s)]
        self.mu_n_grid = self._build_mu_n_table()
        # 空穴迁移率 μ_p [cm^2/(V·s)]
        self.mu_p_grid = self._build_mu_p_table()
        # 吸收系数 @ 600 nm [cm^{-1}]
        self.alpha_grid = self._build_alpha_table()

    def _build_eg_table(self) -> np.ndarray:
        """
        带隙表：Eg(T, x) = Eg0(x) - S * T^2 / (T + Θ_D)
        MAPbI3: Eg0 ≈ 1.57 eV; MAPbBr3: Eg0 ≈ 2.29 eV
        Varshni 参数 S ≈ 8e-4 eV/K, Θ_D ≈ 150 K
        """
        Tg, Xg = np.meshgrid(self.T_grid, self.x_grid, indexing='ij')
        Eg0 = 1.57 + 0.72 * Xg  # 线性插值 Eg0(x)
        S = 8.0e-4
        theta_D = 150.0
        Eg = Eg0 - S * Tg ** 2 / (Tg + theta_D)
        return Eg

    def _build_mu_n_table(self) -> np.ndarray:
        """
        电子迁移率：μ_n(T, x) = μ_n0 * (T/300)^{-1.5} * (1 + 0.2*x)^{-1}
        """
        Tg, Xg = np.meshgrid(self.T_grid, self.x_grid, indexing='ij')
        mu_n0 = 20.0  # cm^2/(V·s) @ 300K, x=0
        mu_n = mu_n0 * (Tg / 300.0) ** (-1.5) * (1.0 + 0.2 * Xg) ** (-1)
        return np.clip(mu_n, 0.1, 500.0)

    def _build_mu_p_table(self) -> np.ndarray:
        """
        空穴迁移率：μ_p(T, x) = μ_p0 * (T/300)^{-1.5} * (1 + 0.3*x)^{-1}
        """
        Tg, Xg = np.meshgrid(self.T_grid, self.x_grid, indexing='ij')
        mu_p0 = 10.0
        mu_p = mu_p0 * (Tg / 300.0) ** (-1.5) * (1.0 + 0.3 * Xg) ** (-1)
        return np.clip(mu_p, 0.1, 500.0)

    def _build_alpha_table(self) -> np.ndarray:
        """
        吸收系数：α ∝ sqrt(E_photon - Eg(T, x))
        当 E_photon < Eg 时，α → 0（ Urbach 尾近似）
        """
        Tg, Xg = np.meshgrid(self.T_grid, self.x_grid, indexing='ij')
        Eg0 = 1.57 + 0.72 * Xg
        S = 8.0e-4
        theta_D = 150.0
        Eg = Eg0 - S * Tg ** 2 / (Tg + theta_D)
        E_photon = 2.07  # eV (≈ 600 nm)
        alpha0 = 5.0e4  # cm^{-1}
        alpha = alpha0 * np.sqrt(np.maximum(E_photon - Eg, 0.0) + 0.01)
        return np.clip(alpha, 1e3, 1e6)

    def get_params(self, T: float, x: float) -> dict:
        """
        获取指定 (T, x) 处的材料参数。
        若 T 或 x 超出网格范围，则使用最近边界值（外推保护）。
        """
        # 边界保护
        T_clipped = np.clip(T, self.T_grid.min(), self.T_grid.max())
        x_clipped = np.clip(x, self.x_grid.min(), self.x_grid.max())

        Eg = pwl_interp_2d_scalar(len(self.T_grid), len(self.x_grid),
                                   self.T_grid, self.x_grid, self.Eg_grid,
                                   T_clipped, x_clipped)
        mu_n = pwl_interp_2d_scalar(len(self.T_grid), len(self.x_grid),
                                     self.T_grid, self.x_grid, self.mu_n_grid,
                                     T_clipped, x_clipped)
        mu_p = pwl_interp_2d_scalar(len(self.T_grid), len(self.x_grid),
                                     self.T_grid, self.x_grid, self.mu_p_grid,
                                     T_clipped, x_clipped)
        alpha = pwl_interp_2d_scalar(len(self.T_grid), len(self.x_grid),
                                      self.T_grid, self.x_grid, self.alpha_grid,
                                      T_clipped, x_clipped)

        # 处理插值失败
        if not np.isfinite(Eg):
            Eg = float(self.Eg_grid[len(self.T_grid)//2, len(self.x_grid)//2])
        if not np.isfinite(mu_n):
            mu_n = 10.0
        if not np.isfinite(mu_p):
            mu_p = 5.0
        if not np.isfinite(alpha):
            alpha = 5.0e4

        return {
            "bandgap_eV": float(Eg),
            "electron_mobility": float(mu_n),
            "hole_mobility": float(mu_p),
            "absorption_coeff_600nm": float(alpha),
        }


if __name__ == "__main__":
    mat = PerovskiteMaterial()
    params = mat.get_params(300.0, 0.3)
    print("Perovskite @ T=300K, x=0.3:", params)
