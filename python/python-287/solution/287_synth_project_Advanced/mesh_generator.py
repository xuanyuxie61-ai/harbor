# -*- coding: utf-8 -*-
"""
mesh_generator.py
=================

二维/三维结构化网格生成模块.

来源种子项目:
  - 789_navier_stokes_mesh2d   -> 结构化网格节点/单元提取
  - 415_fem2d_scalar_display_brief -> 二维标量场的网格表示
  - 1190_VitjanZ_3DSR          -> 多层网格与上采样思想 (用于 3D 场重构)

物理背景:
  我们模拟的是一个 Harris 电流片中的撕裂模不稳定性.
  计算域:  (x, y) in [0, Lx] x [-Ly, Ly]
  平衡磁场: B0x(y) = B0 * tanh(y/L_cs)  (沿 x 方向)
  电流片中心在 y = 0.

  网格在电流片附近需要加密以解析撕裂模本征函数的精细结构.
  我们使用拉伸变换 (tanh-stretching) 实现:
      y_j = Ly * tanh(alpha * (2j/Ny - 1)) / tanh(alpha)
  其中 alpha 控制拉伸强度.

  撕裂模扰动具有周期性:
      f(x, y, t) = f_hat(y, t) * exp(i k x) + c.c.
  因此 x 方向使用均匀网格, y 方向使用拉伸网格.
"""

from __future__ import annotations
import numpy as np

from plasma_constants import L_CS


# -------------------------------------------------------------------
#  一维拉伸网格 (tanh-stretching, 中心加密)
# -------------------------------------------------------------------
def tanh_stretched_1d(a: float, n: int, alpha: float) -> np.ndarray:
    """
    在区间 [-a, a] 上生成 n 个节点的拉伸网格.

    变换: y(xi) = a * tanh(alpha * xi) / tanh(alpha),  xi in [-1, 1].
    当 alpha -> 0 时退化为均匀网格.
    当 alpha 增大时, 网格在 y = 0 附近加密.

    参数:
        a:     半宽度
        n:     节点数
        alpha: 拉伸参数 (>= 0)

    返回:
        长度为 n 的数组, 严格递增, y[0] = -a, y[-1] = a.
    """
    if a <= 0.0:
        raise ValueError(f"半宽度 a 必须为正, 当前 a = {a}")
    if n < 2:
        raise ValueError(f"节点数 n 必须 >= 2, 当前 n = {n}")
    if alpha < 0.0:
        raise ValueError(f"拉伸参数 alpha 必须 >= 0, 当前 alpha = {alpha}")

    xi = np.linspace(-1.0, 1.0, n)
    if alpha < 1.0e-12:
        return a * xi
    return a * np.tanh(alpha * xi) / np.tanh(alpha)


# -------------------------------------------------------------------
#  二维结构化网格
# -------------------------------------------------------------------
class StructuredMesh2D:
    """
    二维结构化网格, 用于 Harris 电流片撕裂模模拟.

    属性:
        nx, ny       : x, y 方向的节点数
        lx, ly       : 计算域 [0, lx] x [-ly, ly]
        x, y         : 一维坐标数组 (长度分别为 nx, ny)
        dx, dy       : 一维间距数组 (长度分别为 nx-1, ny-1)
        X, Y         : 二维网格坐标 (ny, nx)
        jacobian_y   : y 方向拉伸的 Jacobi 矩阵 d(y_phys)/d(xi)  (长度 ny)
        area_element : 每个单元的面积 (ny-1, nx-1)
    """

    def __init__(
        self,
        nx: int,
        ny: int,
        lx: float,
        ly: float,
        alpha_stretch: float = 2.0,
    ):
        if nx < 3 or ny < 3:
            raise ValueError(f"网格尺寸必须 >= 3, 当前 nx={nx}, ny={ny}")
        if lx <= 0.0 or ly <= 0.0:
            raise ValueError(f"域尺寸必须为正, 当前 lx={lx}, ly={ly}")

        self.nx = int(nx)
        self.ny = int(ny)
        self.lx = float(lx)
        self.ly = float(ly)
        self.alpha_stretch = float(alpha_stretch)

        # x 方向: 均匀网格, 周期性
        self.x = np.linspace(0.0, self.lx, self.nx, endpoint=False)
        self.dx = np.full(self.nx - 1, self.lx / self.nx)

        # y 方向: 拉伸网格, 中心加密
        self.y = tanh_stretched_1d(self.ly, self.ny, self.alpha_stretch)
        self.dy = np.diff(self.y)

        # 二维网格
        self.X, self.Y = np.meshgrid(self.x, self.y)

        # y 方向 Jacobi (dy_phys / d_xi, 其中 xi 均匀分布于 [-1,1])
        a = self.alpha_stretch
        if a < 1.0e-12:
            self.jacobian_y = np.full(self.ny, 2.0 * self.ly / (self.ny - 1))
        else:
            xi = np.linspace(-1.0, 1.0, self.ny)
            self.jacobian_y = (
                self.ly * a * (1.0 - np.tanh(a * xi) ** 2) / np.tanh(a)
            )

        # 单元面积
        dx_row = self.dx[np.newaxis, :]          # (1, nx-1)
        dy_col = self.dy[:, np.newaxis]           # (ny-1, 1)
        self.area_element = dy_col * dx_row       # (ny-1, nx-1)

    # -----------------------------------------------------------
    #  网格质量指标
    # -----------------------------------------------------------
    def min_cell_size(self) -> float:
        """最小单元尺寸 (用于 CFL 条件)."""
        return min(self.dx.min(), self.dy.min())

    def max_aspect_ratio(self) -> float:
        """最大长宽比."""
        dx2d = self.dx[np.newaxis, :] * np.ones((self.ny - 1, 1))
        dy2d = self.dy[:, np.newaxis] * np.ones((1, self.nx - 1))
        ratio = np.maximum(dx2d / dy2d, dy2d / dx2d)
        return float(ratio.max())

    def total_area(self) -> float:
        """计算域总面积 (用于归一化检验)."""
        return float(self.area_element.sum())

    # -----------------------------------------------------------
    #  Harris 平衡场 (在网格上采样)
    # -----------------------------------------------------------
    def harris_bx(self, b0: float = None) -> np.ndarray:
        """
        Harris 平衡磁场 B_x(y) = B0 * tanh(y / L_cs).
        返回形状 (ny, nx) 的二维数组.
        """
        if b0 is None:
            from plasma_constants import B0 as b0
        by_over_l = self.Y / L_CS
        # 数值安全: tanh 对大参数饱和
        return b0 * np.tanh(np.clip(by_over_l, -20.0, 20.0))

    def harris_jz(self, b0: float = None, mu0: float = None) -> np.ndarray:
        """
        Harris 平衡电流密度 j_z = -(B0/(mu_0 L_cs)) * sech^2(y/L_cs).
        返回形状 (ny, nx) 的二维数组.
        """
        if b0 is None:
            from plasma_constants import B0 as b0
        if mu0 is None:
            from plasma_constants import MU_0 as mu0
        arg = np.clip(self.Y / L_CS, -20.0, 20.0)
        sech2 = 1.0 / np.cosh(arg) ** 2
        return -(b0 / (mu0 * L_CS)) * sech2

    # -----------------------------------------------------------
    #  摘要
    # -----------------------------------------------------------
    def summary(self) -> str:
        lines = [
            "=" * 60,
            "  二维结构化网格 (Harris 电流片)",
            "=" * 60,
            f"  nx, ny        = {self.nx}, {self.ny}",
            f"  Lx            = {self.lx:.4e}",
            f"  Ly            = {self.ly:.4e}",
            f"  alpha_stretch = {self.alpha_stretch:.4e}",
            f"  min dx        = {self.dx.min():.4e}",
            f"  min dy        = {self.dy.min():.4e}",
            f"  max dy/min dy = {self.dy.max()/self.dy.min():.2f}",
            f"  total area    = {self.total_area():.4e}",
            f"  (expected)    = {2.0*self.ly*self.lx:.4e}",
            f"  max aspect    = {self.max_aspect_ratio():.2f}",
            "=" * 60,
        ]
        return "\n".join(lines)


# -------------------------------------------------------------------
#  三维网格扩展 (用于 3D 场重构)
# -------------------------------------------------------------------
def extend_to_3d(
    mesh2d: StructuredMesh2D,
    nz: int,
    lz: float,
) -> tuple:
    """
    将二维网格沿 z 方向 (环向) 扩展为三维网格.

    参数:
        mesh2d: 二维网格对象
        nz:     z 方向节点数
        lz:     z 方向长度

    返回:
        (X3, Y3, Z3): 三维网格坐标, 形状均为 (nz, ny, nx)
    """
    z = np.linspace(0.0, lz, nz, endpoint=False)
    X3 = np.broadcast_to(mesh2d.X, (nz, mesh2d.ny, mesh2d.nx)).copy()
    Y3 = np.broadcast_to(mesh2d.Y, (nz, mesh2d.ny, mesh2d.nx)).copy()
    Z3 = np.zeros_like(X3)
    for k in range(nz):
        Z3[k, :, :] = z[k]
    return X3, Y3, Z3
