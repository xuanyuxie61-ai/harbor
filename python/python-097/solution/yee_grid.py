"""
yee_grid.py

Yee交错网格生成与管理模块。
基于grid_rectangular与grid_polar的核心思想，为FDTD仿真提供结构化网格。

Yee网格的核心特征:
------------------
在三维Yee元胞中，电场和磁场分量交错排列:
- Ex 位于 (i+½, j, k)
- Ey 位于 (i, j+½, k)
- Ez 位于 (i, j, k+½)
- Hx 位于 (i, j+½, k+½)
- Hy 位于 (i+½, j, k+½)
- Hz 位于 (i+½, j+½, k)

这种交错排列使得旋度算子的离散化自然满足中心差分的二阶精度。
"""

import numpy as np


class YeeGrid3D:
    """
    三维Yee交错网格类。

    提供矩形均匀网格和圆柱坐标网格的生成能力。
    """

    def __init__(self, nx, ny, nz, Lx, Ly, Lz, grid_type='rectangular'):
        """
        Parameters
        ----------
        nx, ny, nz : int
            各方向的网格数 (必须 ≥ 3)
        Lx, Ly, Lz : float
            计算域尺寸
        grid_type : str
            'rectangular' 或 'polar_cylindrical'
        """
        if nx < 3 or ny < 3 or nz < 3:
            raise ValueError("网格数必须至少为3")
        if Lx <= 0 or Ly <= 0 or Lz <= 0:
            raise ValueError("计算域尺寸必须为正")

        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.Lx = Lx
        self.Ly = Ly
        self.Lz = Lz
        self.grid_type = grid_type

        self.dx = Lx / (nx - 1)
        self.dy = Ly / (ny - 1)
        self.dz = Lz / (nz - 1)

        # 主网格坐标 (电场节点)
        self.x = np.linspace(0.0, Lx, nx)
        self.y = np.linspace(0.0, Ly, ny)
        self.z = np.linspace(0.0, Lz, nz)

        # 交错网格坐标 (磁场节点)
        self.xh = np.zeros(nx)
        self.yh = np.zeros(ny)
        self.zh = np.zeros(nz)
        self.xh[:-1] = 0.5 * (self.x[:-1] + self.x[1:])
        self.yh[:-1] = 0.5 * (self.y[:-1] + self.y[1:])
        self.zh[:-1] = 0.5 * (self.z[:-1] + self.z[1:])
        # 边界外推
        if nx > 1:
            self.xh[-1] = self.x[-1] + 0.5 * self.dx
        if ny > 1:
            self.yh[-1] = self.y[-1] + 0.5 * self.dy
        if nz > 1:
            self.zh[-1] = self.z[-1] + 0.5 * self.dz

        # 生成三维网格坐标
        self.X, self.Y, self.Z = np.meshgrid(self.x, self.y, self.z, indexing='ij')
        self.Xh, self.Yh, self.Zh = np.meshgrid(self.xh, self.yh, self.zh, indexing='ij')

    def get_staggered_electric_shape(self):
        """返回电场各分量的数组形状。"""
        return (self.nx, self.ny, self.nz)

    def get_staggered_magnetic_shape(self):
        """返回磁场各分量的数组形状（与电场相同，值在交错位置计算）。"""
        return (self.nx, self.ny, self.nz)

    def cell_volume(self):
        """单个网格元胞的体积。"""
        return self.dx * self.dy * self.dz

    def total_nodes(self):
        """总节点数。"""
        return self.nx * self.ny * self.nz


class CylindricalYeeGrid:
    """
    圆柱坐标系下的Yee网格。
    用于分析圆柱形微波谐振腔（如TM₀₁₀、TE₁₁₁模式）。

    坐标: (r, φ, z)
    由于轴对称性，通常取二维截面 (r, z)。
    """

    def __init__(self, nr, nz, R, Z):
        """
        Parameters
        ----------
        nr, nz : int
            r和z方向的网格数
        R, Z : float
            最大半径和高度
        """
        if nr < 3 or nz < 3:
            raise ValueError("网格数必须至少为3")
        if R <= 0 or Z <= 0:
            raise ValueError("尺寸必须为正")

        self.nr = nr
        self.nz = nz
        self.R = R
        self.Z = Z

        # r方向从0到R，注意r=0处为奇点，需特殊处理
        self.r = np.linspace(0.0, R, nr)
        self.z = np.linspace(0.0, Z, nz)

        self.dr = R / (nr - 1)
        self.dz = Z / (nz - 1)

        # 交错坐标
        self.rh = np.zeros(nr)
        self.zh = np.zeros(nz)
        self.rh[:-1] = 0.5 * (self.r[:-1] + self.r[1:])
        self.zh[:-1] = 0.5 * (self.z[:-1] + self.z[1:])
        if nr > 1:
            self.rh[-1] = self.r[-1] + 0.5 * self.dr
        if nz > 1:
            self.zh[-1] = self.z[-1] + 0.5 * self.dz

        self.R_grid, self.Z_grid = np.meshgrid(self.r, self.z, indexing='ij')

    def radial_weighting(self):
        """
        圆柱坐标下的体积权重因子 (2πr dr dz)。
        在r=0处需要特殊处理以避免奇点。
        """
        weight = np.zeros((self.nr, self.nz))
        for i in range(self.nr):
            r_val = self.r[i]
            if r_val < self.dr * 0.5:
                # r=0附近使用小圆盘面积近似
                weight[i, :] = np.pi * (0.5 * self.dr) ** 2 * self.dz
            else:
                weight[i, :] = 2.0 * np.pi * r_val * self.dr * self.dz
        return weight

    def axis_boundary_condition(self, field):
        """
        在r=0处应用轴对称边界条件。
        对于轴对称场，∂/∂φ = 0，且在r=0处:
        - E_r = 0, E_φ = 0 (切向电场为零)
        - H_z 有限
        """
        if field.shape[0] != self.nr:
            raise ValueError("场数组的第一维必须与nr一致")
        # 在r=0处，将径向场设为零
        field[0, :] = 0.0
        return field


def generate_rectangular_grid(xmin, xmax, nx, ymin, ymax, ny, zmin, zmax, nz):
    """
    生成矩形均匀三维网格（基于grid_rectangular思想）。

    Returns
    -------
    YeeGrid3D
    """
    Lx = xmax - xmin
    Ly = ymax - ymin
    Lz = zmax - zmin
    grid = YeeGrid3D(nx, ny, nz, Lx, Ly, Lz, grid_type='rectangular')
    # 平移坐标
    grid.x += xmin
    grid.y += ymin
    grid.z += zmin
    grid.xh += xmin
    grid.yh += ymin
    grid.zh += zmin
    grid.X, grid.Y, grid.Z = np.meshgrid(grid.x, grid.y, grid.z, indexing='ij')
    grid.Xh, grid.Yh, grid.Zh = np.meshgrid(grid.xh, grid.yh, grid.zh, indexing='ij')
    return grid


def generate_polar_grid_2d(rmin, rmax, nr, zmin, zmax, nz):
    """
    生成二维极坐标(r,z)网格（基于grid_polar思想）。

    Returns
    -------
    CylindricalYeeGrid
    """
    grid = CylindricalYeeGrid(nr, nz, rmax, zmax - zmin)
    grid.z += zmin
    grid.zh += zmin
    grid.Z_grid, _ = np.meshgrid(grid.z, grid.r, indexing='ij')
    _, grid.R_grid = np.meshgrid(grid.z, grid.r, indexing='ij')
    return grid
