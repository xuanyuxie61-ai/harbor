
import numpy as np


class YeeGrid3D:

    def __init__(self, nx, ny, nz, Lx, Ly, Lz, grid_type='rectangular'):
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


        self.x = np.linspace(0.0, Lx, nx)
        self.y = np.linspace(0.0, Ly, ny)
        self.z = np.linspace(0.0, Lz, nz)


        self.xh = np.zeros(nx)
        self.yh = np.zeros(ny)
        self.zh = np.zeros(nz)
        self.xh[:-1] = 0.5 * (self.x[:-1] + self.x[1:])
        self.yh[:-1] = 0.5 * (self.y[:-1] + self.y[1:])
        self.zh[:-1] = 0.5 * (self.z[:-1] + self.z[1:])

        if nx > 1:
            self.xh[-1] = self.x[-1] + 0.5 * self.dx
        if ny > 1:
            self.yh[-1] = self.y[-1] + 0.5 * self.dy
        if nz > 1:
            self.zh[-1] = self.z[-1] + 0.5 * self.dz


        self.X, self.Y, self.Z = np.meshgrid(self.x, self.y, self.z, indexing='ij')
        self.Xh, self.Yh, self.Zh = np.meshgrid(self.xh, self.yh, self.zh, indexing='ij')

    def get_staggered_electric_shape(self):
        return (self.nx, self.ny, self.nz)

    def get_staggered_magnetic_shape(self):
        return (self.nx, self.ny, self.nz)

    def cell_volume(self):
        return self.dx * self.dy * self.dz

    def total_nodes(self):
        return self.nx * self.ny * self.nz


class CylindricalYeeGrid:

    def __init__(self, nr, nz, R, Z):
        if nr < 3 or nz < 3:
            raise ValueError("网格数必须至少为3")
        if R <= 0 or Z <= 0:
            raise ValueError("尺寸必须为正")

        self.nr = nr
        self.nz = nz
        self.R = R
        self.Z = Z


        self.r = np.linspace(0.0, R, nr)
        self.z = np.linspace(0.0, Z, nz)

        self.dr = R / (nr - 1)
        self.dz = Z / (nz - 1)


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
        weight = np.zeros((self.nr, self.nz))
        for i in range(self.nr):
            r_val = self.r[i]
            if r_val < self.dr * 0.5:

                weight[i, :] = np.pi * (0.5 * self.dr) ** 2 * self.dz
            else:
                weight[i, :] = 2.0 * np.pi * r_val * self.dr * self.dz
        return weight

    def axis_boundary_condition(self, field):
        if field.shape[0] != self.nr:
            raise ValueError("场数组的第一维必须与nr一致")

        field[0, :] = 0.0
        return field


def generate_rectangular_grid(xmin, xmax, nx, ymin, ymax, ny, zmin, zmax, nz):
    Lx = xmax - xmin
    Ly = ymax - ymin
    Lz = zmax - zmin
    grid = YeeGrid3D(nx, ny, nz, Lx, Ly, Lz, grid_type='rectangular')

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
    grid = CylindricalYeeGrid(nr, nz, rmax, zmax - zmin)
    grid.z += zmin
    grid.zh += zmin
    grid.Z_grid, _ = np.meshgrid(grid.z, grid.r, indexing='ij')
    _, grid.R_grid = np.meshgrid(grid.z, grid.r, indexing='ij')
    return grid
