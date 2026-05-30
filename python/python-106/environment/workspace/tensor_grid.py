
import numpy as np


def uniform_tensor_grid_1d(a, b, n):
    if n < 2:
        raise ValueError("n must be at least 2.")
    if b <= a:
        raise ValueError("b must exceed a.")
    x = np.linspace(a, b, n)
    dx = (b - a) / (n - 1)
    return x, dx


def uniform_tensor_grid_3d(x_bounds, y_bounds, z_bounds, nx, ny, nz):
    x, dx = uniform_tensor_grid_1d(x_bounds[0], x_bounds[1], nx)
    y, dy = uniform_tensor_grid_1d(y_bounds[0], y_bounds[1], ny)
    z, dz = uniform_tensor_grid_1d(z_bounds[0], z_bounds[1], nz)

    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    return X, Y, Z, dx, dy, dz


def gauss_legendre_tensor_grid_3d(x_bounds, y_bounds, z_bounds, nx, ny, nz):
    xi, wx = np.polynomial.legendre.leggauss(nx)
    yi, wy = np.polynomial.legendre.leggauss(ny)
    zi, wz = np.polynomial.legendre.leggauss(nz)

    ax, bx = x_bounds
    ay, by = y_bounds
    az, bz = z_bounds

    x = 0.5 * (bx - ax) * xi + 0.5 * (bx + ax)
    y = 0.5 * (by - ay) * yi + 0.5 * (by + ay)
    z = 0.5 * (bz - az) * zi + 0.5 * (bz + az)

    wx = 0.5 * (bx - ax) * wx
    wy = 0.5 * (by - ay) * wy
    wz = 0.5 * (bz - az) * wz

    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    return X, Y, Z, wx, wy, wz


def cfl_time_step(dx, dy, dz, c=2.99792458e8, courant_factor=0.99):
    if dx <= 0 or dy <= 0 or dz <= 0:
        raise ValueError("Grid spacings must be positive.")
    dt = courant_factor / (c * np.sqrt((1.0 / dx) ** 2 + (1.0 / dy) ** 2 + (1.0 / dz) ** 2))
    return dt


def grid_points_in_sphere(X, Y, Z, center, radius):
    dist_sq = (X - center[0]) ** 2 + (Y - center[1]) ** 2 + (Z - center[2]) ** 2
    return dist_sq <= radius ** 2
