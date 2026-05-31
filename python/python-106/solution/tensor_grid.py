"""
tensor_grid.py
==============
Tensor-product grid generation for 3D plasmonic simulation domains.

A tensor-product grid in M dimensions is defined by the Cartesian product
of 1D node sets:

    X = x_{1D}^{(1)} ⊗ x_{1D}^{(2)} ⊗ ... ⊗ x_{1D}^{(M)}

For Gauss-Legendre quadrature, the 1D nodes are the roots of Legendre
polynomials P_n(ξ) = 0 on [−1, 1].  Mapped to an interval [a, b]:

    x_i = (b−a)/2 · ξ_i + (a+b)/2

In 3D, the total number of grid points is N = n_x · n_y · n_z.

For plasmonic FDTD simulations, a uniform Cartesian grid is typically used:

    x_i = a + i Δx,   Δx = (b−a) / (n_x − 1)

with the Courant-Friedrichs-Lewy (CFL) stability condition:

    Δt ≤ 1 / (c √((1/Δx)² + (1/Δy)² + (1/Δz)²))

This module generates both uniform and Gauss-Legendre tensor grids,
and computes CFL-limited time steps.
"""

import numpy as np


def uniform_tensor_grid_1d(a, b, n):
    """
    Generate a uniform 1D grid.

    Parameters
    ----------
    a, b : float
    n : int

    Returns
    -------
    x : ndarray, shape (n,)
    dx : float
    """
    if n < 2:
        raise ValueError("n must be at least 2.")
    if b <= a:
        raise ValueError("b must exceed a.")
    x = np.linspace(a, b, n)
    dx = (b - a) / (n - 1)
    return x, dx


def uniform_tensor_grid_3d(x_bounds, y_bounds, z_bounds, nx, ny, nz):
    """
    Generate a 3D uniform Cartesian tensor grid.

    Parameters
    ----------
    x_bounds, y_bounds, z_bounds : tuple
        (min, max) for each direction.
    nx, ny, nz : int

    Returns
    -------
    X, Y, Z : ndarray, shape (nx, ny, nz)
    dx, dy, dz : float
    """
    x, dx = uniform_tensor_grid_1d(x_bounds[0], x_bounds[1], nx)
    y, dy = uniform_tensor_grid_1d(y_bounds[0], y_bounds[1], ny)
    z, dz = uniform_tensor_grid_1d(z_bounds[0], z_bounds[1], nz)

    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    return X, Y, Z, dx, dy, dz


def gauss_legendre_tensor_grid_3d(x_bounds, y_bounds, z_bounds, nx, ny, nz):
    """
    Generate a 3D tensor grid using Gauss-Legendre nodes in each direction.

    Parameters
    ----------
    x_bounds, y_bounds, z_bounds : tuple
    nx, ny, nz : int

    Returns
    -------
    X, Y, Z : ndarray, shape (nx, ny, nz)
    wx, wy, wz : ndarray
        1D quadrature weights.
    """
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
    """
    Compute the maximum stable time step for 3D FDTD via the CFL condition.

    Parameters
    ----------
    dx, dy, dz : float
    c : float
        Speed of light in the medium (m/s).
    courant_factor : float
        Safety factor < 1.

    Returns
    -------
    dt : float
    """
    if dx <= 0 or dy <= 0 or dz <= 0:
        raise ValueError("Grid spacings must be positive.")
    dt = courant_factor / (c * np.sqrt((1.0 / dx) ** 2 + (1.0 / dy) ** 2 + (1.0 / dz) ** 2))
    return dt


def grid_points_in_sphere(X, Y, Z, center, radius):
    """
    Return a boolean mask for grid points inside a sphere.

    Parameters
    ----------
    X, Y, Z : ndarray
    center : ndarray, shape (3,)
    radius : float

    Returns
    -------
    mask : ndarray
    """
    dist_sq = (X - center[0]) ** 2 + (Y - center[1]) ** 2 + (Z - center[2]) ** 2
    return dist_sq <= radius ** 2
