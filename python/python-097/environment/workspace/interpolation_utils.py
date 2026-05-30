
import numpy as np


def lagrange_basis_1d(xd, xi):
    xd = np.asarray(xd).flatten()
    xi = np.asarray(xi).flatten()
    nd = len(xd)
    ni = len(xi)

    lb = np.ones((ni, nd))
    for j in range(nd):
        for m in range(nd):
            if m != j:
                denom = xd[j] - xd[m]
                if abs(denom) < 1e-15:
                    denom = 1e-15
                lb[:, j] *= (xi - xd[m]) / denom

    return lb


def lagrange_value_1d(xd, yd, xi):
    lb = lagrange_basis_1d(xd, xi)
    yd = np.asarray(yd).flatten()
    yi = lb @ yd
    return yi


def lagrange_derivative_1d(xd, yd, xi):
    xd = np.asarray(xd).flatten()
    yd = np.asarray(yd).flatten()
    xi = np.asarray(xi).flatten()
    nd = len(xd)
    ni = len(xi)

    dy = np.zeros((ni, nd))
    for j in range(nd):
        for i in range(ni):
            dLj = 0.0
            for m in range(nd):
                if m != j:
                    denom = xd[j] - xd[m]
                    if abs(denom) < 1e-15:
                        denom = 1e-15
                    prod = 1.0 / denom
                    for k in range(nd):
                        if k != j and k != m:
                            denom2 = xd[j] - xd[k]
                            if abs(denom2) < 1e-15:
                                denom2 = 1e-15
                            prod *= (xi[i] - xd[k]) / denom2
                    dLj += prod
            dy[i, j] = dLj

    dyi = dy @ yd
    return dyi


def trilinear_interpolation(field, x, y, z, x_grid, y_grid, z_grid):
    scalar_input = np.isscalar(x)
    if scalar_input:
        x, y, z = np.array([x]), np.array([y]), np.array([z])
    else:
        x = np.asarray(x)
        y = np.asarray(y)
        z = np.asarray(z)

    nx, ny, nz = field.shape
    dx = x_grid[1] - x_grid[0] if nx > 1 else 1.0
    dy = y_grid[1] - y_grid[0] if ny > 1 else 1.0
    dz = z_grid[1] - z_grid[0] if nz > 1 else 1.0


    ix = (x - x_grid[0]) / dx
    iy = (y - y_grid[0]) / dy
    iz = (z - z_grid[0]) / dz


    ix = np.clip(ix, 0, nx - 2)
    iy = np.clip(iy, 0, ny - 2)
    iz = np.clip(iz, 0, nz - 2)

    i0 = np.floor(ix).astype(int)
    j0 = np.floor(iy).astype(int)
    k0 = np.floor(iz).astype(int)

    i1 = np.minimum(i0 + 1, nx - 1)
    j1 = np.minimum(j0 + 1, ny - 1)
    k1 = np.minimum(k0 + 1, nz - 1)

    tx = ix - i0
    ty = iy - j0
    tz = iz - k0


    c000 = field[i0, j0, k0]
    c001 = field[i0, j0, k1]
    c010 = field[i0, j1, k0]
    c011 = field[i0, j1, k1]
    c100 = field[i1, j0, k0]
    c101 = field[i1, j0, k1]
    c110 = field[i1, j1, k0]
    c111 = field[i1, j1, k1]

    c00 = c000 * (1 - tz) + c001 * tz
    c01 = c010 * (1 - tz) + c011 * tz
    c10 = c100 * (1 - tz) + c101 * tz
    c11 = c110 * (1 - tz) + c111 * tz

    c0 = c00 * (1 - ty) + c01 * ty
    c1 = c10 * (1 - ty) + c11 * ty

    value = c0 * (1 - tx) + c1 * tx

    if scalar_input:
        return float(value)
    return value


def chebyshev_nodes_1d(n, a=-1.0, b=1.0):
    k = np.arange(n)
    nodes = 0.5 * (a + b) + 0.5 * (b - a) * np.cos((2.0 * k + 1.0) * np.pi / (2.0 * n))
    return nodes


def interpolate_material_profile(z_coords, epsilon_profile, z_query, method='lagrange'):
    if method == 'linear':
        return np.interp(z_query, z_coords, epsilon_profile)
    elif method == 'lagrange':

        result = np.zeros_like(z_query)
        for i, zq in enumerate(z_query):

            idx = np.argsort(np.abs(z_coords - zq))[:4]
            idx = np.sort(idx)
            xd = z_coords[idx]
            yd = epsilon_profile[idx]
            result[i] = lagrange_value_1d(xd, yd, np.array([zq]))[0]
        return result
    else:
        raise ValueError(f"未知插值方法: {method}")
