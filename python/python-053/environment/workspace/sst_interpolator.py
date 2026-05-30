
import numpy as np
from typing import Tuple


def lagrange_basis_1d(m: int, xd: np.ndarray, i: int, x: float) -> float:
    if i < 0 or i > m:
        raise ValueError("Index i out of range")


    if np.isclose(x, xd[i]):
        return 1.0


    for j in range(m + 1):
        if j != i and np.isclose(xd[i], xd[j]):
            raise ValueError(f"Duplicate nodes detected: xd[{i}] == xd[{j}]")

    value = 1.0
    for j in range(m + 1):
        if j != i:
            denom = xd[i] - xd[j]
            if abs(denom) < 1e-14:
                raise ValueError("Denominator too small in Lagrange basis")
            value *= (x - xd[j]) / denom
    return float(value)


def lagrange_interp_2d(mx: int, my: int,
                       xd_1d: np.ndarray, yd_1d: np.ndarray,
                       zd: np.ndarray,
                       xi: np.ndarray, yi: np.ndarray) -> np.ndarray:
    if xd_1d.shape[0] != mx + 1:
        raise ValueError("xd_1d length must be mx+1")
    if yd_1d.shape[0] != my + 1:
        raise ValueError("yd_1d length must be my+1")
    if zd.shape[0] != (mx + 1) * (my + 1):
        raise ValueError("zd length must be (mx+1)*(my+1)")
    if xi.shape != yi.shape:
        raise ValueError("xi and yi must have the same shape")

    ni = xi.shape[0]
    zi = np.zeros(ni, dtype=float)

    for k in range(ni):

        x_min, x_max = xd_1d.min(), xd_1d.max()
        y_min, y_max = yd_1d.min(), yd_1d.max()
        if xi[k] < x_min - 1e-10 or xi[k] > x_max + 1e-10:

            scale = 0.0
        elif yi[k] < y_min - 1e-10 or yi[k] > y_max + 1e-10:
            scale = 0.0
        else:
            scale = 1.0

        val = 0.0
        l = 0
        for i_idx in range(mx + 1):
            lx = lagrange_basis_1d(mx, xd_1d, i_idx, xi[k])
            for j_idx in range(my + 1):
                ly = lagrange_basis_1d(my, yd_1d, j_idx, yi[k])
                val += zd[l] * lx * ly
                l += 1
        zi[k] = val * scale

    return zi


def chebyshev_nodes_1d(n: int, a: float, b: float) -> np.ndarray:
    if n < 1:
        raise ValueError("n must be at least 1")
    if b <= a:
        raise ValueError("b must be greater than a")
    k = np.arange(n)
    nodes = 0.5 * (a + b) + 0.5 * (b - a) * np.cos(np.pi * (2.0 * k + 1.0) / (2.0 * n))
    return nodes


def interpolate_sst_field(lon_grid: np.ndarray, lat_grid: np.ndarray,
                          sst_data: np.ndarray,
                          lon_target: np.ndarray, lat_target: np.ndarray,
                          degree_lon: int = 5, degree_lat: int = 5) -> np.ndarray:
    if lon_grid.ndim != 1 or lat_grid.ndim != 1:
        raise ValueError("lon_grid and lat_grid must be 1D")
    if sst_data.shape != (lon_grid.shape[0], lat_grid.shape[0]):
        raise ValueError("sst_data shape mismatch with grid dimensions")

    ni = lon_target.shape[0]
    sst_interp = np.zeros(ni, dtype=float)


    nx, ny = lon_grid.shape[0], lat_grid.shape[0]
    n_block_x = max(1, nx // (degree_lon + 1))
    n_block_y = max(1, ny // (degree_lat + 1))

    for k in range(ni):
        x, y = lon_target[k], lat_target[k]


        ix = min(n_block_x - 1, max(0, int((x - lon_grid[0]) / (lon_grid[-1] - lon_grid[0]) * n_block_x)))
        iy = min(n_block_y - 1, max(0, int((y - lat_grid[0]) / (lat_grid[-1] - lat_grid[0]) * n_block_y)))

        i0 = ix * (degree_lon + 1)
        i1 = min(nx, i0 + degree_lon + 1)
        j0 = iy * (degree_lat + 1)
        j1 = min(ny, j0 + degree_lat + 1)

        if i1 - i0 < 2 or j1 - j0 < 2:

            i_near = np.argmin(np.abs(lon_grid - x))
            j_near = np.argmin(np.abs(lat_grid - y))
            sst_interp[k] = sst_data[i_near, j_near]
            continue


        xd = lon_grid[i0:i1]
        yd = lat_grid[j0:j1]
        mx = xd.shape[0] - 1
        my = yd.shape[0] - 1


        zd = np.zeros((mx + 1) * (my + 1), dtype=float)
        l = 0
        for jj in range(j0, j1):
            for ii in range(i0, i1):
                zd[l] = sst_data[ii, jj]
                l += 1


        result = lagrange_interp_2d(mx, my, xd, yd, zd,
                                     np.array([x]), np.array([y]))
        sst_interp[k] = result[0]

    return sst_interp


def nino34_index(sst_anomaly: np.ndarray, lon: np.ndarray, lat: np.ndarray) -> float:
    if sst_anomaly.shape != (lon.shape[0], lat.shape[0]):
        raise ValueError("Array shape mismatch")


    mask = ((lon >= -170.0) & (lon <= -120.0))[:, None] & \
           ((lat >= -5.0) & (lat <= 5.0))[None, :]

    if not np.any(mask):
        return 0.0


    cos_lat = np.cos(np.radians(lat))[None, :]
    weights = mask.astype(float) * cos_lat
    total_weight = np.sum(weights)

    if total_weight < 1e-14:
        return 0.0

    index = np.sum(sst_anomaly * weights) / total_weight
    return float(index)
