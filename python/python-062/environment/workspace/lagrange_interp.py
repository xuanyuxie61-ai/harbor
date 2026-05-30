
import numpy as np


def lagrange_basis_1d(x_nodes, x_target):
    n = len(x_nodes)
    L = np.ones(n, dtype=np.float64)

    for j in range(n):
        for k in range(n):
            if k != j:
                denom = x_nodes[j] - x_nodes[k]
                if abs(denom) < 1e-15:

                    L = np.zeros(n)
                    L[j] = 1.0
                    return L
                L[j] *= (x_target - x_nodes[k]) / denom

    return L


def lagrange_interp_nd(grid_nodes, values, target):
    target = np.atleast_1d(target)
    ndim = len(grid_nodes)

    if len(target) != ndim:
        raise ValueError("lagrange_interp_nd: 目标点维度与网格维度不匹配")

    if values.ndim != ndim:
        raise ValueError(f"lagrange_interp_nd: values.ndim={values.ndim} 不等于 ndim={ndim}")


    basis_list = []
    for d in range(ndim):
        Ld = lagrange_basis_1d(grid_nodes[d], target[d])
        basis_list.append(Ld)


    if ndim == 1:
        return float(np.dot(basis_list[0], values))
    elif ndim == 2:
        result = 0.0
        nx, ny = values.shape
        for i in range(min(nx, len(basis_list[0]))):
            for j in range(min(ny, len(basis_list[1]))):
                result += basis_list[0][i] * basis_list[1][j] * values[i, j]
        return float(result)
    elif ndim == 3:
        result = 0.0
        nx, ny, nz = values.shape
        for i in range(min(nx, len(basis_list[0]))):
            for j in range(min(ny, len(basis_list[1]))):
                for k in range(min(nz, len(basis_list[2]))):
                    result += (basis_list[0][i] * basis_list[1][j] *
                               basis_list[2][k] * values[i, j, k])
        return float(result)
    else:
        raise NotImplementedError("lagrange_interp_nd: 仅支持 1D/2D/3D")


def interpolate_velocity_to_particles(particles, grid, u_field, v_field, w_field, order=3):
    n_particle = particles.shape[0]
    u_p = np.zeros(n_particle, dtype=np.float64)
    v_p = np.zeros(n_particle, dtype=np.float64)
    w_p = np.zeros(n_particle, dtype=np.float64)

    x_grid, y_grid, z_grid = grid
    nx, ny, nz = u_field.shape

    for p in range(n_particle):
        pt = particles[p]


        def get_local_nodes(full_grid, x_target, n):
            idx = np.argmin(np.abs(full_grid - x_target))
            half = n // 2
            i0 = max(0, idx - half)
            i1 = min(len(full_grid), i0 + n)
            i0 = max(0, i1 - n)
            return full_grid[i0:i1], i0

        local_x, ix0 = get_local_nodes(x_grid, pt[0], order)
        local_y, iy0 = get_local_nodes(y_grid, pt[1], order)
        local_z, iz0 = get_local_nodes(z_grid, pt[2], order)

        ix1 = min(ix0 + order, nx)
        iy1 = min(iy0 + order, ny)
        iz1 = min(iz0 + order, nz)
        ix0 = max(0, ix1 - order)
        iy0 = max(0, iy1 - order)
        iz0 = max(0, iz1 - order)

        u_slice = u_field[ix0:ix1, iy0:iy1, iz0:iz1]
        v_slice = v_field[ix0:ix1, iy0:iy1, iz0:iz1]
        w_slice = w_field[ix0:ix1, iy0:iy1, iz0:iz1]

        if u_slice.size == 0:
            continue

        local_grid = [local_x, local_y, local_z]
        local_pt = pt.copy()

        for d in range(3):
            local_pt[d] = np.clip(local_pt[d], local_grid[d][0], local_grid[d][-1])

        u_p[p] = lagrange_interp_nd(local_grid, u_slice, local_pt)
        v_p[p] = lagrange_interp_nd(local_grid, v_slice, local_pt)
        w_p[p] = lagrange_interp_nd(local_grid, w_slice, local_pt)

    return u_p, v_p, w_p
