# -*- coding: utf-8 -*-

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as spla


def assemble_poisson_matrix_2d(nx, ny, dx, dy, dielectric=None):
    if nx < 1 or ny < 1:
        raise ValueError("nx 和 ny 必须 >= 1。")
    if dx <= 0 or dy <= 0:
        raise ValueError("dx 和 dy 必须为正。")

    N = nx * ny
    if dielectric is None:
        dielectric = np.ones((nx, ny), dtype=float)
    else:
        dielectric = np.asarray(dielectric, dtype=float)
        if dielectric.shape != (nx, ny):
            raise ValueError("dielectric 形状必须与 (nx, ny) 匹配。")


    row_inds = []
    col_inds = []
    data_vals = []

    def idx(i, j):
        return i * ny + j

    for i in range(nx):
        for j in range(ny):
            center = idx(i, j)
            eps_c = dielectric[i, j]


            diag_val = 0.0


            if i > 0:
                eps_w = 0.5 * (eps_c + dielectric[i - 1, j])
                coeff = eps_w / dx**2
                row_inds.append(center)
                col_inds.append(idx(i - 1, j))
                data_vals.append(coeff)
                diag_val -= coeff
            if i < nx - 1:
                eps_e = 0.5 * (eps_c + dielectric[i + 1, j])
                coeff = eps_e / dx**2
                row_inds.append(center)
                col_inds.append(idx(i + 1, j))
                data_vals.append(coeff)
                diag_val -= coeff


            if j > 0:
                eps_s = 0.5 * (eps_c + dielectric[i, j - 1])
                coeff = eps_s / dy**2
                row_inds.append(center)
                col_inds.append(idx(i, j - 1))
                data_vals.append(coeff)
                diag_val -= coeff
            if j < ny - 1:
                eps_n = 0.5 * (eps_c + dielectric[i, j + 1])
                coeff = eps_n / dy**2
                row_inds.append(center)
                col_inds.append(idx(i, j + 1))
                data_vals.append(coeff)
                diag_val -= coeff


            row_inds.append(center)
            col_inds.append(center)
            data_vals.append(diag_val)

    A = sparse.coo_matrix((data_vals, (row_inds, col_inds)), shape=(N, N)).tocsr()
    return A


def solve_poisson_2d(rho, nx, ny, dx, dy, dielectric=None, tol=1e-10, max_iter=5000):
    from physics_constants import EPSILON_0

    rho = np.asarray(rho, dtype=float)
    if rho.shape != (nx, ny):
        raise ValueError("rho 形状必须与 (nx, ny) 匹配。")

    A = assemble_poisson_matrix_2d(nx, ny, dx, dy, dielectric)
    b = -(rho / EPSILON_0).reshape(-1)



    def idx(i, j):
        return i * ny + j

    boundary_nodes = []
    for i in range(nx):
        for j in range(ny):
            if i == 0 or i == nx - 1 or j == 0 or j == ny - 1:
                boundary_nodes.append(idx(i, j))


    A = A.tolil()
    for node in boundary_nodes:
        A[node, :] = 0.0
        A[node, node] = 1.0
        b[node] = 0.0
    A = A.tocsr()


    try:
        phi_vec = spla.spsolve(A, b)
        info = 0
    except Exception as e:

        try:
            phi_vec, info = spla.bicgstab(A, b, rtol=tol, atol=0.0, maxiter=max_iter)
        except TypeError:
            phi_vec, info = spla.bicgstab(A, b, tol=tol, maxiter=max_iter)
        if info != 0:
            raise RuntimeError(f"泊松方程求解失败: {e}")

    phi = phi_vec.reshape((nx, ny))
    residual = np.linalg.norm(A @ phi_vec - b) / max(np.linalg.norm(b), 1e-30)
    return phi, residual, info


def compute_electric_field_from_potential(phi, dx, dy):
    phi = np.asarray(phi, dtype=float)
    nx, ny = phi.shape
    Ex = np.zeros((nx, ny), dtype=float)
    Ey = np.zeros((nx, ny), dtype=float)


    if nx > 2 and ny > 2:
        Ex[1:-1, :] = -(phi[2:, :] - phi[:-2, :]) / (2.0 * dx)
        Ey[:, 1:-1] = -(phi[:, 2:] - phi[:, :-2]) / (2.0 * dy)


    if nx > 1:
        Ex[0, :] = -(phi[1, :] - phi[0, :]) / dx
        Ex[-1, :] = -(phi[-1, :] - phi[-2, :]) / dx
    if ny > 1:
        Ey[:, 0] = -(phi[:, 1] - phi[:, 0]) / dy
        Ey[:, -1] = -(phi[:, -1] - phi[:, -2]) / dy

    return Ex, Ey


def block_assemble_sparse_matrix(block_generators, row_blks, col_blks):
    N_blk = len(row_blks)
    if len(col_blks) != N_blk or len(block_generators) != N_blk:
        raise ValueError("row_blks, col_blks, block_generators 长度必须一致。")

    i_blk = np.cumsum([0] + row_blks[:-1])
    j_blk = np.cumsum([0] + col_blks[:-1])

    II = []
    JJ = []
    VV = []

    for k in range(N_blk):
        A_kb = block_generators[k](k)
        r_dim, c_dim = row_blks[k], col_blks[k]
        if A_kb.shape != (r_dim, c_dim):
            raise ValueError(f"块 {k} 的形状不匹配。")

        n_els = r_dim * c_dim
        i_row = np.arange(r_dim)
        i_global = i_blk[k] + np.repeat(i_row[:, None], c_dim, axis=1).reshape(-1)
        j_global = j_blk[k] + np.tile(np.arange(c_dim), r_dim)
        vals = A_kb.reshape(-1)

        II.append(i_global)
        JJ.append(j_global)
        VV.append(vals)

    II = np.concatenate(II)
    JJ = np.concatenate(JJ)
    VV = np.concatenate(VV)

    M = sum(row_blks)
    N = sum(col_blks)
    A_sparse = sparse.coo_matrix((VV, (II, JJ)), shape=(M, N)).tocsr()
    return A_sparse
