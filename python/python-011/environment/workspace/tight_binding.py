# -*- coding: utf-8 -*-

import numpy as np
from scipy import sparse as sp
from scipy.sparse import linalg as spla
from scipy.interpolate import CubicSpline


def build_tight_binding_hamiltonian(Nx, Ny, t=1.0, tp=0.3, mu=0.0, open_boundary=False):
    if Nx < 1 or Ny < 1:
        raise ValueError("Nx, Ny 必须 >= 1。")
    N = Nx * Ny
    rows = []
    cols = []
    vals = []

    def idx(ix, iy):

        ix = ix % Nx
        iy = iy % Ny
        return ix + iy * Nx

    for iy in range(Ny):
        for ix in range(Nx):
            i = idx(ix, iy)

            rows.append(i)
            cols.append(i)
            vals.append(-mu)


            jx = ix + 1
            if jx < Nx or not open_boundary:
                j = idx(jx, iy)
                rows.append(i)
                cols.append(j)
                vals.append(-t)
                rows.append(j)
                cols.append(i)
                vals.append(-t)


            jy = iy + 1
            if jy < Ny or not open_boundary:
                j = idx(ix, jy)
                rows.append(i)
                cols.append(j)
                vals.append(-t)
                rows.append(j)
                cols.append(i)
                vals.append(-t)


            jx = ix + 1
            jy = iy + 1
            if (jx < Nx or not open_boundary) and (jy < Ny or not open_boundary):
                j = idx(jx, jy)
                rows.append(i)
                cols.append(j)
                vals.append(tp)
                rows.append(j)
                cols.append(i)
                vals.append(tp)


            jx = ix + 1
            jy = iy - 1
            if (jx < Nx or not open_boundary) and (jy >= 0 or not open_boundary):
                j = idx(jx, jy)
                rows.append(i)
                cols.append(j)
                vals.append(tp)
                rows.append(j)
                cols.append(i)
                vals.append(tp)

    H = sp.csr_matrix((vals, (rows, cols)), shape=(N, N), dtype=float)
    return H


def dense_to_ccs(A_dense, tol=1e-14):
    A_dense = np.asarray(A_dense, dtype=float)
    m, n = A_dense.shape
    colptr = [0]
    rowind = []
    vals = []
    for j in range(n):
        for i in range(m):
            if abs(A_dense[i, j]) > tol:
                rowind.append(i)
                vals.append(A_dense[i, j])
        colptr.append(len(vals))
    return len(vals), np.array(colptr, dtype=int), np.array(rowind, dtype=int), np.array(vals, dtype=float)


def ccs_to_csr(colptr, rowind, vals, m, n):

    csc = sp.csc_matrix((vals, rowind, colptr), shape=(m, n))
    return csc.tocsr()


def dispersion_2d(kx, ky, t=1.0, tp=0.3, mu=0.0):

    raise NotImplementedError("Hole_1: implement ε(k) = -2t(cos kx + cos ky) + 4t' cos kx cos ky - μ")


def band_structure_interpolation(k_points, energies, k_fine):
    k_points = np.asarray(k_points, dtype=float)
    energies = np.asarray(energies, dtype=float)
    k_fine = np.asarray(k_fine, dtype=float)
    if k_points.size < 2:
        raise ValueError("至少需要 2 个 k 点进行插值。")
    if not np.all(np.diff(k_points) > 0):
        raise ValueError("k_points 必须严格递增。")
    cs = CubicSpline(k_points, energies, bc_type='natural')
    return cs(k_fine)


def diagonalize_sparse_hamiltonian(H, k=6, sigma=None):
    if H.shape[0] != H.shape[1]:
        raise ValueError("H 必须是方阵。")
    N = H.shape[0]
    k = min(k, N - 1)
    if k < 1:
        return np.array([]), np.zeros((N, 0))
    try:
        if sigma is not None:
            eigvals, eigvecs = spla.eigsh(H, k=k, sigma=sigma, which='LM')
        else:
            eigvals, eigvecs = spla.eigsh(H, k=k, which='SA')
    except Exception as e:

        H_dense = H.toarray()
        eigvals, eigvecs = np.linalg.eigh(H_dense)
        eigvals = eigvals[:k]
        eigvecs = eigvecs[:, :k]
    return eigvals, eigvecs


def validate_hamiltonian_hermiticity(H, tol=1e-12):
    diff = H - H.T.conj()
    norm = np.linalg.norm(diff.toarray(), ord='fro')
    return norm < tol
