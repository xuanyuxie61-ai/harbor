# -*- coding: utf-8 -*-
"""
tight_binding.py
----------------
二维方格紧束缚模型哈密顿量构建与稀疏矩阵存储。

对应种子项目：
  - 457_ge_to_ccs：稠密矩阵到 CCS 稀疏格式的转换
  - 593_interp_ncs：自然三次样条用于能带插值
  - 1206_test_eigen：测试矩阵验证对角化器

物理模型：
  二维方格晶格上的单带紧束缚哈密顿量
      H = -t Σ_{<i,j>} (c_i^† c_j + h.c.)
        + t' Σ_{<<i,j>>} (c_i^† c_j + h.c.)
        - μ Σ_i c_i^† c_i
  其中 t 为最近邻跃迁，t' 为次近邻跃迁，μ 为化学势。
  动量空间表示：
      ε(k) = -2t (cos k_x + cos k_y) + 4t' cos k_x cos k_y - μ
"""

import numpy as np
from scipy import sparse as sp
from scipy.sparse import linalg as spla
from scipy.interpolate import CubicSpline


def build_tight_binding_hamiltonian(Nx, Ny, t=1.0, tp=0.3, mu=0.0, open_boundary=False):
    """
    构建二维 Nx × Ny 方格紧束缚哈密顿量的稀疏矩阵表示。

    Parameters
    ----------
    Nx, Ny : int
        x, y 方向的格点数，必须 >=1。
    t : float
        最近邻跃迁积分（通常取为能量单位）。
    tp : float
        次近邻跃迁积分。
    mu : float
        化学势。
    open_boundary : bool
        若为 True 使用开边界；否则周期边界。

    Returns
    -------
    H : scipy.sparse.csr_matrix, shape (Nx*Ny, Nx*Ny)
        哈密顿量稀疏矩阵。
    """
    if Nx < 1 or Ny < 1:
        raise ValueError("Nx, Ny 必须 >= 1。")
    N = Nx * Ny
    rows = []
    cols = []
    vals = []

    def idx(ix, iy):
        # 将二维坐标展平为一维索引
        ix = ix % Nx
        iy = iy % Ny
        return ix + iy * Nx

    for iy in range(Ny):
        for ix in range(Nx):
            i = idx(ix, iy)
            # 在位能
            rows.append(i)
            cols.append(i)
            vals.append(-mu)

            # 最近邻 +x
            jx = ix + 1
            if jx < Nx or not open_boundary:
                j = idx(jx, iy)
                rows.append(i)
                cols.append(j)
                vals.append(-t)
                rows.append(j)
                cols.append(i)
                vals.append(-t)

            # 最近邻 +y
            jy = iy + 1
            if jy < Ny or not open_boundary:
                j = idx(ix, jy)
                rows.append(i)
                cols.append(j)
                vals.append(-t)
                rows.append(j)
                cols.append(i)
                vals.append(-t)

            # 次近邻 +x,+y
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

            # 次近邻 +x,-y
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
    """
    将稠密矩阵转换为 CCS (Compressed Column Storage) 格式。

    返回 nz_num, colptr, rowind, vals。
    其中 colptr 长度为 n+1，colptr[j] 到 colptr[j+1]-1 为第 j 列的非零元。
    """
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
    """
    将 CCS 格式转换为 scipy CSR 矩阵。
    """
    # 先转 CSC 再转 CSR
    csc = sp.csc_matrix((vals, rowind, colptr), shape=(m, n))
    return csc.tocsr()


def dispersion_2d(kx, ky, t=1.0, tp=0.3, mu=0.0):
    """
    二维方格紧束缚色散关系：
        ε(k) = -2t(cos k_x + cos k_y) + 4t' cos k_x cos k_y - μ
    """
    # TODO: Hole_1 - implement the 2D tight-binding dispersion relation
    raise NotImplementedError("Hole_1: implement ε(k) = -2t(cos kx + cos ky) + 4t' cos kx cos ky - μ")


def band_structure_interpolation(k_points, energies, k_fine):
    """
    使用自然三次样条对能带进行插值。

    对应种子项目 593_interp_ncs。
    """
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
    """
    对稀疏哈密顿量求解前 k 个本征值/本征向量。
    使用 ARPACK 的 shift-invert 模式（若 sigma 给定）。

    Parameters
    ----------
    H : scipy.sparse.csr_matrix
    k : int
        求解本征值数量。
    sigma : float, optional
        shift 值，用于寻找费米面附近的态。

    Returns
    -------
    eigvals : ndarray, shape (k,)
    eigvecs : ndarray, shape (N, k)
    """
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
        # 回退到稠密矩阵
        H_dense = H.toarray()
        eigvals, eigvecs = np.linalg.eigh(H_dense)
        eigvals = eigvals[:k]
        eigvecs = eigvecs[:, :k]
    return eigvals, eigvecs


def validate_hamiltonian_hermiticity(H, tol=1e-12):
    """
    验证 H 的厄米性：||H - H^†||_F < tol。
    """
    diff = H - H.T.conj()
    norm = np.linalg.norm(diff.toarray(), ord='fro')
    return norm < tol
