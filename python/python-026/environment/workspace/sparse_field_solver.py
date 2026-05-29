# -*- coding: utf-8 -*-
"""
sparse_field_solver.py

基于 sparse_parfor 的稀疏矩阵并行组装与泊松方程求解模块。

原项目 1111_sparse_parfor 展示了如何将分块稀疏矩阵在多核上并行组装。
在激光-等离子体相互作用中，这一思想被用于:
    1. 在二维结构化网格上组装等离子体泊松方程的稀疏系数矩阵。
    2. 使用有限差分法离散化 ∇·(ε ∇φ) = -ρ/ε_0。
    3. 求解等离子体电势 φ 以计算有质动力修正。

核心公式:
    泊松方程:
        ∇² φ = -ρ / ε_0
    其中 ρ = e * (Z*n_i - n_e) 为电荷密度。

    二维五点差分格式 (在单元中心网格上):
        (φ_{i+1,j} - 2φ_{i,j} + φ_{i-1,j})/dx² + (φ_{i,j+1} - 2φ_{i,j} + φ_{i,j-1})/dy² = -ρ_{i,j}/ε_0

    矩阵形式: A * φ_vec = b
    其中 A 为 (nxc*nyc) × (nxc*nyc) 的稀疏矩阵。
"""

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as spla


def assemble_poisson_matrix_2d(nx, ny, dx, dy, dielectric=None):
    """
    组装二维泊松方程的稀疏有限差分矩阵。

    Parameters
    ----------
    nx, ny : int
        x 和 y 方向的内部网格点数。
    dx, dy : float
        网格间距。
    dielectric : ndarray, shape (nx, ny), optional
        介电常数分布 ε/ε_0，默认为 1（真空）。

    Returns
    -------
    A : scipy.sparse.csr_matrix
        稀疏系数矩阵，形状 (nx*ny, nx*ny)。
    """
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

    # 使用稀疏 COO 格式组装
    row_inds = []
    col_inds = []
    data_vals = []

    def idx(i, j):
        return i * ny + j

    for i in range(nx):
        for j in range(ny):
            center = idx(i, j)
            eps_c = dielectric[i, j]

            # 对角元
            diag_val = 0.0

            # x 方向邻居
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

            # y 方向邻居
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

            # 中心对角元
            row_inds.append(center)
            col_inds.append(center)
            data_vals.append(diag_val)

    A = sparse.coo_matrix((data_vals, (row_inds, col_inds)), shape=(N, N)).tocsr()
    return A


def solve_poisson_2d(rho, nx, ny, dx, dy, dielectric=None, tol=1e-10, max_iter=5000):
    """
    求解二维泊松方程 ∇·(ε ∇φ) = -ρ/ε_0，带零 Dirichlet 边界条件。

    Parameters
    ----------
    rho : ndarray, shape (nx, ny)
        电荷密度分布 [C/m^3]。
    nx, ny : int
        内部网格点数。
    dx, dy : float
        网格间距 [m]。
    dielectric : ndarray, optional
        相对介电常数分布。
    tol : float, optional
        求解容差。
    max_iter : int, optional
        最大迭代次数。

    Returns
    -------
    phi : ndarray, shape (nx, ny)
        电势分布 [V]。
    residual : float
        残差范数。
    info : int
        求解器返回信息。
    """
    from physics_constants import EPSILON_0

    rho = np.asarray(rho, dtype=float)
    if rho.shape != (nx, ny):
        raise ValueError("rho 形状必须与 (nx, ny) 匹配。")

    A = assemble_poisson_matrix_2d(nx, ny, dx, dy, dielectric)
    b = -(rho / EPSILON_0).reshape(-1)

    # 添加 Dirichlet 边界条件: 边界上的点直接设为零
    # 对边界行，将对角元设为 1，非对角元设为 0，RHS 设为 0
    def idx(i, j):
        return i * ny + j

    boundary_nodes = []
    for i in range(nx):
        for j in range(ny):
            if i == 0 or i == nx - 1 or j == 0 or j == ny - 1:
                boundary_nodes.append(idx(i, j))

    # 将边界条件直接融入 COO 数据（通过修改 CSR）
    A = A.tolil()
    for node in boundary_nodes:
        A[node, :] = 0.0
        A[node, node] = 1.0
        b[node] = 0.0
    A = A.tocsr()

    # 对于中小规模矩阵，直接使用稀疏直接求解器（更鲁棒）
    try:
        phi_vec = spla.spsolve(A, b)
        info = 0
    except Exception as e:
        # 回退到 BiCGSTAB
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
    """
    由电势计算电场 E = -∇φ。

    使用中心差分:
        E_x[i,j] ≈ -(φ[i+1,j] - φ[i-1,j]) / (2dx)
        E_y[i,j] ≈ -(φ[i,j+1] - φ[i,j-1]) / (2dy)

    边界上使用单侧差分。

    Parameters
    ----------
    phi : ndarray, shape (nx, ny)
        电势分布 [V]。
    dx, dy : float
        网格间距 [m]。

    Returns
    -------
    Ex, Ey : ndarray
        电场分量 [V/m]。
    """
    phi = np.asarray(phi, dtype=float)
    nx, ny = phi.shape
    Ex = np.zeros((nx, ny), dtype=float)
    Ey = np.zeros((nx, ny), dtype=float)

    # 内部点中心差分
    if nx > 2 and ny > 2:
        Ex[1:-1, :] = -(phi[2:, :] - phi[:-2, :]) / (2.0 * dx)
        Ey[:, 1:-1] = -(phi[:, 2:] - phi[:, :-2]) / (2.0 * dy)

    # 边界单侧差分
    if nx > 1:
        Ex[0, :] = -(phi[1, :] - phi[0, :]) / dx
        Ex[-1, :] = -(phi[-1, :] - phi[-2, :]) / dx
    if ny > 1:
        Ey[:, 0] = -(phi[:, 1] - phi[:, 0]) / dy
        Ey[:, -1] = -(phi[:, -1] - phi[:, -2]) / dy

    return Ex, Ey


def block_assemble_sparse_matrix(block_generators, row_blks, col_blks):
    """
    基于原 sparse_parfor 思想的分块稀疏矩阵组装。

    将多个独立的块矩阵并行生成后组装为全局稀疏矩阵。
    （在纯 Python 中，使用列表推导模拟并行生成。）

    Parameters
    ----------
    block_generators : list of callable
        每个块矩阵的生成函数，签名为 f(k) -> ndarray(row_blks[k], col_blks[k])。
    row_blks : list of int
        每块的行维度。
    col_blks : list of int
        每块的列维度。

    Returns
    -------
    A_sparse : scipy.sparse.csr_matrix
        全局稀疏矩阵。
    """
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
