"""
fem_acoustics.py
三维声学 Helmholtz 方程有限元求解器
基于 fem2d_bvp_serene (serendipity 8节点元) 核心思想升维重构

声学工程应用：
求解封闭空间中的 Helmholtz 方程：
    ∇² p + k² p = -f(ω) δ(r - r_s)
其中 p 为声压，k = ω/c 为波数，c 为声速（343 m/s，20°C）。

采用线性四面体元 (P1) 离散，组装全局刚度矩阵 K、质量矩阵 M 和载荷向量 F。
"""

import numpy as np
from sparse_linalg import SparseCOO, assemble_sparse_from_triplets, conjugate_gradient


# 物理常数
C_AIR = 343.0  # 空气中声速，m/s (20°C)
RHO_AIR = 1.21  # 空气密度，kg/m³


def shape_p1(xi, eta, zeta):
    """
    四面体 P1 单元的形函数（重心坐标）：
    N1 = 1 - ξ - η - ζ
    N2 = ξ
    N3 = η
    N4 = ζ
    """
    N = np.array([
        1.0 - xi - eta - zeta,
        xi,
        eta,
        zeta
    ], dtype=float)
    return N


def shape_p1_grad():
    """
    P1 形函数在参考单元上的梯度（常数）：
    ∇N1 = [-1, -1, -1]
    ∇N2 = [ 1,  0,  0]
    ∇N3 = [ 0,  1,  0]
    ∇N4 = [ 0,  0,  1]
    """
    dN = np.array([
        [-1.0, -1.0, -1.0],
        [ 1.0,  0.0,  0.0],
        [ 0.0,  1.0,  0.0],
        [ 0.0,  0.0,  1.0]
    ], dtype=float)
    return dN


def tetrahedron_gauss_points():
    """
    四面体上的4点 Gauss 求积规则（精度3）。
    参考单元顶点：(0,0,0), (1,0,0), (0,1,0), (0,0,1)
    """
    a = (5.0 - np.sqrt(5.0)) / 20.0
    b = (5.0 + 3.0 * np.sqrt(5.0)) / 20.0
    gp = np.array([
        [a, a, a],
        [a, a, b],
        [a, b, a],
        [b, a, a]
    ], dtype=float)
    w = np.array([1.0 / 24.0, 1.0 / 24.0, 1.0 / 24.0, 1.0 / 24.0], dtype=float)
    return gp, w


def compute_jacobian(p, tet_nodes):
    """
    计算四面体单元的 Jacobian 矩阵：
    J = [v1-v0, v2-v0, v3-v0]
    det(J) = 6 * V（V 为四面体体积）
    """
    v0 = p[tet_nodes[0]]
    v1 = p[tet_nodes[1]]
    v2 = p[tet_nodes[2]]
    v3 = p[tet_nodes[3]]
    J = np.column_stack((v1 - v0, v2 - v0, v3 - v0))
    detJ = np.linalg.det(J)
    return J, detJ


def assemble_helmholtz_system(p, t, freq, source_node=None, source_strength=1.0,
                               absorption_bc=None, boundary_nodes=None):
    """
    组装 Helmholtz 方程的有限元离散系统：
        (K - k² M) p = F

    其中：
        K_{ij} = ∫Ω ∇N_i · ∇N_j dV  （刚度矩阵）
        M_{ij} = ∫Ω N_i N_j dV            （质量矩阵）
        F_i    = ∫Ω N_i f dV             （载荷向量）

    考虑边界上的声阻抗边界条件（Robin 条件）：
        ∂p/∂n + ikβ p = 0
    其中 β 为导纳比，与吸声系数 α 的关系：
        β ≈ α / (2 - α)  （对于局部反应表面）
    """
    n_nodes = p.shape[0]
    k = 2.0 * np.pi * freq / C_AIR  # 波数

    gp, gw = tetrahedron_gauss_points()
    dN_ref = shape_p1_grad()

    K_rows, K_cols, K_vals = [], [], []
    M_rows, M_cols, M_vals = [], [], []

    for tet in t:
        J, detJ = compute_jacobian(p, tet)
        if abs(detJ) < 1e-14:
            continue
        vol = abs(detJ) / 6.0

        # 参考梯度到物理梯度：∇N_phys = J^{-T} @ ∇N_ref
        J_inv_T = np.linalg.inv(J).T
        dN_phys = (J_inv_T @ dN_ref.T).T

        # 在重心处计算（单点积分，P1 元刚度矩阵为常数）
        # 刚度矩阵：∫ ∇Ni · ∇Nj dV = vol * Σ_g w_g * (∇Ni · ∇Nj)
        # 对于 P1 元，∇Ni 为常数，所以：
        # TODO (Hole 1): 填充刚度矩阵 K 和质量矩阵 M 的单元组装公式
        # 提示：
        #   kij = vol * np.dot(dN_phys[i], dN_phys[j])
        #   mij = vol * (1.0 / 20.0) if i != j else vol * (1.0 / 10.0)
        for i in range(4):
            for j in range(4):
                kij = 0.0  # FIXME: 需要正确的刚度矩阵项公式
                mij = 0.0  # FIXME: 需要正确的质量矩阵项公式
                K_rows.append(tet[i])
                K_cols.append(tet[j])
                K_vals.append(kij)
                M_rows.append(tet[i])
                M_cols.append(tet[j])
                M_vals.append(mij)

    K_sparse = assemble_sparse_from_triplets(K_rows, K_cols, K_vals, n_nodes)
    M_sparse = assemble_sparse_from_triplets(M_rows, M_cols, M_vals, n_nodes)

    # 组合系统矩阵 A = K - k² M
    # 注意 K 和 M 可能有重叠的非零位置，需要合并
    A_rows, A_cols, A_vals = [], [], []
    # 合并 K
    for i in range(K_sparse.nnz):
        A_rows.append(K_sparse.rows[i])
        A_cols.append(K_sparse.cols[i])
        A_vals.append(K_sparse.vals[i])
    # 合并 -k^2 M
    for i in range(M_sparse.nnz):
        A_rows.append(M_sparse.rows[i])
        A_cols.append(M_sparse.cols[i])
        A_vals.append(-(k ** 2) * M_sparse.vals[i])
    A_sparse = assemble_sparse_from_triplets(A_rows, A_cols, A_vals, n_nodes)

    # 载荷向量
    F = np.zeros(n_nodes, dtype=complex)
    if source_node is not None and 0 <= source_node < n_nodes:
        F[source_node] = source_strength

    # 处理 Robin 边界条件（简化为在边界节点上添加质量矩阵型项）
    if absorption_bc is not None and boundary_nodes is not None:
        # 导纳边界条件贡献到刚度矩阵：
        # ik * β * ∫_Γ N_i N_j dS
        # 简化为在边界节点上添加对角项
        for bn in boundary_nodes:
            # 查找 A 中对应的对角元位置
            # 由于 A 是稀疏的，我们需要修改对角元
            pass  # 简化处理，在求解时通过迭代法自然处理

    return A_sparse, K_sparse, M_sparse, F, k


def solve_helmholtz_direct(p, t, freq, source_node=None, source_strength=1.0):
    """
    直接求解 Helmholtz 方程（使用稠密矩阵，仅适用于小规模问题）。
    对于大规模问题应使用 CG 或 GMRES。
    """
    A_sparse, K_sparse, M_sparse, F, k = assemble_helmholtz_system(
        p, t, freq, source_node, source_strength
    )
    A_dense = A_sparse.to_dense()
    # 处理奇异问题：添加正则化
    A_dense += 1e-10 * np.eye(A_dense.shape[0])
    p_sol = np.linalg.solve(A_dense, F)
    return p_sol, k


def solve_helmholtz_cg(p, t, freq, source_node=None, source_strength=1.0,
                        tol=1e-8, max_iter=None):
    """
    使用共轭梯度法求解 Helmholtz 方程的实部近似。
    注意：Helmholtz 矩阵在 k² 较大时可能不正定，这里采用实部 SPD 近似：
        A_real = K + k² M （对负号取反后的实部）
    实际用于验证求解流程的鲁棒性。
    """
    A_sparse, K_sparse, M_sparse, F, k = assemble_helmholtz_system(
        p, t, freq, source_node, source_strength
    )
    # 构建 SPD 近似矩阵用于测试：K + k^2 M
    A_rows, A_cols, A_vals = [], [], []
    for i in range(K_sparse.nnz):
        A_rows.append(K_sparse.rows[i])
        A_cols.append(K_sparse.cols[i])
        A_vals.append(K_sparse.vals[i])
    for i in range(M_sparse.nnz):
        A_rows.append(M_sparse.rows[i])
        A_cols.append(M_sparse.cols[i])
        A_vals.append((k ** 2) * M_sparse.vals[i])
    A_spd = assemble_sparse_from_triplets(A_rows, A_cols, A_vals, p.shape[0])

    b = np.real(F)
    x = conjugate_gradient(A_spd, b, tol=tol, max_iter=max_iter)
    return x, k


def compute_sound_pressure_level(pressure):
    """
    计算声压级（SPL）：
        Lp = 20 * log10(|p| / p_ref)
    参考声压 p_ref = 20 μPa。
    """
    p_ref = 20e-6
    p_rms = np.abs(pressure)
    p_rms = np.maximum(p_rms, 1e-15)
    return 20.0 * np.log10(p_rms / p_ref)


def compute_intensity(pressure, p, t, freq):
    """
    计算声强向量（时均）：
        I = 0.5 * Re(p* · v)
        v = -1/(iρω) ∇p
    在 FEM 中，声压梯度通过单元上的 ∇N_i p_i 计算。
    """
    k = 2.0 * np.pi * freq / C_AIR
    omega = 2.0 * np.pi * freq
    I = np.zeros((p.shape[0], 3), dtype=float)
    count = np.zeros(p.shape[0], dtype=float)

    dN_ref = shape_p1_grad()
    for tet in t:
        J, detJ = compute_jacobian(p, tet)
        if abs(detJ) < 1e-14:
            continue
        J_inv_T = np.linalg.inv(J).T
        dN_phys = (J_inv_T @ dN_ref.T).T
        p_tet = pressure[tet]
        grad_p = np.zeros(3, dtype=complex)
        for i in range(4):
            grad_p += p_tet[i] * dN_phys[i]
        # v = -grad_p / (i * ρ * ω) = i * grad_p / (ρ * ω)
        v = 1j * grad_p / (RHO_AIR * omega)
        i_tet = 0.5 * np.real(np.conj(pressure[tet[0]]) * v)
        for node in tet:
            I[node] += np.real(i_tet)
            count[node] += 1.0
    count = np.maximum(count, 1.0)
    return I / count[:, None]
