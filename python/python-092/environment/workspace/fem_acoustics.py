
import numpy as np
from sparse_linalg import SparseCOO, assemble_sparse_from_triplets, conjugate_gradient



C_AIR = 343.0
RHO_AIR = 1.21


def shape_p1(xi, eta, zeta):
    N = np.array([
        1.0 - xi - eta - zeta,
        xi,
        eta,
        zeta
    ], dtype=float)
    return N


def shape_p1_grad():
    dN = np.array([
        [-1.0, -1.0, -1.0],
        [ 1.0,  0.0,  0.0],
        [ 0.0,  1.0,  0.0],
        [ 0.0,  0.0,  1.0]
    ], dtype=float)
    return dN


def tetrahedron_gauss_points():
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
    v0 = p[tet_nodes[0]]
    v1 = p[tet_nodes[1]]
    v2 = p[tet_nodes[2]]
    v3 = p[tet_nodes[3]]
    J = np.column_stack((v1 - v0, v2 - v0, v3 - v0))
    detJ = np.linalg.det(J)
    return J, detJ


def assemble_helmholtz_system(p, t, freq, source_node=None, source_strength=1.0,
                               absorption_bc=None, boundary_nodes=None):
    n_nodes = p.shape[0]
    k = 2.0 * np.pi * freq / C_AIR

    gp, gw = tetrahedron_gauss_points()
    dN_ref = shape_p1_grad()

    K_rows, K_cols, K_vals = [], [], []
    M_rows, M_cols, M_vals = [], [], []

    for tet in t:
        J, detJ = compute_jacobian(p, tet)
        if abs(detJ) < 1e-14:
            continue
        vol = abs(detJ) / 6.0


        J_inv_T = np.linalg.inv(J).T
        dN_phys = (J_inv_T @ dN_ref.T).T








        for i in range(4):
            for j in range(4):
                kij = 0.0
                mij = 0.0
                K_rows.append(tet[i])
                K_cols.append(tet[j])
                K_vals.append(kij)
                M_rows.append(tet[i])
                M_cols.append(tet[j])
                M_vals.append(mij)

    K_sparse = assemble_sparse_from_triplets(K_rows, K_cols, K_vals, n_nodes)
    M_sparse = assemble_sparse_from_triplets(M_rows, M_cols, M_vals, n_nodes)



    A_rows, A_cols, A_vals = [], [], []

    for i in range(K_sparse.nnz):
        A_rows.append(K_sparse.rows[i])
        A_cols.append(K_sparse.cols[i])
        A_vals.append(K_sparse.vals[i])

    for i in range(M_sparse.nnz):
        A_rows.append(M_sparse.rows[i])
        A_cols.append(M_sparse.cols[i])
        A_vals.append(-(k ** 2) * M_sparse.vals[i])
    A_sparse = assemble_sparse_from_triplets(A_rows, A_cols, A_vals, n_nodes)


    F = np.zeros(n_nodes, dtype=complex)
    if source_node is not None and 0 <= source_node < n_nodes:
        F[source_node] = source_strength


    if absorption_bc is not None and boundary_nodes is not None:



        for bn in boundary_nodes:


            pass

    return A_sparse, K_sparse, M_sparse, F, k


def solve_helmholtz_direct(p, t, freq, source_node=None, source_strength=1.0):
    A_sparse, K_sparse, M_sparse, F, k = assemble_helmholtz_system(
        p, t, freq, source_node, source_strength
    )
    A_dense = A_sparse.to_dense()

    A_dense += 1e-10 * np.eye(A_dense.shape[0])
    p_sol = np.linalg.solve(A_dense, F)
    return p_sol, k


def solve_helmholtz_cg(p, t, freq, source_node=None, source_strength=1.0,
                        tol=1e-8, max_iter=None):
    A_sparse, K_sparse, M_sparse, F, k = assemble_helmholtz_system(
        p, t, freq, source_node, source_strength
    )

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
    p_ref = 20e-6
    p_rms = np.abs(pressure)
    p_rms = np.maximum(p_rms, 1e-15)
    return 20.0 * np.log10(p_rms / p_ref)


def compute_intensity(pressure, p, t, freq):
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

        v = 1j * grad_p / (RHO_AIR * omega)
        i_tet = 0.5 * np.real(np.conj(pressure[tet[0]]) * v)
        for node in tet:
            I[node] += np.real(i_tet)
            count[node] += 1.0
    count = np.maximum(count, 1.0)
    return I / count[:, None]
