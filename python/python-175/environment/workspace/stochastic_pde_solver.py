
import numpy as np
from fem1d_solver import uniform_mesh_1d, solve_fem1d, fem1d_l2_error
from chaos_expansion import (gpc_projection_coefficients, gpc_mean_variance,
                              gpc_reconstruct, gpc_sobol_sensitivity,
                              gpc_total_order_sobol, enumerate_multi_indices_total_degree)
from quadrature_rules import smolyak_sparse_grid
from sparse_linear_solver import SparseMatrixCOO, conjugate_gradient_sparse


def kl_eigenvalues_1d_exponential(sigma, Lc, xL, xR, n_modes):
    L = xR - xL
    omegas = np.zeros(n_modes)

    for k in range(1, n_modes + 1):
        a = (k - 0.5) * np.pi / L
        b = k * np.pi / L


        for _ in range(60):
            mid = 0.5 * (a + b)
            if np.isclose(a, b, atol=1e-15):
                break
            fa = np.tan(a * L) + 2.0 * Lc * a
            fm = np.tan(mid * L) + 2.0 * Lc * mid
            if np.isnan(fa) or np.isinf(fa):
                a = mid
                continue
            if fa * fm <= 0:
                b = mid
            else:
                a = mid
        omegas[k - 1] = 0.5 * (a + b)
    lambdas = 2.0 * sigma ** 2 * Lc / (1.0 + (omegas * Lc) ** 2)
    return lambdas, omegas


def kl_eigenfunctions_1d(x, omegas, xL, xR):
    x = np.asarray(x, dtype=float)
    L = xR - xL
    n_modes = len(omegas)
    phi = np.zeros((len(x), n_modes))
    for k, w in enumerate(omegas):

        denom = np.sqrt(0.5 * L - np.sin(2.0 * w * L) / (4.0 * w))
        if abs(denom) < 1e-15:
            denom = np.sqrt(L / 2.0)
        phi[:, k] = np.sin(w * (x - xL)) / denom
    return phi


def diffusion_coefficient_kl(x, xi, a0, sigma, Lc, xL, xR):
    n_modes = len(xi)
    lambdas, omegas = kl_eigenvalues_1d_exponential(sigma, Lc, xL, xR, n_modes)
    phi = kl_eigenfunctions_1d(x, omegas, xL, xR)
    a = a0(x)
    if np.isscalar(a):
        a = np.full_like(x, a, dtype=float)
    else:
        a = np.asarray(a, dtype=float).copy()
    for k in range(n_modes):
        a += np.sqrt(max(lambdas[k], 0.0)) * phi[:, k] * xi[k]

    a_min = np.min(a)
    if a_min <= 0:
        a = a - a_min + 0.01
    return a


def solve_stochastic_pde_projection(xL, xR, n_elem, a0_func, f_func, bc_left, bc_right,
                                    n_modes, max_poly_degree, quad_level):
    d = n_modes
    mesh = uniform_mesh_1d(xL, xR, n_elem)
    n_nodes = len(mesh)


    index_set = enumerate_multi_indices_total_degree(d, max_poly_degree)
    N_poly = index_set.shape[0]


    xi_q, w_q = smolyak_sparse_grid(d, quad_level)
    if xi_q.size == 0:

        from quadrature_rules import gauss_legendre_tensor
        xi_q, w_q = gauss_legendre_tensor(d, quad_level + 1)

    w_q = w_q / (2.0 ** d)

    n_q = xi_q.shape[0]

    u_q = np.zeros((n_q, n_nodes))
    for q in range(n_q):
        xi = xi_q[q]

        def a_func(x):
            return diffusion_coefficient_kl(x, xi, a0_func, sigma=0.3, Lc=0.1, xL=xL, xR=xR)
        u_q[q, :] = solve_fem1d(mesh, a_func, lambda x: 0.0, f_func, bc_left, bc_right)


    from chaos_expansion import gpc_projection_coefficients
    gpc_coeffs = gpc_projection_coefficients(xi_q, w_q, u_q, index_set, "uniform")
    return mesh, gpc_coeffs, index_set, {'xi_q': xi_q, 'w_q': w_q, 'n_q': n_q}


def solve_stochastic_pde_galerkin(xL, xR, n_elem, a0_func, f_func, bc_left, bc_right,
                                  n_modes, max_poly_degree, sigma_kl=0.3, Lc_kl=0.1):
    d = n_modes
    mesh = uniform_mesh_1d(xL, xR, n_elem)
    n_nodes = len(mesh)
    index_set = enumerate_multi_indices_total_degree(d, max_poly_degree)
    N_poly = index_set.shape[0]


    lambdas, omegas = kl_eigenvalues_1d_exponential(sigma_kl, Lc_kl, xL, xR, n_modes)
    phi_kl = kl_eigenfunctions_1d(mesh, omegas, xL, xR)




    N_total = n_nodes * N_poly
    A_dense = np.zeros((N_total, N_total))
    b_dense = np.zeros(N_total)



    from quadrature_rules import gauss_legendre_tensor
    quad_order = max_poly_degree * 2 + 1
    xi_q, w_q_raw = gauss_legendre_tensor(d, quad_order)
    w_q = w_q_raw / (2.0 ** d)


    from chaos_expansion import gpc_basis_eval
    Psi = np.zeros((xi_q.shape[0], N_poly))
    for i, alpha in enumerate(index_set):
        Psi[:, i] = gpc_basis_eval(alpha, xi_q, "uniform")


    T = np.zeros((N_poly, N_poly, N_poly))
    for i in range(N_poly):
        for j in range(N_poly):
            for k in range(N_poly):
                T[i, j, k] = np.sum(w_q * Psi[:, i] * Psi[:, j] * Psi[:, k])


    from fem1d_solver import build_fem1d_system

    K0, _ = build_fem1d_system(mesh, a0_func, lambda x: 0.0, f_func, bc_left, bc_right)
    K0d = K0.to_dense()

    Kk = []
    for k in range(n_modes):
        a_k_func = lambda x, k=k: np.sqrt(max(lambdas[k], 0.0)) * np.interp(x, mesh, phi_kl[:, k])

        Kk_d = np.zeros((n_nodes, n_nodes))
        for e in range(n_elem):
            xL_e = mesh[e]
            xR_e = mesh[e + 1]
            h_e = xR_e - xL_e
            a_avg = 0.0
            for gg in range(3):
                xi_g = [-np.sqrt(3.0/5.0), 0.0, np.sqrt(3.0/5.0)][gg]
                w_g = [5.0/9.0, 8.0/9.0, 5.0/9.0][gg]
                x_g = 0.5*(xL_e+xR_e) + 0.5*h_e*xi_g
                a_avg += w_g * a_k_func(x_g)
            a_avg *= 0.5





            k_local = None
            idx = [e, e+1]
            for ii in range(2):
                for jj in range(2):
                    Kk_d[idx[ii], idx[jj]] += k_local[ii, jj]
        Kk.append(Kk_d)



    for i in range(N_poly):
        for j in range(N_poly):
            block = K0d.copy()
            for k in range(n_modes):
                block += Kk[k] * T[i, j, k + 1] if (k + 1) < N_poly else Kk[k] * 0.0

            for n1 in range(n_nodes):
                for n2 in range(n_nodes):
                    A_dense[i * n_nodes + n1, j * n_nodes + n2] += block[n1, n2]


    zero_alpha = np.zeros(d, dtype=int)
    idx0 = None
    for i, alpha in enumerate(index_set):
        if np.array_equal(alpha, zero_alpha):
            idx0 = i
            break


    f_vals = np.array([f_func(x) for x in mesh])

    for i, alpha in enumerate(index_set):
        coeff = 1.0 if np.array_equal(alpha, zero_alpha) else 0.0
        b_dense[i * n_nodes:(i + 1) * n_nodes] = coeff * f_vals



    for i in range(N_poly):
        row = i * n_nodes
        if bc_left[0] == 'D':
            A_dense[row, :] = 0.0
            A_dense[row, row] = 1.0
            b_dense[row] = bc_left[1]
        if bc_right[0] == 'D':
            row2 = i * n_nodes + n_nodes - 1
            A_dense[row2, :] = 0.0
            A_dense[row2, row2] = 1.0
            b_dense[row2] = bc_right[1]


    if N_total > 200:

        pass
    u_galerkin = np.linalg.solve(A_dense, b_dense)

    gpc_coeffs = u_galerkin.reshape(N_poly, n_nodes)
    return mesh, gpc_coeffs, index_set


def test_stochastic_pde_solver():
    xL, xR = 0.0, 1.0
    n_elem = 20
    a0 = lambda x: 1.0
    f = lambda x: np.sin(np.pi * x)

    mesh, coeffs, idx_set, _ = solve_stochastic_pde_projection(
        xL, xR, n_elem, a0, f, ('D', 0.0), ('D', 0.0),
        n_modes=2, max_poly_degree=2, quad_level=2)
    mean, var = gpc_mean_variance(coeffs, idx_set)

    u_det = solve_fem1d(mesh, a0, lambda x: 0.0, f, ('D', 0.0), ('D', 0.0))
    assert np.linalg.norm(mean - u_det) / np.linalg.norm(u_det) < 0.1
    assert var >= 0.0
    print("stochastic_pde_solver: all self-tests passed")


if __name__ == "__main__":
    test_stochastic_pde_solver()
