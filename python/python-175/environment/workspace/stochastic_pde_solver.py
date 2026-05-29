"""
stochastic_pde_solver.py
========================
Stochastic elliptic PDE solver combining 1-D FEM spatial discretization with
generalized polynomial chaos (gPC) expansion for uncertainty quantification.

Fused from seed projects:
- 1160_standing_wave_exact : exact analytical solutions for validation
- 546_house_data           : polygonal domain geometry
- 120_broyden              : quasi-Newton iteration for nonlinear stochastic systems

Mathematical foundation
-----------------------
Consider the 1-D stochastic elliptic boundary-value problem on D=[xL,xR]:
    -d/dx( a(x,xi) du/dx ) = f(x),   x in D
    u(xL) = uL,  u(xR) = uR

where the random diffusion coefficient has a Karhunen-Loeve expansion:
    a(x,xi) = a_0(x) + sum_{k=1}^{d} sqrt(lambda_k) phi_k(x) xi_k

with xi_k ~ U(-1,1) i.i.d.  The eigenpairs (lambda_k, phi_k) come from the
exponential covariance kernel:
    C(x,y) = sigma^2 exp( -|x-y| / L_c )
which has analytical KL eigenfunctions on [0,1] involving transcendental equations.

For simplicity in 1-D we use the analytical approximation:
    lambda_k = 2 sigma^2 L_c / (1 + (omega_k L_c)^2)
    phi_k(x) = sin(omega_k x) / sqrt(0.5 - sin(2 omega_k)/(4 omega_k))
where omega_k are positive roots of
    tan(omega) + 2 L_c omega = 0   (for Dirichlet-Dirichlet)

The gPC Galerkin formulation:
    For each multi-index beta in Lambda:
        sum_{alpha} K_{beta,alpha} u_alpha = f_beta
    where
        K_{beta,alpha} = E[ a(x,xi) Psi_beta(xi) Psi_alpha(xi) ]
        f_beta = f(x) delta_{beta,0}

This yields a block-sparse linear system of size N_x * |Lambda|.
The block structure can be solved by CG on the Kronecker-structured matrix
or by a Broyden quasi-Newton iteration if a nonlinear term is added.

For the 1-D deterministic solver at each realization xi, we use the FEM
module (fem1d_solver.py).  The gPC projection then computes the coefficients
by Smolyak sparse-grid quadrature.
"""

import numpy as np
from fem1d_solver import uniform_mesh_1d, solve_fem1d, fem1d_l2_error
from chaos_expansion import (gpc_projection_coefficients, gpc_mean_variance,
                              gpc_reconstruct, gpc_sobol_sensitivity,
                              gpc_total_order_sobol, enumerate_multi_indices_total_degree)
from quadrature_rules import smolyak_sparse_grid
from sparse_linear_solver import SparseMatrixCOO, conjugate_gradient_sparse


def kl_eigenvalues_1d_exponential(sigma, Lc, xL, xR, n_modes):
    """
    Approximate Karhunen-Loeve eigenvalues for the exponential covariance
    kernel C(x,y) = sigma^2 exp(-|x-y|/Lc) on [xL,xR] with Dirichlet BCs.

    Roots satisfy: tan(omega*L) + 2*Lc*omega = 0,  L = xR - xL.
    For simplicity we approximate omega_k ~ (k-0.5)*pi/L for large k.
    """
    L = xR - xL
    omegas = np.zeros(n_modes)
    # Solve approximately: for k=1, use bisection near (k-0.5)*pi/L
    for k in range(1, n_modes + 1):
        a = (k - 0.5) * np.pi / L
        b = k * np.pi / L
        # Function g(omega) = tan(omega*L) + 2*Lc*omega
        # We want the root in ((k-0.5)*pi/L, k*pi/L)
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
    """
    Evaluate KL eigenfunctions phi_k(x) for the exponential kernel on [xL,xR].
    Normalized such that int_{xL}^{xR} phi_k(x)^2 dx = 1.
    """
    x = np.asarray(x, dtype=float)
    L = xR - xL
    n_modes = len(omegas)
    phi = np.zeros((len(x), n_modes))
    for k, w in enumerate(omegas):
        # sin(w*(x-xL)) with normalization
        denom = np.sqrt(0.5 * L - np.sin(2.0 * w * L) / (4.0 * w))
        if abs(denom) < 1e-15:
            denom = np.sqrt(L / 2.0)
        phi[:, k] = np.sin(w * (x - xL)) / denom
    return phi


def diffusion_coefficient_kl(x, xi, a0, sigma, Lc, xL, xR):
    """
    Evaluate a(x,xi) = a0(x) + sum_k sqrt(lambda_k) phi_k(x) xi_k.
    Ensures positivity by clamping to a small positive value.
    """
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
    # Ensure strict positivity for ellipticity
    a_min = np.min(a)
    if a_min <= 0:
        a = a - a_min + 0.01
    return a


def solve_stochastic_pde_projection(xL, xR, n_elem, a0_func, f_func, bc_left, bc_right,
                                    n_modes, max_poly_degree, quad_level):
    """
    Solve the stochastic PDE by pseudo-spectral gPC projection with Smolyak
    sparse-grid quadrature.

    Returns
    -------
    mesh : ndarray
    gpc_coeffs : ndarray, shape (N_poly, n_nodes)
    index_set : ndarray, shape (N_poly, n_modes)
    quad_info : dict
    """
    d = n_modes
    mesh = uniform_mesh_1d(xL, xR, n_elem)
    n_nodes = len(mesh)

    # Generate multi-index set (total degree truncation)
    index_set = enumerate_multi_indices_total_degree(d, max_poly_degree)
    N_poly = index_set.shape[0]

    # Smolyak sparse-grid quadrature in d dimensions
    xi_q, w_q = smolyak_sparse_grid(d, quad_level)
    if xi_q.size == 0:
        # Fallback to small tensor grid
        from quadrature_rules import gauss_legendre_tensor
        xi_q, w_q = gauss_legendre_tensor(d, quad_level + 1)
    # Normalize weights for probability measure on [-1,1]^d
    w_q = w_q / (2.0 ** d)

    n_q = xi_q.shape[0]
    # Evaluate PDE solution at each quadrature point
    u_q = np.zeros((n_q, n_nodes))
    for q in range(n_q):
        xi = xi_q[q]
        # Diffusion coefficient at this realization
        def a_func(x):
            return diffusion_coefficient_kl(x, xi, a0_func, sigma=0.3, Lc=0.1, xL=xL, xR=xR)
        u_q[q, :] = solve_fem1d(mesh, a_func, lambda x: 0.0, f_func, bc_left, bc_right)

    # Project onto gPC basis
    from chaos_expansion import gpc_projection_coefficients
    gpc_coeffs = gpc_projection_coefficients(xi_q, w_q, u_q, index_set, "uniform")
    return mesh, gpc_coeffs, index_set, {'xi_q': xi_q, 'w_q': w_q, 'n_q': n_q}


def solve_stochastic_pde_galerkin(xL, xR, n_elem, a0_func, f_func, bc_left, bc_right,
                                  n_modes, max_poly_degree, sigma_kl=0.3, Lc_kl=0.1):
    """
    Solve the stochastic PDE by intrusive Galerkin projection.
    Assembles the global block matrix A_galerkin and solves it by CG.
    """
    d = n_modes
    mesh = uniform_mesh_1d(xL, xR, n_elem)
    n_nodes = len(mesh)
    index_set = enumerate_multi_indices_total_degree(d, max_poly_degree)
    N_poly = index_set.shape[0]

    # KL eigenpairs (precomputed)
    lambdas, omegas = kl_eigenvalues_1d_exponential(sigma_kl, Lc_kl, xL, xR, n_modes)
    phi_kl = kl_eigenfunctions_1d(mesh, omegas, xL, xR)

    # Build FEM matrices for each KL mode and mean
    # We need K0, K1, ..., Kd where K_k corresponds to sqrt(lambda_k) phi_k(x)
    # For simplicity we use dense assembly; large systems would use block-sparse CG.
    N_total = n_nodes * N_poly
    A_dense = np.zeros((N_total, N_total))
    b_dense = np.zeros(N_total)

    # Compute triple products E[Psi_alpha Psi_beta Psi_gamma]
    # For total degree P, use Gauss-Legendre tensor product of sufficient order
    from quadrature_rules import gauss_legendre_tensor
    quad_order = max_poly_degree * 2 + 1
    xi_q, w_q_raw = gauss_legendre_tensor(d, quad_order)
    w_q = w_q_raw / (2.0 ** d)

    # Precompute Psi matrix: shape (n_q, N_poly)
    from chaos_expansion import gpc_basis_eval
    Psi = np.zeros((xi_q.shape[0], N_poly))
    for i, alpha in enumerate(index_set):
        Psi[:, i] = gpc_basis_eval(alpha, xi_q, "uniform")

    # Triple product tensor T[i,j,k] = E[Psi_i Psi_j Psi_k]
    T = np.zeros((N_poly, N_poly, N_poly))
    for i in range(N_poly):
        for j in range(N_poly):
            for k in range(N_poly):
                T[i, j, k] = np.sum(w_q * Psi[:, i] * Psi[:, j] * Psi[:, k])

    # Assemble spatial FEM matrices
    from fem1d_solver import build_fem1d_system
    # Mean diffusion a0
    K0, _ = build_fem1d_system(mesh, a0_func, lambda x: 0.0, f_func, bc_left, bc_right)
    K0d = K0.to_dense()
    # KL mode matrices
    Kk = []
    for k in range(n_modes):
        a_k_func = lambda x, k=k: np.sqrt(max(lambdas[k], 0.0)) * np.interp(x, mesh, phi_kl[:, k])
        # Build element-wise
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
            # TODO: Hole 2 - Assemble local stiffness for KL mode k and add to global Kk_d
            # The local stiffness for element e with length h_e and average diffusion a_avg
            # must be computed and scattered into Kk_d using the same element mapping as
            # the deterministic FEM solver (see fem1d_solver.build_fem1d_system).
            # Ensure consistency with the local stiffness formula used there.
            k_local = None  # FIX: compute 2x2 local stiffness matrix
            idx = [e, e+1]
            for ii in range(2):
                for jj in range(2):
                    Kk_d[idx[ii], idx[jj]] += k_local[ii, jj]
        Kk.append(Kk_d)

    # Build global Galerkin matrix: A[(i,n1), (j,n2)] = sum_k Kk[n1,n2] * T[i,j,k_mode]
    # Here k_mode=0 corresponds to mean, k_mode=1..d to KL modes
    for i in range(N_poly):
        for j in range(N_poly):
            block = K0d.copy()
            for k in range(n_modes):
                block += Kk[k] * T[i, j, k + 1] if (k + 1) < N_poly else Kk[k] * 0.0
            # Place block
            for n1 in range(n_nodes):
                for n2 in range(n_nodes):
                    A_dense[i * n_nodes + n1, j * n_nodes + n2] += block[n1, n2]

    # RHS: only alpha=0 contributes (deterministic source)
    zero_alpha = np.zeros(d, dtype=int)
    idx0 = None
    for i, alpha in enumerate(index_set):
        if np.array_equal(alpha, zero_alpha):
            idx0 = i
            break

    # Evaluate source projection
    f_vals = np.array([f_func(x) for x in mesh])
    # RHS vector for each alpha: b_alpha[n] = f(x_n) * delta_{alpha,0}
    for i, alpha in enumerate(index_set):
        coeff = 1.0 if np.array_equal(alpha, zero_alpha) else 0.0
        b_dense[i * n_nodes:(i + 1) * n_nodes] = coeff * f_vals

    # Apply Dirichlet BCs to the global system
    # Left boundary (node 0) for all polynomial blocks
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

    # Solve
    if N_total > 200:
        # Use CG with block-sparse structure (not implemented here; fallback to dense)
        pass
    u_galerkin = np.linalg.solve(A_dense, b_dense)
    # Reshape to (N_poly, n_nodes)
    gpc_coeffs = u_galerkin.reshape(N_poly, n_nodes)
    return mesh, gpc_coeffs, index_set


def test_stochastic_pde_solver():
    """Self-test with deterministic mean comparison."""
    xL, xR = 0.0, 1.0
    n_elem = 20
    a0 = lambda x: 1.0
    f = lambda x: np.sin(np.pi * x)
    # Projection method
    mesh, coeffs, idx_set, _ = solve_stochastic_pde_projection(
        xL, xR, n_elem, a0, f, ('D', 0.0), ('D', 0.0),
        n_modes=2, max_poly_degree=2, quad_level=2)
    mean, var = gpc_mean_variance(coeffs, idx_set)
    # Mean should be close to deterministic solution
    u_det = solve_fem1d(mesh, a0, lambda x: 0.0, f, ('D', 0.0), ('D', 0.0))
    assert np.linalg.norm(mean - u_det) / np.linalg.norm(u_det) < 0.1
    assert var >= 0.0
    print("stochastic_pde_solver: all self-tests passed")


if __name__ == "__main__":
    test_stochastic_pde_solver()
