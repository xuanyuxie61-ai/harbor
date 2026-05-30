
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve


def debye_length(epsilon_r, temperature, ionic_strength):

    epsilon_0 = 8.854187817e-12
    k_B = 1.380649e-23
    N_A = 6.02214076e23
    e_charge = 1.602176634e-19


    I_mol_m3 = ionic_strength * 1000.0

    lambda_D_m = np.sqrt(
        epsilon_0 * epsilon_r * k_B * temperature
        / (2.0 * N_A * e_charge ** 2 * I_mol_m3)
    )
    lambda_D_nm = lambda_D_m * 1e9
    return lambda_D_nm


def build_pb_jacobian_residual(n, h, rho, phi, kappa, boundary="neumann"):
    numnodes = n * n
    J = sparse.lil_matrix((numnodes, numnodes))
    residual = np.zeros(numnodes)

    h2 = h * h
    kappa2 = kappa * kappa

    for i in range(n):
        for j in range(n):
            k = i * n + j
            phi_k = phi[k]
            rho_k = rho[k]


            im1 = (i - 1) * n + j if i > 0 else k
            ip1 = (i + 1) * n + j if i < n - 1 else k
            jm1 = i * n + (j - 1) if j > 0 else k
            jp1 = i * n + (j + 1) if j < n - 1 else k


            laplace = 0.0
            if i > 0:
                laplace += phi[im1]
            else:
                laplace += phi_k
            if i < n - 1:
                laplace += phi[ip1]
            else:
                laplace += phi_k
            if j > 0:
                laplace += phi[jm1]
            else:
                laplace += phi_k
            if j < n - 1:
                laplace += phi[jp1]
            else:
                laplace += phi_k
            laplace -= 4.0 * phi_k



            residual[k] = -laplace / h2 + kappa2 * np.sinh(phi_k) + rho_k


            J[k, k] = 4.0 / h2 + kappa2 * np.cosh(phi_k)
            if i > 0:
                J[k, im1] = -1.0 / h2
            else:
                J[k, k] += -1.0 / h2
            if i < n - 1:
                J[k, ip1] = -1.0 / h2
            else:
                J[k, k] += -1.0 / h2
            if j > 0:
                J[k, jm1] = -1.0 / h2
            else:
                J[k, k] += -1.0 / h2
            if j < n - 1:
                J[k, jp1] = -1.0 / h2
            else:
                J[k, k] += -1.0 / h2

    return J.tocsr(), residual


def solve_nonlinear_pb(n, h, rho, kappa, tol=1e-8, max_iter=50):
    phi = np.zeros(n * n)
    for it in range(max_iter):
        J, res = build_pb_jacobian_residual(n, h, rho, phi, kappa)
        norm_res = np.linalg.norm(res)
        if norm_res < tol:
            return phi, True, it
        try:
            delta = spsolve(J, -res)
        except Exception:

            delta = np.linalg.lstsq(J.toarray(), -res, rcond=None)[0]
        phi += delta
        if np.linalg.norm(delta) < tol:
            return phi, True, it
    return phi, False, max_iter


def electrostatic_free_energy(phi, rho, nodes, elements):
    from tet_mesh_core import integrate_over_tet_mesh
    integrand = rho * phi
    G_el, _ = integrate_over_tet_mesh(nodes, elements, integrand)
    G_el *= 0.5
    return G_el


def setup_dna_charge_density(n, h, dna_x_range, dna_y_range, charge_per_unit=-1.0):
    rho = np.zeros(n * n)
    x_min, x_max = dna_x_range
    y_min, y_max = dna_y_range
    for i in range(n):
        for j in range(n):
            if x_min <= j <= x_max and y_min <= i <= y_max:
                k = i * n + j
                rho[k] = charge_per_unit
    return rho
