"""
electrostatics.py
Nonlinear Poisson-Boltzmann solver for DNA-protein electrostatics.

Derived from: 871_plasma_matrix

Solves the nonlinear Poisson-Boltzmann equation in 2D:
    -nabla^2 phi + kappa^2 * sinh(phi) = -4*pi*rho / epsilon

where phi = e*psi/(k_B*T) is the dimensionless potential,
kappa is the inverse Debye length, rho is the charge density.

For DNA damage repair, electrostatic steering is the dominant
mechanism by which positively charged repair proteins are guided
to negatively charged DNA damage sites.

Key formulas:
  - Debye length: lambda_D = sqrt(epsilon * k_B * T / (2 * N_A * e^2 * I))
  - Electrostatic free energy: G_el = 1/2 * int rho(r) * phi(r) d^3r
  - Discretized Jacobian for Newton iteration:
      J_{ii} = -4/h^2 - kappa^2 * cosh(phi_i)
      J_{i,i+1} = J_{i,i-1} = J_{i,i+n} = J_{i+n,i} = 1/h^2
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve


def debye_length(epsilon_r, temperature, ionic_strength):
    """
    Compute the Debye screening length lambda_D (nm).

    lambda_D = sqrt(epsilon_0 * epsilon_r * k_B * T / (2 * N_A * e^2 * I))

    Parameters
    ----------
    epsilon_r : float
        Relative permittivity of solvent (~78 for water at 298K).
    temperature : float
        Temperature in Kelvin.
    ionic_strength : float
        Ionic strength in mol/L.

    Returns
    -------
    lambda_D : float
        Debye length in nanometers.
    """
    # Physical constants in SI
    epsilon_0 = 8.854187817e-12  # F/m
    k_B = 1.380649e-23  # J/K
    N_A = 6.02214076e23  # mol^-1
    e_charge = 1.602176634e-19  # C

    # Convert ionic strength to mol/m^3
    I_mol_m3 = ionic_strength * 1000.0

    lambda_D_m = np.sqrt(
        epsilon_0 * epsilon_r * k_B * temperature
        / (2.0 * N_A * e_charge ** 2 * I_mol_m3)
    )
    lambda_D_nm = lambda_D_m * 1e9
    return lambda_D_nm


def build_pb_jacobian_residual(n, h, rho, phi, kappa, boundary="neumann"):
    """
    Build the Jacobian and residual for the nonlinear Poisson-Boltzmann
    equation on a uniform n x n grid with spacing h.

    Parameters
    ----------
    n : int
        Grid points per dimension.
    h : float
        Grid spacing (nm).
    rho : ndarray, shape (n*n,)
        Charge density at each node.
    phi : ndarray, shape (n*n,)
        Current guess for potential.
    kappa : float
        Inverse Debye length (1/nm).
    boundary : str
        'neumann' or 'dirichlet'.

    Returns
    -------
    J : scipy.sparse.csr_matrix
        Jacobian matrix.
    residual : ndarray, shape (n*n,)
        Residual vector F(phi).
    """
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

            # Neumann boundary: ghost points equal to interior
            im1 = (i - 1) * n + j if i > 0 else k
            ip1 = (i + 1) * n + j if i < n - 1 else k
            jm1 = i * n + (j - 1) if j > 0 else k
            jp1 = i * n + (j + 1) if j < n - 1 else k

            # Finite-difference Laplacian with boundary handling
            laplace = 0.0
            if i > 0:
                laplace += phi[im1]
            else:
                laplace += phi_k  # Neumann: dphi/dn = 0
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

            # Nonlinear PB residual: -nabla^2 phi + kappa^2 * sinh(phi) + 4*pi*rho/eps
            # We set eps = 1 in reduced units, 4*pi factor absorbed
            residual[k] = -laplace / h2 + kappa2 * np.sinh(phi_k) + rho_k

            # Jacobian entries
            J[k, k] = 4.0 / h2 + kappa2 * np.cosh(phi_k)
            if i > 0:
                J[k, im1] = -1.0 / h2
            else:
                J[k, k] += -1.0 / h2  # Neumann correction
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
    """
    Solve the nonlinear Poisson-Boltzmann equation using Newton's method.

    Parameters
    ----------
    n : int
        Grid dimension.
    h : float
        Grid spacing.
    rho : ndarray, shape (n*n,)
        Charge density.
    kappa : float
        Inverse Debye length.
    tol : float
        Convergence tolerance for residual norm.
    max_iter : int

    Returns
    -------
    phi : ndarray, shape (n*n,)
        Converged potential.
    converged : bool
    iterations : int
    """
    phi = np.zeros(n * n)
    for it in range(max_iter):
        J, res = build_pb_jacobian_residual(n, h, rho, phi, kappa)
        norm_res = np.linalg.norm(res)
        if norm_res < tol:
            return phi, True, it
        try:
            delta = spsolve(J, -res)
        except Exception:
            # Fallback to pseudo-inverse for singular cases
            delta = np.linalg.lstsq(J.toarray(), -res, rcond=None)[0]
        phi += delta
        if np.linalg.norm(delta) < tol:
            return phi, True, it
    return phi, False, max_iter


def electrostatic_free_energy(phi, rho, nodes, elements):
    """
    Compute the electrostatic free energy from the PB solution:

        G_el = 1/2 * integral_Omega rho(r) * phi(r) dV

    Parameters
    ----------
    phi : ndarray, shape (N,)
        Potential at nodes.
    rho : ndarray, shape (N,)
        Charge density at nodes.
    nodes : ndarray, shape (N, 3)
    elements : ndarray, shape (M, 4)

    Returns
    -------
    G_el : float
    """
    from tet_mesh_core import integrate_over_tet_mesh
    integrand = rho * phi
    G_el, _ = integrate_over_tet_mesh(nodes, elements, integrand)
    G_el *= 0.5
    return G_el


def setup_dna_charge_density(n, h, dna_x_range, dna_y_range, charge_per_unit=-1.0):
    """
    Set up a synthetic charge density representing negatively charged DNA
    located in a rectangular region of the simulation domain.

    Parameters
    ----------
    n : int
        Grid points per dimension.
    h : float
        Grid spacing.
    dna_x_range : tuple
        (x_min, x_max) in grid index space.
    dna_y_range : tuple
        (y_min, y_max) in grid index space.
    charge_per_unit : float
        Charge density magnitude.

    Returns
    -------
    rho : ndarray, shape (n*n,)
    """
    rho = np.zeros(n * n)
    x_min, x_max = dna_x_range
    y_min, y_max = dna_y_range
    for i in range(n):
        for j in range(n):
            if x_min <= j <= x_max and y_min <= i <= y_max:
                k = i * n + j
                rho[k] = charge_per_unit
    return rho
