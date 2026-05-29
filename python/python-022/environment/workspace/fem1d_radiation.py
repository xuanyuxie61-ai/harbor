"""
1D FEM Radiation Diffusion Solver for ICF

Based on:
- fem1d_lagrange (Project 393): Lagrange basis FEM stiffness/mass assembly
- r83v (Project 967): Tridiagonal CG solver
- rcm (Project 1016): Matrix reordering

Solves the radiation diffusion equation in 1D spherical geometry:
  dE_r/dt = (1/r^2) * d/dr (r^2 * D_r * dE_r/dr) + c * kappa_P * (a_R T_g^4 - E_r)

where D_r = c / (3 * kappa_R) is the radiation diffusion coefficient.
"""

import numpy as np
from matrix_utils import r83v_cg, build_tridiagonal_from_fem
from quadrature_utils import gauss_legendre_rule

# Physical constants
C_LIGHT = 2.99792458e8
A_RAD = 7.5657e-16  # radiation constant [J/m^3/K^4]


def lagrange_value(x_nodes, x_eval):
    """
    Evaluate Lagrange basis polynomials L_i(x) such that L_i(x_j) = delta_ij.
    Based on lagrange_value from Project 393.
    """
    n_nodes = len(x_nodes)
    n_eval = len(x_eval)
    L = np.zeros((n_eval, n_nodes))

    for j in range(n_eval):
        x = x_eval[j]
        for i in range(n_nodes):
            li = 1.0
            for k in range(n_nodes):
                if k != i:
                    denom = x_nodes[i] - x_nodes[k]
                    if abs(denom) < 1e-30:
                        li = 0.0
                        break
                    li *= (x - x_nodes[k]) / denom
            L[j, i] = li

    return L


def lagrange_derivative(x_nodes, x_eval):
    """
    Evaluate derivatives of Lagrange basis polynomials.
    Based on lagrange_derivative from Project 393.
    """
    n_nodes = len(x_nodes)
    n_eval = len(x_eval)
    dL = np.zeros((n_eval, n_nodes))

    for j in range(n_eval):
        x = x_eval[j]
        for i in range(n_nodes):
            dli = 0.0
            for k in range(n_nodes):
                if k != i:
                    denom = x_nodes[i] - x_nodes[k]
                    if abs(denom) < 1e-30:
                        dli = 0.0
                        break
                    term = 1.0 / denom
                    for m in range(n_nodes):
                        if m != i and m != k:
                            denom2 = x_nodes[i] - x_nodes[m]
                            if abs(denom2) < 1e-30:
                                term = 0.0
                                break
                            term *= (x - x_nodes[m]) / denom2
                    dli += term
            dL[j, i] = dli

    return dL


def assemble_fem_matrices_spherical(x_nodes, q_num, kappa_R_func, T_gas_func):
    """
    Assemble stiffness and mass matrices for 1D spherical radiation diffusion.
    Based on fem1d_lagrange_stiffness from Project 393.
    """
    n_nodes = len(x_nodes)
    r_min = x_nodes[0]
    r_max = x_nodes[-1]

    if abs(r_max - r_min) < 1e-30:
        return np.eye(n_nodes), np.eye(n_nodes), np.zeros(n_nodes)

    # Gauss-Legendre quadrature on [r_min, r_max]
    q_x, q_w = gauss_legendre_rule(q_num, r_min, r_max)

    # Evaluate basis and derivatives
    L = lagrange_value(x_nodes, q_x)
    dL = lagrange_derivative(x_nodes, q_x)

    # Initialize matrices
    A = np.zeros((n_nodes, n_nodes))  # Stiffness (diffusion + coupling)
    M = np.zeros((n_nodes, n_nodes))  # Mass
    B = np.zeros(n_nodes)             # RHS source

    for qi in range(q_num):
        r = q_x[qi]
        w = q_w[qi]

        # Spherical geometric factor: 4*pi*r^2
        geom = 4.0 * np.pi * r**2

        # Diffusion coefficient: D_r = c / (3 * kappa_R)
        kappa_R = kappa_R_func(r)
        D_r = C_LIGHT / (3.0 * max(kappa_R, 1e-30))

        # Planck coupling coefficient
        T_g = T_gas_func(r)
        kappa_P = kappa_R  # Use Rosseland as Planck for simplicity
        coupling = C_LIGHT * kappa_P

        for i in range(n_nodes):
            li = L[qi, i]
            dli = dL[qi, i]
            for j_node in range(n_nodes):
                lj = L[qi, j_node]
                dlj = dL[qi, j_node]
                A[i, j_node] += w * geom * D_r * dli * dlj
                M[i, j_node] += w * geom * li * lj

            # Add coupling to diagonal of stiffness matrix
            A[i, i] += w * geom * coupling * li * li

            # Source term: c * kappa_P * a_R * T_g^4
            source = coupling * A_RAD * T_g**4
            B[i] += w * geom * li * source

    return A, M, B


def radiation_diffusion_step(E_r_old, x_nodes, dt, kappa_R_func, T_gas_func,
                              theta=0.5, q_num=8):
    """
    One time step of radiation diffusion using theta-method FEM.
    Solves: (M + theta*dt*A) E_r^{n+1} = (M - (1-theta)*dt*A) E_r^n + dt*B
    """
    n_nodes = len(x_nodes)

    if abs(x_nodes[-1] - x_nodes[0]) < 1e-30:
        return E_r_old.copy()

    A, M, B = assemble_fem_matrices_spherical(x_nodes, q_num, kappa_R_func, T_gas_func)

    # Build tridiagonal system for theta-method
    a_tri, b_tri, c_tri, rhs_mat = build_tridiagonal_from_fem(A, M, dt, theta)

    # RHS vector
    rhs = rhs_mat @ E_r_old + dt * B

    # Boundary conditions
    # Inner boundary (r=0): symmetry => zero gradient (natural in FEM)
    # Outer boundary: E_r = a_R T_outer^4 (Marshak approximate)
    T_outer = T_gas_func(x_nodes[-1])
    E_r_outer = A_RAD * T_outer**4
    rhs[-1] = E_r_outer
    b_tri[-1] = 1.0
    if n_nodes > 1:
        if len(a_tri) > 0:
            a_tri[-1] = 0.0
        if len(c_tri) > 0:
            c_tri[-1] = 0.0

    # Solve tridiagonal system with CG
    E_r_new = r83v_cg(n_nodes, a_tri, b_tri, c_tri, rhs, E_r_old, tol=1e-12)

    # Ensure non-negative
    E_r_new = np.maximum(E_r_new, 0.0)

    return E_r_new


def run_radiation_diffusion(x_nodes, E_r_init, T_gas_func, kappa_R_func,
                            t_end, dt, theta=0.5, q_num=8):
    """
    Run radiation diffusion to time t_end.
    """
    n_nodes = len(x_nodes)
    E_r = E_r_init.copy()
    t = 0.0
    n_steps = max(int(np.ceil(t_end / dt)), 1)
    dt_actual = t_end / n_steps

    history = [E_r.copy()]

    for _ in range(n_steps):
        E_r = radiation_diffusion_step(E_r, x_nodes, dt_actual, kappa_R_func,
                                       T_gas_func, theta, q_num)
        t += dt_actual
        history.append(E_r.copy())

    return E_r, history
