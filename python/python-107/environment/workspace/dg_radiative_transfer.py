"""
dg_radiative_transfer.py

1D Discontinuous Galerkin (DG) solver for the diffusion approximation of the
radiative transfer equation in layered tissue.

Adapted from dg1d_heat (Nodal Discontinuous Galerkin Methods by Hesthaven &
Warburton, Springer 2007). The original heat equation solver is reformulated
for the 1D steady-state diffusion equation:

    -D(z) d^2 phi/dz^2 + mu_a(z) phi = S(z)

with Robin boundary conditions:
    -D dphi/dz + 0.5 phi = 0   at z = z_min  (extrapolated boundary)
     D dphi/dz + 0.5 phi = 0   at z = z_max

This models photon fluence rate phi(z) in tissue under diffuse illumination.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Jacobi polynomial evaluation (from JacobiP, GradJacobiP)
# ---------------------------------------------------------------------------

def jacobi_p(x, alpha, beta, n):
    """
    Evaluate Jacobi polynomial P_n^{(alpha,beta)}(x) using recurrence.

    Three-term recurrence:
      P_0 = 1
      P_1 = 0.5*(alpha-beta) + 0.5*(alpha+beta+2)*x
      a_n P_{n+1} = (b_n + c_n x) P_n - d_n P_{n-1}

    Parameters
    ----------
    x : array_like
        Points at which to evaluate.
    alpha, beta : float
        Jacobi parameters.
    n : int
        Polynomial degree.

    Returns
    -------
    P : ndarray
        Polynomial values.
    """
    x = np.asarray(x, dtype=float)
    if n == 0:
        return np.ones_like(x)
    if n == 1:
        return 0.5 * (alpha - beta) + 0.5 * (alpha + beta + 2.0) * x

    # Recurrence
    pl = np.ones_like(x)
    p = 0.5 * (alpha - beta) + 0.5 * (alpha + beta + 2.0) * x
    for k in range(1, n):
        a1 = 2.0 * (k + 1.0) * (k + alpha + beta + 1.0) * (2.0 * k + alpha + beta)
        a2 = (2.0 * k + alpha + beta + 1.0) * (alpha * alpha - beta * beta)
        a3 = (2.0 * k + alpha + beta) * (2.0 * k + alpha + beta + 1.0) * (2.0 * k + alpha + beta + 2.0)
        a4 = 2.0 * (k + alpha) * (k + beta) * (2.0 * k + alpha + beta + 2.0)
        denom = a1
        if abs(denom) < 1e-14:
            denom = 1e-14
        p_new = ((a2 + a3 * x) * p - a4 * pl) / denom
        pl = p.copy()
        p = p_new
    return p


def grad_jacobi_p(x, alpha, beta, n):
    """
    Derivative of Jacobi polynomial.

    d/dx P_n^{(alpha,beta)}(x) = 0.5 * (alpha + beta + n + 1) * P_{n-1}^{(alpha+1,beta+1)}(x)

    Parameters
    ----------
    x : array_like
    alpha, beta : float
    n : int

    Returns
    -------
    dP : ndarray
    """
    x = np.asarray(x, dtype=float)
    if n == 0:
        return np.zeros_like(x)
    return 0.5 * (alpha + beta + n + 1.0) * jacobi_p(x, alpha + 1.0, beta + 1.0, n - 1)


# ---------------------------------------------------------------------------
# Gauss-Lobatto nodes (from JacobiGL)
# ---------------------------------------------------------------------------

def jacobi_gl_nodes(alpha, beta, n):
    """
    Compute n+1 Gauss-Lobatto nodes for Jacobi polynomial (alpha, beta).

    Nodes include endpoints x = -1 and x = 1.
    For Legendre: alpha = beta = 0.

    Parameters
    ----------
    alpha, beta : float
    n : int
        Polynomial order (n+1 nodes).

    Returns
    -------
    r : ndarray, shape (n+1,)
        Nodes.
    """
    if n == 0:
        return np.array([-1.0, 1.0])
    if n == 1:
        return np.array([-1.0, 0.0, 1.0])

    # Interior nodes are roots of P_{n-1}^{(alpha+1,beta+1)}
    # Use Newton-Raphson on Chebyshev-Gauss-Lobatto initial guess
    r = -np.cos(np.pi * np.arange(n + 1) / n)
    # Only refine interior points
    for i in range(1, n):
        x0 = r[i]
        for _ in range(50):
            p = jacobi_p(x0, alpha + 1.0, beta + 1.0, n - 1)
            dp = grad_jacobi_p(x0, alpha + 1.0, beta + 1.0, n - 1)
            if abs(dp) < 1e-14:
                break
            dx = p / dp
            x0 = x0 - dx
            if abs(dx) < 1e-14:
                break
        r[i] = x0
    r[0] = -1.0
    r[-1] = 1.0
    return r


# ---------------------------------------------------------------------------
# Vandermonde and differentiation matrices
# ---------------------------------------------------------------------------

def vandermonde_1d(n, r):
    """
    Build 1D Vandermonde matrix V_{ij} = P_j(r_i) for j=0..n.

    Parameters
    ----------
    n : int
        Polynomial order.
    r : ndarray
        Nodes.

    Returns
    -------
    V : ndarray
    """
    r = np.asarray(r, dtype=float)
    Np = len(r)
    V = np.zeros((Np, n + 1), dtype=float)
    for j in range(n + 1):
        V[:, j] = jacobi_p(r, 0.0, 0.0, j)
    return V


def d_matrix_1d(n, r, V):
    """
    Compute differentiation matrix D such that D V = V_x.

    D = V_x * V^{-1}

    Parameters
    ----------
    n : int
    r : ndarray
    V : ndarray
        Vandermonde matrix.

    Returns
    -------
    D : ndarray
    """
    r = np.asarray(r, dtype=float)
    Np = len(r)
    Vx = np.zeros((Np, n + 1), dtype=float)
    for j in range(n + 1):
        Vx[:, j] = grad_jacobi_p(r, 0.0, 0.0, j)
    D = np.linalg.solve(V.T, Vx.T).T
    return D


# ---------------------------------------------------------------------------
# DG element assembly for 1D diffusion equation
# ---------------------------------------------------------------------------

def dg_diffusion_solve_1d(z_min, z_max, n_elements, poly_order,
                          diffusivity_func, absorption_func, source_func,
                          robin_left=(0.5, 0.0), robin_right=(0.5, 0.0)):
    """
    Solve 1D steady-state diffusion equation with DG.

    Equation: -D(z) d^2 phi/dz^2 + mu_a(z) phi = S(z)
    Discretized on n_elements elements with polynomial order poly_order.

    Robin BC: a_L phi + b_L D dphi/dz = g_L at left
              a_R phi + b_R D dphi/dz = g_R at right

    Parameters
    ----------
    z_min, z_max : float
        Domain.
    n_elements : int
        Number of elements.
    poly_order : int
        Polynomial order per element.
    diffusivity_func : callable
        D(z).
    absorption_func : callable
        mu_a(z).
    source_func : callable
        S(z).
    robin_left, robin_right : tuple
        (a, g) with b fixed as -1 for left, +1 for right in standard form.

    Returns
    -------
    z_global : ndarray
        Global node coordinates.
    phi : ndarray
        Solution at global nodes.
    """
    if n_elements < 1 or poly_order < 1:
        raise ValueError("n_elements and poly_order must be >= 1.")

    # Reference element nodes and matrices
    r = jacobi_gl_nodes(0.0, 0.0, poly_order)
    Np = len(r)  # nodes per element
    V = vandermonde_1d(poly_order, r)
    Dr = d_matrix_1d(poly_order, r, V)
    # Mass matrix: integrate with GL quadrature (weights from GL nodes)
    # For Legendre GL, weights are 2/(N(N+1)) at endpoints, interior from JacobiGQ
    # Simplification: use approximate GL-Lobatto weights via inverse Vandermonde
    # M = inv(V) * inv(V)^T  (exact for polynomial up to degree 2N-1)
    invV = np.linalg.inv(V)
    M = np.dot(invV, invV.T)

    # Element coordinates
    va = np.linspace(z_min, z_max, n_elements + 1)[:-1]
    vb = np.linspace(z_min, z_max, n_elements + 1)[1:]

    total_dof = n_elements * Np
    A = np.zeros((total_dof, total_dof), dtype=float)
    b_vec = np.zeros(total_dof, dtype=float)
    z_global = np.zeros(total_dof, dtype=float)

    for k in range(n_elements):
        h = vb[k] - va[k]
        # Map reference nodes r in [-1,1] to physical nodes
        x_local = va[k] + 0.5 * (r + 1.0) * h
        # Jacobian: dx/dr = h/2
        J = h / 2.0
        # Local matrices
        D_local = Dr / J
        M_local = M * J

        # Evaluate coefficients at local nodes
        D_vals = np.array([diffusivity_func(xi) for xi in x_local])
        mu_vals = np.array([absorption_func(xi) for xi in x_local])
        S_vals = np.array([source_func(xi) for xi in x_local])

        # Stiffness: integrate D * dphi/dz * dpsi/dz + mu_a * phi * psi
        # Using nodal quadrature with LGL weights (diagonal of mass matrix)
        K_local = np.zeros((Np, Np), dtype=float)
        for q in range(Np):
            wq = M_local[q, q]
            for i in range(Np):
                for j in range(Np):
                    K_local[i, j] += D_vals[q] * D_local[q, i] * D_local[q, j] * wq
                    if i == j:
                        K_local[i, j] += mu_vals[q] * wq

        f_local = S_vals * np.diag(M_local)

        # Global assembly
        dof_start = k * Np
        A[dof_start:dof_start + Np, dof_start:dof_start + Np] += K_local
        b_vec[dof_start:dof_start + Np] += f_local
        z_global[dof_start:dof_start + Np] = x_local

    # DG interface fluxes: simplified interior penalty
    for k in range(n_elements - 1):
        dof_k_end = (k + 1) * Np - 1
        dof_kp1_start = (k + 1) * Np
        # Penalty for discontinuity
        dz = z_global[dof_kp1_start] - z_global[dof_k_end]
        if dz < 1e-14:
            dz = 1e-14
        penalty = 1.0 / dz
        A[dof_k_end, dof_k_end] += penalty
        A[dof_k_end, dof_kp1_start] -= penalty
        A[dof_kp1_start, dof_k_end] -= penalty
        A[dof_kp1_start, dof_kp1_start] += penalty

    # Boundary conditions
    # Left Robin: a_L phi - D dphi/dz = g_L
    aL, gL = robin_left
    D0 = diffusivity_func(z_global[0])
    A[0, 0] += aL
    b_vec[0] += gL

    # Right Robin: a_R phi + D dphi/dz = g_R
    aR, gR = robin_right
    Dn = diffusivity_func(z_global[-1])
    A[-1, -1] += aR
    b_vec[-1] += gR

    # Solve
    phi = np.linalg.solve(A, b_vec)
    return z_global, phi


# ---------------------------------------------------------------------------
# Wrapper for multi-layer tissue diffusion
# ---------------------------------------------------------------------------

def solve_tissue_diffusion_dg(layer_boundaries, layer_optical_props,
                              source_profile='uniform', poly_order=4, n_elements_per_layer=4):
    """
    Solve steady-state photon diffusion in multi-layer tissue using DG.

    Parameters
    ----------
    layer_boundaries : array_like
        z-coordinates of layer interfaces.
    layer_optical_props : list of dict
        Each dict has 'mu_a', 'mu_s', 'g', 'n'.
    source_profile : str
        'uniform' or 'gaussian'.
    poly_order : int
        DG polynomial order.
    n_elements_per_layer : int
        Elements per layer.

    Returns
    -------
    z : ndarray
        Solution coordinates.
    phi : ndarray
        Photon fluence rate.
    """
    boundaries = np.asarray(layer_boundaries, dtype=float)
    n_layers = len(boundaries) - 1

    def D_func(z):
        # TODO: Implement the layer-wise diffusion coefficient for DG solver.
        # Must be consistent with the diffusion_coefficient() definition in oct_physics.py.
        raise NotImplementedError("Hole 2: D_func in solve_tissue_diffusion_dg needs to be implemented.")

    def mu_a_func(z):
        for i in range(n_layers):
            if boundaries[i] <= z <= boundaries[i + 1]:
                return layer_optical_props[i]['mu_a']
        return 0.0

    def S_func(z):
        if source_profile == 'uniform':
            return 1.0 if boundaries[0] <= z <= boundaries[-1] else 0.0
        elif source_profile == 'gaussian':
            z0 = (boundaries[0] + boundaries[-1]) / 2.0
            sigma = (boundaries[-1] - boundaries[0]) / 4.0
            return np.exp(-0.5 * ((z - z0) / sigma) ** 2)
        else:
            return 1.0

    n_elements = n_layers * n_elements_per_layer
    z, phi = dg_diffusion_solve_1d(
        boundaries[0], boundaries[-1], n_elements, poly_order,
        D_func, mu_a_func, S_func,
        robin_left=(0.5, 1.0), robin_right=(0.5, 0.0)
    )
    return z, phi
