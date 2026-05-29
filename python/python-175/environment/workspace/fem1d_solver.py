"""
fem1d_solver.py
===============
One-dimensional finite element solver with piecewise-linear hat functions,
used as the spatial discretization engine for the stochastic PDE solver.

Fused from seed project:
- 397_fem1d_project : Galerkin projection, hat functions, Gauss-Legendre assembly

Mathematical foundation
-----------------------
Consider the deterministic 1-D elliptic boundary-value problem on [xL, xR]:
    -d/dx( a(x) du/dx ) + c(x) u = f(x),   u(xL)=uL, u(xR)=uR

Discretize into n_elem elements with n_nodes = n_elem + 1 nodes.
Piecewise-linear hat function phi_i satisfies phi_i(x_j) = delta_{ij}.

On element e = [x_e, x_{e+1}] with length h_e:
    phi_e^L(x) = (x_{e+1} - x) / h_e
    phi_e^R(x) = (x - x_e) / h_e

Local stiffness matrix (diffusion term):
    K_e^{LL} = K_e^{RR} =  a_avg / h_e
    K_e^{LR} = K_e^{RL} = -a_avg / h_e
where a_avg is the average of a(x) on the element.

Local mass matrix (reaction term):
    M_e^{LL} = M_e^{RR} = h_e / 3
    M_e^{LR} = M_e^{RL} = h_e / 6

Local load vector (source term, using 2-point Gauss-Legendre quadrature):
    x_g1 = x_e + h_e*(0.5 - sqrt(3)/6),  w1 = h_e/2
    x_g2 = x_e + h_e*(0.5 + sqrt(3)/6),  w2 = h_e/2
    b_e^L = w1*f(x_g1)*phi_e^L(x_g1) + w2*f(x_g2)*phi_e^R(x_g2)
    b_e^R = w1*f(x_g1)*phi_e^R(x_g1) + w2*f(x_g2)*phi_e^R(x_g2)

The global system A u = b is assembled by summing element contributions,
then Dirichlet boundary conditions are applied by modifying the first and last rows.
"""

import numpy as np
from sparse_linear_solver import SparseMatrixCOO, conjugate_gradient_sparse


def uniform_mesh_1d(xL, xR, n_elem):
    """Generate a uniform 1-D mesh."""
    return np.linspace(xL, xR, n_elem + 1)


def build_fem1d_system(mesh, a_func, c_func, f_func, bc_left, bc_right):
    """
    Assemble the 1-D FEM stiffness matrix and load vector.

    Parameters
    ----------
    mesh : ndarray, shape (n_nodes,)
        Node coordinates.
    a_func : callable
        Diffusion coefficient a(x).
    c_func : callable
        Reaction coefficient c(x) (can be zero).
    f_func : callable
        Source term f(x).
    bc_left : tuple (type, value)
        ('D', uL) for Dirichlet, ('N', qL) for Neumann flux at left.
    bc_right : tuple (type, value)
        ('D', uR) for Dirichlet, ('N', qR) for Neumann flux at right.

    Returns
    -------
    A : SparseMatrixCOO
    b : ndarray
    """
    n_nodes = len(mesh)
    n_elem = n_nodes - 1
    A = SparseMatrixCOO(n_nodes, n_nodes)
    b = np.zeros(n_nodes)

    sqrt3 = np.sqrt(3.0)
    gauss_xi = np.array([-1.0 / sqrt3, 1.0 / sqrt3])
    gauss_w = np.array([1.0, 1.0])

    for e in range(n_elem):
        xL_e = mesh[e]
        xR_e = mesh[e + 1]
        h_e = xR_e - xL_e
        if h_e <= 0:
            raise ValueError("Mesh must be strictly increasing")

        # Average diffusion coefficient on element
        a_avg = 0.0
        for xi, w in zip(gauss_xi, gauss_w):
            x_g = 0.5 * (xL_e + xR_e) + 0.5 * h_e * xi
            a_avg += w * a_func(x_g)
        a_avg *= 0.5

        # TODO: Hole 1 - Implement local stiffness and mass matrices for 1-D linear FEM element
        # The element has length h_e and average diffusion a_avg.
        # For piecewise-linear hat functions, derive:
        #   k_local (2x2 diffusion stiffness) and m_local (2x2 reaction mass).
        # Hint: K_e^{LL} = K_e^{RR} = a_avg / h_e,  K_e^{LR} = K_e^{RL} = -a_avg / h_e
        #       M_e^{LL} = M_e^{RR} = h_e / 3,      M_e^{LR} = M_e^{RL} = h_e / 6
        k_local = None  # FIX: compute 2x2 local stiffness matrix
        m_local = None  # FIX: compute 2x2 local mass matrix

        # Local load
        f_local = np.zeros(2)
        for xi, w in zip(gauss_xi, gauss_w):
            x_g = 0.5 * (xL_e + xR_e) + 0.5 * h_e * xi
            phiL = 0.5 * (1.0 - xi)
            phiR = 0.5 * (1.0 + xi)
            f_val = f_func(x_g)
            f_local[0] += w * f_val * phiL * 0.5 * h_e
            f_local[1] += w * f_val * phiR * 0.5 * h_e

        # Reaction contribution to load
        c_local = np.zeros(2)
        for xi, w in zip(gauss_xi, gauss_w):
            x_g = 0.5 * (xL_e + xR_e) + 0.5 * h_e * xi
            phiL = 0.5 * (1.0 - xi)
            phiR = 0.5 * (1.0 + xi)
            c_val = c_func(x_g)
            c_local[0] += w * c_val * phiL * 0.5 * h_e
            c_local[1] += w * c_val * phiR * 0.5 * h_e

        # Assembly
        idx = [e, e + 1]
        for i in range(2):
            for j in range(2):
                A.add_entry(idx[i], idx[j], k_local[i, j] + m_local[i, j])
            b[idx[i]] += f_local[i]

    # Boundary conditions
    if bc_left[0] == 'D':
        uL = bc_left[1]
        for j in range(n_nodes):
            val = A.to_dense()[0, j] if n_nodes <= 100 else 0.0
        # Replace first row
        # Simplification: clear first row and set A[0,0]=1, b[0]=uL
        # Since SparseMatrixCOO doesn't support deletion, we rebuild for small N
        # or we handle it after converting to dense for small systems.
        pass  # handled below
    if bc_right[0] == 'D':
        pass  # handled below

    # Apply Dirichlet BCs by modifying rows (for small systems convert to dense)
    if n_nodes <= 500:
        Ad = A.to_dense()
        if bc_left[0] == 'D':
            Ad[0, :] = 0.0
            Ad[0, 0] = 1.0
            b[0] = bc_left[1]
        if bc_right[0] == 'D':
            Ad[-1, :] = 0.0
            Ad[-1, -1] = 1.0
            b[-1] = bc_right[1]
        if bc_left[0] == 'N':
            b[0] -= bc_left[1]  # incoming flux
        if bc_right[0] == 'N':
            b[-1] += bc_right[1]
        A = SparseMatrixCOO(n_nodes, n_nodes)
        for i in range(n_nodes):
            for j in range(n_nodes):
                if abs(Ad[i, j]) > 1e-15:
                    A.add_entry(i, j, Ad[i, j])
    else:
        # For large systems, use penalty method or fixed-point iteration
        # Not needed for the demo scale
        pass

    return A, b


def solve_fem1d(mesh, a_func, c_func, f_func, bc_left, bc_right, use_cg=False):
    """
    Solve the 1-D elliptic problem and return nodal solution values.
    """
    A, b = build_fem1d_system(mesh, a_func, c_func, f_func, bc_left, bc_right)
    n_nodes = len(mesh)
    if use_cg and n_nodes > 200:
        x0 = np.zeros(n_nodes)
        u, info = conjugate_gradient_sparse(A, b, x0=x0, tol=1e-12)
        if not info['converged']:
            # fallback to dense
            u = np.linalg.solve(A.to_dense(), b)
    else:
        u = np.linalg.solve(A.to_dense(), b)
    return u


def fem1d_l2_error(mesh, u_numeric, u_exact_func):
    """
    Compute L2 error ||u_exact - u_numeric||_{L^2} using 3-point Gauss quadrature
    on each element.
    """
    n_nodes = len(mesh)
    error_sq = 0.0
    sqrt35 = np.sqrt(3.0 / 5.0)
    xi3 = np.array([-sqrt35, 0.0, sqrt35])
    w3 = np.array([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0])
    for e in range(n_nodes - 1):
        xL_e = mesh[e]
        xR_e = mesh[e + 1]
        h_e = xR_e - xL_e
        uL = u_numeric[e]
        uR = u_numeric[e + 1]
        for xi, w in zip(xi3, w3):
            x_g = 0.5 * (xL_e + xR_e) + 0.5 * h_e * xi
            phiL = 0.5 * (1.0 - xi)
            phiR = 0.5 * (1.0 + xi)
            u_h = uL * phiL + uR * phiR
            diff = u_exact_func(x_g) - u_h
            error_sq += w * diff ** 2 * 0.5 * h_e
    return np.sqrt(error_sq)


def test_fem1d_solver():
    """Self-test: -u'' = sin(pi x), u(0)=u(1)=0, exact u = sin(pi x)/pi^2."""
    xL, xR = 0.0, 1.0
    n_elem = 40
    mesh = uniform_mesh_1d(xL, xR, n_elem)
    a_func = lambda x: 1.0
    c_func = lambda x: 0.0
    f_func = lambda x: np.sin(np.pi * x)
    u_num = solve_fem1d(mesh, a_func, c_func, f_func, ('D', 0.0), ('D', 0.0))
    u_exact = lambda x: np.sin(np.pi * x) / (np.pi ** 2)
    err = fem1d_l2_error(mesh, u_num, u_exact)
    assert err < 1e-3, f"FEM L2 error too large: {err}"
    print("fem1d_solver: all self-tests passed")


if __name__ == "__main__":
    test_fem1d_solver()
