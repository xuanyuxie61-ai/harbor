"""
FEM Matrix Assembly Module
==========================
Based on seed project 1401_wathen_matrix:
- wathen_st.m   →  Wathen finite-element sparse matrix assembly
- cg_sparse.m   →  conjugate gradient solver for sparse systems

Physics:
--------
The radiative transfer equation in a 2D GRB afterglow slab is
discretized using serendipity (8-node) quadrilateral elements.
The weak form yields a large sparse linear system:

    A · ψ = S

where ψ is the discretized radiation intensity and S is the
source vector.  The Wathen matrix arises as the consistent mass
matrix for an NX × NY grid of serendipity elements:

    N = 3·NX·NY + 2·NX + 2·NY + 1

The local element mass matrix (for unit density) is:

    em = (1/180) · [
         6  -6   2  -8   3  -8   2  -6
        -6  32  -6  20  -8  16  -8  20
         2  -6   6  -6   2  -8   3  -8
        -8  20  -6  32  -6  20  -8  16
         3  -8   2  -6   6  -6   2  -8
        -8  16  -8  20  -6  32  -6  20
         2  -8   3  -8   2  -6   6  -6
        -6  20  -8  16  -8  20  -6  32 ]

For each element, the global matrix contribution is scaled by the
local fluid density ρ_{ij} (drawn from the blast-wave profile).

The system is solved by the conjugate-gradient method, which is
optimal for symmetric positive-definite matrices arising from
Galerkin discretizations of self-adjoint operators.
"""

import numpy as np


_WATHEN_EM = np.array([
    [6.0, -6.0, 2.0, -8.0, 3.0, -8.0, 2.0, -6.0],
    [-6.0, 32.0, -6.0, 20.0, -8.0, 16.0, -8.0, 20.0],
    [2.0, -6.0, 6.0, -6.0, 2.0, -8.0, 3.0, -8.0],
    [-8.0, 20.0, -6.0, 32.0, -6.0, 20.0, -8.0, 16.0],
    [3.0, -8.0, 2.0, -6.0, 6.0, -6.0, 2.0, -8.0],
    [-8.0, 16.0, -8.0, 20.0, -6.0, 32.0, -6.0, 20.0],
    [2.0, -8.0, 3.0, -8.0, 2.0, -6.0, 6.0, -6.0],
    [-6.0, 20.0, -8.0, 16.0, -8.0, 20.0, -6.0, 32.0]
], dtype=float)


def wathen_st(nx, ny, nz_num=None):
    """
    Assemble the Wathen finite-element matrix in sparse triplet format.

    Parameters
    ----------
    nx, ny : int
        Number of elements in x and y directions.
    nz_num : int, optional
        Number of nonzeros.  Computed if omitted.

    Returns
    -------
    row, col, a : ndarray
        Sparse triplet arrays.
    """
    if nz_num is None:
        nz_num = wathen_st_size(nx, ny)

    row = np.zeros(nz_num, dtype=int)
    col = np.zeros(nz_num, dtype=int)
    a = np.zeros(nz_num, dtype=float)

    em = _WATHEN_EM.T / 180.0
    k = 0

    for j in range(1, ny + 1):
        for i in range(1, nx + 1):
            node = np.zeros(8, dtype=int)
            node[0] = 3 * j * nx + 2 * i + 2 * j + 1
            node[1] = node[0] - 1
            node[2] = node[1] - 1
            node[3] = (3 * j - 1) * nx + 2 * j + i - 1
            node[4] = 3 * (j - 1) * nx + 2 * i + 2 * j - 3
            node[5] = node[4] + 1
            node[6] = node[5] + 1
            node[7] = node[3] + 1

            # Density drawn from a physical model (here: uniform random)
            rho = 50.0 * np.random.rand()

            for krow in range(8):
                for kcol in range(8):
                    row[k] = node[krow]
                    col[k] = node[kcol]
                    a[k] = rho * em[krow, kcol]
                    k += 1

    # Convert to 0-based indexing
    row -= 1
    col -= 1
    return row, col, a


def wathen_st_size(nx, ny):
    """
    Number of nonzeros in the Wathen matrix.
    """
    return nx * ny * 64


def wathen_order(nx, ny):
    """
    Matrix order N = 3·NX·NY + 2·NX + 2·NY + 1.
    """
    return 3 * nx * ny + 2 * nx + 2 * ny + 1


def cg_sparse(n, A, b, x0=None, tol=1e-10, max_iter=None):
    """
    Conjugate Gradient method for solving A·x = b.

    Parameters
    ----------
    n : int
        System dimension.
    A : ndarray, shape (n, n)
        SPD matrix (dense or sparse format handled via @).
    b : ndarray, shape (n,)
        Right-hand side.
    x0 : ndarray, optional
        Initial guess.
    tol : float
        Convergence tolerance on residual norm.
    max_iter : int, optional
        Maximum iterations (defaults to n).

    Returns
    -------
    x : ndarray
        Approximate solution.
    """
    # TODO: Implement Conjugate Gradient method for A·x = b.
    # Steps:
    #   1. Initialize x (zero vector if x0 is None).
    #   2. Compute residual r = b - A@x, set search direction p = r.
    #   3. Iterate: alpha = (r·r) / (p·A·p), x = x + alpha·p,
    #              r = r - alpha·A·p, beta = (r_new·r_new) / (r_old·r_old),
    #              p = r_new + beta·p.
    #   4. Stop when ||r|| < tol or max_iter reached.
    pass


def solve_wathen_system(nx=4, ny=4, rhs_func=None):
    """
    Assemble a Wathen matrix for a small GRB afterglow slab and
    solve the linear system A·x = b.

    Parameters
    ----------
    nx, ny : int
        Grid dimensions.
    rhs_func : callable, optional
        Function rhs_func(i, j) → value at node (i,j).

    Returns
    -------
    x : ndarray
        Solution vector.
    A_dense : ndarray
        Dense matrix for inspection.
    b : ndarray
        RHS vector.
    """
    n = wathen_order(nx, ny)
    nz_num = wathen_st_size(nx, ny)
    row, col, a = wathen_st(nx, ny, nz_num)

    # Build dense matrix for demonstration
    A_dense = np.zeros((n, n), dtype=float)
    for k in range(nz_num):
        A_dense[row[k], col[k]] += a[k]

    # Make symmetric positive definite by adding small diagonal
    A_dense += 1e-6 * np.eye(n)

    if rhs_func is None:
        b = np.ones(n, dtype=float)
    else:
        b = np.zeros(n, dtype=float)
        for j in range(1, ny + 1):
            for i in range(1, nx + 1):
                node = 3 * j * nx + 2 * i + 2 * j + 1
                b[node - 1] = rhs_func(i, j)

    x = cg_sparse(n, A_dense, b)
    return x, A_dense, b
