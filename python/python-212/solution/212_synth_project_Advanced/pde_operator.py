"""
pde_operator.py
===============
Discretisation of the Bloch-Torrey-type steady-state reaction-diffusion PDE
on the unit square Omega = [0,1]^2 with Dirichlet boundary conditions.

Discretisation
--------------
A uniform grid of (N+2) x (N+2) nodes is laid out with mesh width h = 1/(N+1).
Interior nodes are numbered in lexicographic order, giving N^2 unknowns.

The 5-point Laplacian stencil at (i,j) is

    (Delta_h y)_{ij} = ( y_{i-1,j} + y_{i+1,j} + y_{i,j-1} + y_{i,j+1} - 4 y_{ij} ) / h^2

The full PDE operator is

    A = -nu * Delta_h  +  diag(kappa_1)  +  kappa_2 * diag(y^2 / (1+y^2))

which is SPD for kappa_1 > 0, kappa_2 >= 0.

KKT role
--------
``A`` is the (1,1) block of the KKT saddle-point system.  Its transpose
``A^T`` appears in the adjoint equation  A^T p = M (y_d - y).
"""

from __future__ import annotations
import numpy as np
from physics_models import PhysicalParameters


# ---------------------------------------------------------------------------
def create_grid(N: int):
    """Return 1-D coordinate arrays for interior nodes.

    x_int, y_int : 1-D arrays of length N with values h, 2h, ..., N*h
    h            : mesh width  1/(N+1)
    XX, YY       : 2-D meshgrid of interior nodes (for convenience)
    """
    if N < 2:
        raise ValueError("pde_operator.create_grid: N must be >= 2")
    h = 1.0 / (N + 1)
    x_int = np.linspace(h, 1.0 - h, N)
    y_int = np.linspace(h, 1.0 - h, N)
    XX, YY = np.meshgrid(x_int, y_int, indexing="ij")
    return x_int, y_int, h, XX, YY


# ---------------------------------------------------------------------------
def laplacian_5pt(N: int, h: float) -> np.ndarray:
    """Assemble the N^2 x N^2 sparse (dense storage) 5-point Laplacian.

    Lexicographic ordering: node (i,j) -> index k = i*N + j,  0 <= i,j < N.

    The matrix is  -Delta_h  (positive definite), i.e.
        L_{k,k}   =  4 / h^2
        L_{k,k+-1} = -1 / h^2   (if |k - k'| == 1  and same row)
        L_{k,k+-N} = -1 / h^2   (if |k - k'| == N)
    """
    n2 = N * N
    L = np.zeros((n2, n2), dtype=np.float64)
    inv_h2 = 1.0 / (h * h)
    for i in range(N):
        for j in range(N):
            k = i * N + j
            L[k, k] = 4.0 * inv_h2
            if i > 0:
                L[k, (i - 1) * N + j] = -inv_h2
            if i < N - 1:
                L[k, (i + 1) * N + j] = -inv_h2
            if j > 0:
                L[k, i * N + (j - 1)] = -inv_h2
            if j < N - 1:
                L[k, i * N + (j + 1)] = -inv_h2
    return L


# ---------------------------------------------------------------------------
def build_pde_operator(
    N: int,
    params: PhysicalParameters,
    y_lin: np.ndarray | None = None,
) -> np.ndarray:
    """Assemble the full PDE operator  A = -nu*Delta_h + diag(dR/dy(y_lin)).

    Parameters
    ----------
    N        : number of interior nodes per dimension
    params   : physical parameters
    y_lin    : linearisation point (length N^2).  If None, the linear part
               kappa_1 * I is used (i.e. dR/dy evaluated at y = 0).

    Returns
    -------
    A : (N^2, N^2) SPD matrix
    """
    h = 1.0 / (N + 1)
    L = laplacian_5pt(N, h)
    A = params.nu * L  # -nu * Delta_h  (positive definite)

    n2 = N * N
    if y_lin is None:
        # Linear reaction  R'(0) = kappa_1
        A += params.kappa_1 * np.eye(n2)
    else:
        y_lin = np.asarray(y_lin, dtype=np.float64).ravel()
        if y_lin.size != n2:
            raise ValueError(
                f"build_pde_operator: y_lin has size {y_lin.size}, expected {n2}"
            )
        dRdy = params.reaction_derivative(y_lin)
        A += np.diag(dRdy)
    return A


# ---------------------------------------------------------------------------
def build_rhs(
    N: int,
    h: float,
    u: np.ndarray,
    params: PhysicalParameters,
    g_bc: np.ndarray | None = None,
) -> np.ndarray:
    """Assemble the RHS vector  f = u + f_bc  where f_bc encodes Dirichlet BCs.

    The Dirichlet BC contribution enters the Laplacian stencil at boundary-
    adjacent nodes: for node (0,j) with y_{-1,j} = g(0, y_j),

        (-Delta_h y)_{0,j} includes  g(0, y_j) / h^2.
    """
    n2 = N * N
    rhs = np.asarray(u, dtype=np.float64).ravel().copy()
    if rhs.size != n2:
        raise ValueError(f"build_rhs: u has size {rhs.size}, expected {n2}")

    if g_bc is not None:
        # g_bc is a callable  g_bc(x, y) -> value  OR a 4-tuple
        # (g_left, g_right, g_bottom, g_top) each scalar.
        inv_h2 = 1.0 / (h * h)
        if callable(g_bc):
            x_int, y_int, _, _, _ = create_grid(N)
            # Left boundary  i=0
            rhs[0 * N: 1 * N] += g_bc(0.0, y_int[:]) * inv_h2
            # Right boundary  i=N-1
            rhs[(N - 1) * N: N * N] += g_bc(1.0, y_int[:]) * inv_h2
            # Bottom boundary  j=0
            rhs[:N] += g_bc(x_int[:], 0.0) * inv_h2
            # Top boundary  j=N-1
            rhs[N * (N - 1): N * N] += g_bc(x_int[:], 1.0) * inv_h2
        else:
            gL, gR, gB, gT = [float(v) for v in g_bc]
            inv_h2 = 1.0 / (h * h)
            x_int, y_int, _, _, _ = create_grid(N)
            rhs[0 * N: 1 * N] += gL * inv_h2
            rhs[(N - 1) * N: N * N] += gR * inv_h2
            rhs[:N] += gB * inv_h2
            rhs[N * (N - 1): N * N] += gT * inv_h2
    return rhs


# ---------------------------------------------------------------------------
def solve_state(
    A: np.ndarray,
    u: np.ndarray,
    N: int,
    params: PhysicalParameters,
    g_bc=None,
) -> np.ndarray:
    """Solve the state equation  A y = u + f_bc.

    Uses a direct PLU solve from linear_algebra.
    """
    from linear_algebra import plu_solve

    h = 1.0 / (N + 1)
    rhs = build_rhs(N, h, u, params, g_bc=g_bc)
    y = plu_solve(A, rhs)
    return y


# ---------------------------------------------------------------------------
def solve_adjoint(
    A: np.ndarray,
    y: np.ndarray,
    y_d: np.ndarray,
    N: int,
    params: PhysicalParameters,
    mass_diag: np.ndarray | None = None,
) -> np.ndarray:
    """Solve the adjoint equation  A^T p = M (y_d - y).

    The adjoint operator is A^T (transpose of the state operator).
    With Dirichlet BCs on a uniform grid, A is symmetric, so A^T = A.
    """
    from linear_algebra import plu_solve

    n2 = N * N
    rhs = y_d.ravel() - y.ravel()
    if mass_diag is not None:
        rhs = rhs * mass_diag
    else:
        h = 1.0 / (N + 1)
        rhs = rhs * (h * h)
    p = plu_solve(A.T, rhs)
    return p


# ---------------------------------------------------------------------------
def lumped_mass_matrix(N: int) -> np.ndarray:
    """Return the diagonal of the lumped mass matrix  M = h^2 * I.

    For a uniform grid on [0,1]^2 with mesh width h = 1/(N+1), each
    interior node carries mass h^2.
    """
    h = 1.0 / (N + 1)
    n2 = N * N
    return np.full(n2, h * h, dtype=np.float64)
