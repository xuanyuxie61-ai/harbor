"""
high_order_fd.py
================
High-order (2p-th order) centered finite-difference operators for the
1D and 2D Poisson and drift-diffusion equations governing defect states
in perovskite solar cells.

The central object is the 2p-point Laplacian stencil:
    d^2 u / dx^2 |_i = (1/h^2) sum_{k=-p}^{p} c_k u_{i+k} + O(h^{2p})

with coefficients c_k determined by Taylor matching:
    c_0 = -2 sum_{k=1}^{p} 1/k^2  * (-1)^{k+1} * (p!)^2 / ((p+k)!(p-k)!)
    c_k = c_{-k} = (-1)^{k+1} * 2 (p!)^2 / ((p+k)!(p-k)! k^2),   k >= 1

This module provides:
    - stencil construction and validation
    - von Neumann spectral analysis (amplification factor G(theta))
    - 1D Poisson solver with Dirichlet / mixed BCs
    - 2D Poisson solver (tensor-product Laplacian)
    - Richardson extrapolation for convergence-rate estimation
    - adaptive grid refinement via CV (control variate) monitoring

The module is inspired by fd1d_display (359) for the 1D representation of
finite-difference data and quadrature_weights_vandermonde_2d (951) for
constructing high-order integration/derivative weights via Vandermonde
systems. Here we extend the Vandermonde approach to build derivative
weights at arbitrary order on non-uniform defect-mesh spacings.

Mathematical background
-----------------------
The Poisson equation for the electrostatic potential phi(x) in a perovskite
slab with defect charge rho_def(x):
    -eps_r eps_0 d^2 phi/dx^2 = q [p(x) - n(x) + N_D^+(x) - N_A^-(x)]

where n, p are electron and hole densities; N_D^+, N_A^- are ionized donor
and acceptor densities (shallow defects). The high-order FD discretization
gives the banded linear system A phi = b where A is the p-banded Laplacian.

Stability of explicit time stepping (for the transient drift-diffusion):
    dt <= S_p * h^2 / D_max
where S_p = 1 / sum_{k=1}^{p} [2 (p!)^2/((p+k)!(p-k)!k^2)]
and D_max is the largest diffusion coefficient in the system.
"""

from __future__ import annotations
import math
from typing import Tuple, List, Optional

import numpy as np
from numpy.typing import NDArray

from perovskite_constants import (
    FD_ORDER, FD_HALO, DEFAULT_NX, DEVICE_LENGTH_M, EPS_PERP,
    E_CHARGE, V_NET, fd_stability_factor, THERMAL_VOLTAGE
)


# ============================================================================
# Stencil coefficients c_k for the 2p-th order centered second derivative
# ============================================================================
def build_laplacian_stencil(p: int) -> NDArray:
    """Return the 2p+1 coefficients [c_{-p}, ..., c_0, ..., c_p] for the
    centered 2p-th order finite-difference approximation of d^2/dx^2.

    Derivation
    ----------
    Solve the linear system V c = e_2 where V_{j,k} = k^j / j! for j = 0..2p
    and k = -p..p, with e_2 having a 1 in the j=2 position. Equivalently,
    use the closed-form expression (Fornberg 1988):
        c_k = (-1)^{k+1} * 2 (p!)^2 / ((p+k)!(p-k)! k^2),   1 <= k <= p
        c_0 = -2 * sum_{k=1}^{p} 1/k^2 * (p!)^2/((p+k)!(p-k)!) * (-1)^{k+1}
    but we use a Vandermonde solve for robustness.
    """
    if p < 1:
        raise ValueError("Stencil order p must be >= 1")
    n = 2 * p + 1
    pts = np.arange(-p, p + 1, dtype=float)
    # Vandermonde-like matrix for derivative weights: V[j,k] = pts[k]^j / j!
    V = np.zeros((n, n))
    for j in range(n):
        for k in range(n):
            V[j, k] = (pts[k] ** j) / math.factorial(j)
    rhs = np.zeros(n)
    rhs[2] = 1.0  # second derivative
    c = np.linalg.solve(V, rhs)
    # Symmetry enforcement (should be symmetric to machine precision)
    c = 0.5 * (c + c[::-1])
    return c


def validate_stencil(p: int, tol: float = 1.0e-10) -> dict:
    """Validate the stencil by checking it reproduces derivatives of x^2, x^4
    etc., up to the designed order of accuracy. Returns a dict of errors."""
    c = build_laplacian_stencil(p)
    # Test on f(x) = x^4: d^2/dx^2 x^4 = 12 x^2
    pts = np.arange(-p, p + 1, dtype=float)
    f = pts ** 4
    approx = np.dot(c, f)
    exact = 0.0  # d^2/dx^2 x^4 at x=0 is 0
    err_x4 = abs(approx - exact)
    # Test on f(x) = x^2: d^2/dx^2 x^2 = 2
    f2 = pts ** 2
    approx2 = np.dot(c, f2)
    err_x2 = abs(approx2 - 2.0)
    # Test on f(x) = x^{2p}: should NOT be exact (exceeds order)
    f2p = pts ** (2 * p)
    approx2p = np.dot(c, f2p)
    return {"err_x2": err_x2, "err_x4": err_x4, "approx_x2p_at_0": approx2p}


# ============================================================================
# von Neumann amplification factor G(theta) for the explicit scheme
# applied to du/dt = D d^2u/dx^2 with the 2p-point Laplacian.
#   G(theta) = 1 + (D dt / h^2) * sum_{k=-p}^{p} c_k e^{i k theta}
# Since c is symmetric, the sum is real: sum_k c_k cos(k theta).
# Stability requires |G(theta)| <= 1 for all theta in [0, pi].
# ============================================================================
def amplification_factor(theta: NDArray, p: int,
                         r: float) -> NDArray:
    """Compute G(theta) = 1 + r * sum_k c_k cos(k theta).
    r = D dt / h^2 is the mesh ratio.
    """
    c = build_laplacian_stencil(p)
    pts = np.arange(-p, p + 1, dtype=float)
    # Real part only (c symmetric => imaginary part vanishes)
    s = sum(c[k] * np.cos(pts[k] * theta) for k in range(len(c)))
    return 1.0 + r * s


def max_stable_r(p: int, n_theta: int = 2000) -> float:
    """Find the maximum r = D dt / h^2 for which |G(theta)| <= 1 for all theta.
    Uses bisection on r in [0, 1]."""
    theta = np.linspace(0, math.pi, n_theta)

    def worst_G(r: float) -> float:
        G = amplification_factor(theta, p, r)
        return float(np.max(np.abs(G)))

    lo, hi = 0.0, 1.0
    # First bracket: if worst_G(1.0) <= 1 then hi=1 is fine
    if worst_G(1.0) <= 1.0 + 1e-12:
        return 1.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if worst_G(mid) <= 1.0 + 1e-12:
            lo = mid
        else:
            hi = mid
    return lo


# ============================================================================
# 1D Poisson solver with Dirichlet BCs using the 2p-point Laplacian
# ============================================================================
def solve_poisson_1d(source: NDArray, dx: float,
                     bc_left: float = 0.0, bc_right: float = 0.0,
                     p: int = FD_ORDER) -> NDArray:
    """Solve -eps d^2 phi/dx^2 = rho(x) on a uniform grid with Dirichlet BCs.

    Returns phi as an array of length len(source) including the two boundary
    nodes. Interior unknowns are solved by the p-banded linear system.

    Parameters
    ----------
    source : array (N,)
        Right-hand side rho(x_i) / (eps_r eps_0) at grid points i = 0..N-1.
    dx : float
        Grid spacing.
    bc_left, bc_right : float
        Dirichlet values at i=0 and i=N-1.
    p : int
        Half-width of the stencil (2p+1 points).
    """
    N = source.shape[0]
    if N < 2 * p + 1:
        raise ValueError(
            f"Need N >= 2p+1 = {2*p+1} for stencil half-width p = {p}")
    c = build_laplacian_stencil(p)
    # Build banded Laplacian A (negative sign: A phi = -source for Poisson)
    A = np.zeros((N, N))
    for i in range(p, N - p):
        for k in range(-p, p + 1):
            A[i, i + k] = -c[k + p] / (dx * dx)
    # Boundary rows: identity (Dirichlet)
    for j in range(p):
        A[j, j] = 1.0
        A[N - 1 - j, N - 1 - j] = 1.0
    # RHS
    b = np.copy(source)
    # Move known boundary contributions into the RHS for rows near the edge.
    # For row i in [p, 2p-1], some stencil points land on i+k < p (boundary)
    # whose value is set via the BC. But our boundary rows simply enforce
    # phi[j] = bc_j; we therefore overwrite the source contribution at
    # boundary nodes to reflect the Dirichlet value.
    for j in range(p):
        b[j] = bc_left
        b[N - 1 - j] = bc_right
    # Solve
    phi = np.linalg.solve(A, b)
    return phi


# ============================================================================
# 2D Poisson solver (tensor-product extension)
# ============================================================================
def solve_poisson_2d(source_2d: NDArray, dx: float, dy: float,
                     bc: dict, p: int = FD_ORDER) -> NDArray:
    """Solve -eps Laplacian phi = rho on a rectangular grid using the
    tensor-product 2p-point Laplacian in x and y.

    bc : dict with keys 'left','right','bottom','top' giving Dirichlet values.
    """
    Ny, Nx = source_2d.shape
    if Nx < 2 * p + 1 or Ny < 2 * p + 1:
        raise ValueError("Grid too small for the chosen stencil order.")
    cx = build_laplacian_stencil(p)
    cy = build_laplacian_stencil(p)
    # Assemble 2D operator as a sparse-like dense (for small reproducible
    # experiments we use a dense Kronecker sum).
    Ix = np.eye(Nx)
    Iy = np.eye(Ny)
    Lx = np.zeros((Nx, Nx))
    Ly = np.zeros((Ny, Ny))
    for i in range(p, Nx - p):
        for k in range(-p, p + 1):
            Lx[i, i + k] = cx[k + p] / (dx * dx)
    for j in range(p, Ny - p):
        for k in range(-p, p + 1):
            Ly[j, j + k] = cy[k + p] / (dy * dy)
    # Kronecker sum: L = Ly (x) Ix + Iy (x) Lx
    L = np.kron(Ly, Ix) + np.kron(Iy, Lx)
    L = -L  # we solve -Lap phi = source
    b = source_2d.flatten()
    # BC enforcement: set rows for boundary nodes to identity
    for j in range(Ny):
        for i in range(Nx):
            idx = j * Nx + i
            if i < p or i >= Nx - p or j < p or j >= Ny - p:
                L[idx, :] = 0.0
                L[idx, idx] = 1.0
                if i < p:
                    b[idx] = bc['left']
                elif i >= Nx - p:
                    b[idx] = bc['right']
                elif j < p:
                    b[idx] = bc['bottom']
                else:
                    b[idx] = bc['top']
    phi_flat = np.linalg.solve(L, b)
    return phi_flat.reshape(Ny, Nx)


# ============================================================================
# Richardson extrapolation: estimate convergence rate from 3 solutions
# on successively refined grids.
# ============================================================================
def richardson_order(phi_h: NDArray, phi_h2: NDArray, phi_h4: NDArray,
                     probe_idx: int) -> float:
    """Given three solutions on grids with spacings h, h/2, h/4, estimate
    the empirical convergence order p_emp at probe_idx.
    p_emp = log2( (phi_h - phi_h2) / (phi_h2 - phi_h4) )
    """
    d1 = phi_h[probe_idx] - phi_h2[probe_idx]
    d2 = phi_h2[probe_idx] - phi_h4[probe_idx]
    if abs(d2) < 1.0e-30:
        return float('nan')
    return math.log2(abs(d1) / abs(d2))


# ============================================================================
# Time-stepping: explicit FTCS with the 2p-point Laplacian.
# Used for the transient defect-ion diffusion equation.
# ============================================================================
def explicit_diffusion_step(u: NDArray, D: float, dx: float, dt: float,
                            p: int = FD_ORDER) -> NDArray:
    """Advance u by one explicit FTCS step using the 2p-point Laplacian.
    u includes Dirichlet boundary nodes (unchanged by this step)."""
    c = build_laplacian_stencil(p)
    r = D * dt / (dx * dx)
    N = u.shape[0]
    u_new = np.copy(u)
    for i in range(p, N - p):
        lap = sum(c[k + p] * u[i + k - p] for k in range(2 * p + 1)) / (dx * dx)
        u_new[i] = u[i] + D * dt * lap
    # Boundary: Dirichlet (unchanged)
    return u_new


def stable_dt(D: float, dx: float, p: int = FD_ORDER,
              safety: float = 0.9) -> float:
    """Return a stable time step dt for explicit diffusion using the
    2p-point Laplacian. safety in (0,1) is a CFL safety factor."""
    r_max = max_stable_r(p)
    return safety * r_max * dx * dx / D


# ============================================================================
# Driver: compute the equilibrium electrostatic potential for a given
# defect-charge profile rho_def(x) in the perovskite slab.
# ============================================================================
def compute_defect_potential(rho_def_C_per_m3: NDArray,
                             nx: int = DEFAULT_NX,
                             p: int = FD_ORDER) -> Tuple[NDArray, NDArray]:
    """Given a defect charge density profile rho_def [C/m^3] on a uniform
    1D grid of nx points across the perovskite slab, solve the Poisson
    equation and return (x_grid [m], phi [V])."""
    if nx < 2 * p + 1:
        raise ValueError("Grid too coarse for stencil order")
    L = DEVICE_LENGTH_M
    dx = L / (nx - 1)
    x = np.linspace(0.0, L, nx)
    # Source: rho / (eps_r eps_0)
    source = rho_def_C_per_m3 / EPS_PERP
    # BCs: phi(0) = V_NET (contact), phi(L) = 0
    phi = solve_poisson_1d(source, dx,
                           bc_left=V_NET, bc_right=0.0, p=p)
    return x, phi


# ============================================================================
# Vandermonde-based non-uniform derivative weights
# (ported from quadrature_weights_vandermonde_2d)
# ============================================================================
def vandermonde_derivative_weights(pts: NDArray, x_target: float,
                                   deriv_order: int = 2) -> NDArray:
    """Compute FD weights for the deriv_order-th derivative at x_target,
    given arbitrary node locations pts (length N). Returns weights w of
    length N such that f^{(d)}(x_target) ~ sum_k w_k f(pts_k).

    This is the 1D analogue of quadrature_weights_vandermonde_2d: we build
    the Vandermonde matrix V_{j,k} = pts_k^j / j! and solve V^T w = e_d.
    """
    N = len(pts)
    if deriv_order >= N:
        raise ValueError("Not enough points for this derivative order")
    V = np.zeros((N, N))
    for j in range(N):
        for k in range(N):
            V[j, k] = (pts[k] ** j) / math.factorial(j)
    rhs = np.zeros(N)
    rhs[deriv_order] = 1.0
    w = np.linalg.solve(V.T, rhs)
    return w


# ============================================================================
# Self-consistent sanity report
# ============================================================================
def sanity_report() -> dict:
    """Run a collection of sanity checks on the FD module."""
    report = {}
    for p in [1, 2, 3, 4, 5, 6]:
        val = validate_stencil(p)
        report[f"p={p}"] = val
        r_max = max_stable_r(p)
        report[f"p={p}_r_max"] = round(r_max, 6)
    return report
