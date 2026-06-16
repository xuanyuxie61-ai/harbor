"""
stability_analysis.py
=====================
Von Neumann and matrix stability analysis for the high-order finite-
difference discretization of the perovskite defect-state equations.

We analyze three stability questions:
(1) von Neumann stability of the explicit FTCS scheme for the 2p-point
    Laplacian (transient defect-ion diffusion).
(2) Spectral stability of the Gummel-Poisson iteration (does the nonlinear
    fixed point converge?).
(3) CFL limits for the DG advection scheme in the carrier-transport solver.

Mathematical framework
----------------------
(1) von Neumann: substitute u_j^n = G^n exp(i j theta) into the discrete
    scheme u_j^{n+1} = u_j^n + r sum_k c_k u_{j+k}^n. The amplification
    factor is
        G(theta) = 1 + r sum_{k=-p}^{p} c_k exp(i k theta)
    where r = D dt / h^2. Stability requires |G(theta)| <= 1 for all theta.

(2) Gummel iteration as a fixed point phi^{n+1} = F(phi^n). Convergence
    requires rho(J_F) < 1 where J_F is the Jacobian of F at the fixed point.
    We estimate rho by finite-difference approximation of J_F.

(3) DG CFL: for polynomial order N and advection speed a, the time-step
    limit is dt <= CFL * h / (|a| (2N+1)^2) where CFL ~ 1/(2N+1).
"""

from __future__ import annotations
import math
from typing import Tuple, Callable, Optional

import numpy as np
from numpy.typing import NDArray

from perovskite_constants import (
    FD_ORDER, DEVICE_LENGTH_M, THERMAL_VOLTAGE, E_CHARGE,
    fd_stability_factor, DEFECT_DENSITY_DEFAULT
)
from high_order_fd import (
    build_laplacian_stencil, amplification_factor, max_stable_r,
    solve_poisson_1d
)


# ============================================================================
# (1) von Neumann spectral analysis
# ============================================================================
def von_neumann_analysis(p: int = FD_ORDER,
                         n_theta: int = 1000) -> dict:
    """Full von Neumann analysis for the 2p-point Laplacian.
    Returns:
        'theta'           : array of theta values in [0, pi]
        'G_at_r_max'      : G(theta) at the maximum stable r
        'r_max'           : maximum stable r
        'worst_G_vs_r'    : array of worst |G| for r in [0, 1]
    """
    theta = np.linspace(0.0, math.pi, n_theta)
    r_max = max_stable_r(p, n_theta=2000)
    G = amplification_factor(theta, p, r_max)
    # Sweep r
    r_sweep = np.linspace(0.0, 1.0, 200)
    worst = np.zeros_like(r_sweep)
    for i, r in enumerate(r_sweep):
        G_r = amplification_factor(theta, p, r)
        worst[i] = float(np.max(np.abs(G_r)))
    return {
        "theta": theta,
        "G_at_r_max": G,
        "r_max": r_max,
        "r_sweep": r_sweep,
        "worst_G_vs_r": worst,
        "theoretical_S_p": fd_stability_factor(p),
    }


# ============================================================================
# (2) Gummel iteration spectral radius
# ============================================================================
def gummel_jacobian_fd(phi_op: Callable[[NDArray], NDArray],
                       phi0: NDArray,
                       eps: float = 1e-6) -> NDArray:
    """Approximate the Jacobian of the Gummel operator phi -> F(phi)
    by finite differences. phi0 is the current iterate.
    Returns the Jacobian matrix J of shape (N, N).
    """
    N = len(phi0)
    F0 = phi_op(phi0)
    J = np.zeros((N, N))
    for j in range(N):
        phi_p = phi0.copy()
        phi_p[j] += eps
        F_p = phi_op(phi_p)
        J[:, j] = (F_p - F0) / eps
    return J


def gummel_spectral_radius(phi_op: Callable[[NDArray], NDArray],
                           phi0: NDArray) -> float:
    """Estimate the spectral radius of the Gummel iteration Jacobian."""
    J = gummel_jacobian_fd(phi_op, phi0)
    eigvals = np.linalg.eigvals(J)
    return float(np.max(np.abs(eigvals)))


def test_gummel_convergence(nx: int = 32) -> dict:
    """Test Gummel convergence on a simple Poisson problem with a fixed
    charge distribution. Returns the spectral radius and convergence log."""
    L = DEVICE_LENGTH_M
    dx = L / (nx - 1)
    x = np.linspace(0.0, L, nx)
    # Simple Gaussian charge
    xc = 0.5 * L
    sigma = 0.1 * L
    rho = E_CHARGE * DEFECT_DENSITY_DEFAULT * np.exp(
        -((x - xc) ** 2) / (2.0 * sigma ** 2))
    source = rho / (25.0 * 8.854e-12)

    def phi_op(phi):
        # One Gummel step: solve Poisson with a mild nonlinear screening
        # that depends on phi (simple test: source_eff = source * exp(-q phi/kT))
        screening = np.exp(-E_CHARGE * phi / (THERMAL_VOLTAGE * 10.0))
        source_eff = source * screening
        return solve_poisson_1d(source_eff, dx,
                                bc_left=1.0, bc_right=0.0,
                                p=FD_ORDER)

    phi0 = np.linspace(1.0, 0.0, nx)
    rho_J = gummel_spectral_radius(phi_op, phi0)
    # Run Gummel and record residual
    phi = phi0.copy()
    residuals = []
    for it in range(30):
        phi_new = phi_op(phi)
        delta = np.max(np.abs(phi_new - phi))
        residuals.append(delta)
        phi = 0.5 * phi + 0.5 * phi_new  # under-relaxation
        if delta < 1e-10:
            break
    return {
        "spectral_radius": rho_J,
        "converges": rho_J < 1.0,
        "residuals": residuals,
        "n_iters": len(residuals),
    }


# ============================================================================
# (3) DG CFL analysis
# ============================================================================
def dg_cfl_limit(N_poly: int, a: float = 1.0,
                 h: float = 1e-6) -> dict:
    """Compute the CFL limit for the DG method with polynomial order N_poly.
    The advective CFL is dt <= CFL_a * h / |a| where CFL_a ~ 1/(2N+1)^2.
    The diffusive CFL is dt <= CFL_d * h^2 / D where CFL_d ~ 1/(2N+1)^4.
    """
    cfl_a = 1.0 / ((2 * N_poly + 1) ** 2)
    cfl_d = 1.0 / ((2 * N_poly + 1) ** 4)
    dt_a = cfl_a * h / abs(a)
    return {
        "N_poly": N_poly,
        "cfl_advective": cfl_a,
        "cfl_diffusive": cfl_d,
        "dt_advective": dt_a,
        "dt_diffusive_per_D": cfl_d * h * h,
    }


# ============================================================================
# Eigenvalue spectrum of the FD Laplacian
# ============================================================================
def laplacian_eigenvalues(nx: int, p: int = FD_ORDER,
                          L: float = 1.0) -> NDArray:
    """Compute the eigenvalues of the 2p-point FD Laplacian matrix on a
    uniform grid of nx points with Dirichlet BCs. The continuous Laplacian
    on [0, L] with Dirichlet BCs has eigenvalues -(k pi / L)^2; the FD
    approximation has a modified dispersion relation."""
    dx = L / (nx - 1)
    c = build_laplacian_stencil(p)
    A = np.zeros((nx, nx))
    for i in range(p, nx - p):
        for k in range(-p, p + 1):
            A[i, i + k] = c[k + p] / (dx * dx)
    # BC rows
    for j in range(p):
        A[j, j] = 0.0
        A[nx - 1 - j, nx - 1 - j] = 0.0
    eigvals = np.linalg.eigvals(A)
    return np.sort(np.real(eigvals))


def dispersion_relation(p: int = FD_ORDER,
                        n_theta: int = 500) -> Tuple[NDArray, NDArray]:
    """Compute the numerical dispersion relation omega^2(theta) for the
    2p-point Laplacian: omega^2 = -sum_k c_k exp(i k theta).
    Returns (theta, omega_sq).
    """
    theta = np.linspace(0.0, math.pi, n_theta)
    c = build_laplacian_stencil(p)
    stencil_pts = np.arange(-p, p + 1, dtype=float)  # physical indices
    # Real part: sum_{m=-p}^{p} c_{m+p} cos(m theta)
    omega_sq = np.zeros_like(theta)
    for idx, th in enumerate(theta):
        s = 0.0
        for m_idx, m in enumerate(stencil_pts):
            s += c[m_idx] * math.cos(m * th)
        omega_sq[idx] = -s
    return theta, omega_sq


# ============================================================================
# Driver: full stability report
# ============================================================================
def run_full_stability_analysis() -> dict:
    """Run all stability analyses and compile a report."""
    report = {}
    # von Neumann for p = 1, 2, 3, 4, 5, 6
    vn_results = {}
    for p in [1, 2, 3, FD_ORDER]:
        vn_results[p] = von_neumann_analysis(p=p, n_theta=500)
    report["von_neumann"] = {
        p: {"r_max": v["r_max"],
            "theoretical_S_p": v["theoretical_S_p"]}
        for p, v in vn_results.items()
    }
    # Gummel
    report["gummel"] = test_gummel_convergence(nx=32)
    # DG CFL for N = 1, 2, 3, 4
    report["dg_cfl"] = {N: dg_cfl_limit(N) for N in [1, 2, 3, 4]}
    # Dispersion relation
    for p in [1, 3, FD_ORDER]:
        theta, omega_sq = dispersion_relation(p=p)
        report[f"dispersion_p{p}"] = {
            "theta_min_omega_sq": float(theta[np.argmin(omega_sq)]),
            "max_omega_sq": float(np.max(omega_sq)),
        }
    return report
