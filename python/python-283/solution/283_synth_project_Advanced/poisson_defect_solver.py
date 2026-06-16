"""
poisson_defect_solver.py
========================
Self-consistent Poisson + SRH recombination solver for defect states in
perovskite solar cells. Combines the high-order finite-difference Poisson
solver (high_order_fd) with the Shockley-Read-Hall (SRH) recombination model
to compute the steady-state electrostatic potential phi(x), electron and
hole densities n(x), p(x), and the defect occupancy f_t(x).

The physical system
-------------------
For a single defect level at energy E_t in the band gap, the SRH net
recombination rate is
    R_SRH = (np - n_i^2) / [tau_p0 (n + n_1) + tau_n0 (p + p_1)]
where
    n_1 = N_c exp(-(E_c - E_t) / kT)
    p_1 = N_v exp(-(E_t - E_v) / kT)
    tau_n0 = 1 / (v_th,n sigma_n N_t)
    tau_p0 = 1 / (v_th,p sigma_p N_t)
    n_i^2 = N_c N_v exp(-E_g / kT)

The defect occupancy (fraction of defects filled by electrons):
    f_t = [sigma_n n + sigma_p p_1 exp((E_t - E_i)/kT)] /
          [sigma_n (n + n_1) + sigma_p (p + p_1)]

The defect charge density:
    rho_def(x) = q N_t(x) [f_t(x) - f_0]
where f_0 is the equilibrium (dark) occupancy.

The Poisson equation:
    -eps_r eps_0 d^2 phi/dx^2 = q [p - n + N_D^+ - N_A^- + rho_def/q]

Gummel iteration
----------------
The nonlinear Poisson + continuity system is solved by Gummel iteration:
    (1) Given phi^{(k)}, compute n^{(k)}, p^{(k)} via Boltzmann stats:
            n = N_c exp(-(E_c - E_Fn) / kT) = N_c exp(-q(phi - V_Fn)/V_t)
            p = N_v exp(-(E_Fp - E_v) / kT) = N_v exp(-q(V_Fp - phi)/V_t)
    (2) Compute f_t, rho_def from n, p.
    (3) Solve Poisson for phi^{(k+1)}.
    (4) Under-relax: phi <- (1 - omega) phi^{(k)} + omega phi^{(k+1)}.
    (5) Check convergence: max|phi^{(k+1)} - phi^{(k)}| < tol.
"""

from __future__ import annotations
import math
from typing import Tuple, Optional

import numpy as np
from numpy.typing import NDArray

from perovskite_constants import (
    BAND_GAP_EV, BAND_GAP_J, THERMAL_VOLTAGE, E_CHARGE,
    DOS_CONDUCTION, DOS_VALENCE, EPS_PERP,
    V_TH_ELECTRON, V_TH_HOLE, DEFECT_CAPTURE_E, DEFECT_CAPTURE_H,
    DEFECT_DENSITY_DEFAULT, DEFECT_DEPTH_EV, DEVICE_LENGTH_M, V_NET,
    DEFAULT_NX, FD_ORDER
)
from high_order_fd import solve_poisson_1d, build_laplacian_stencil


# ============================================================================
# Energy band reference
# ============================================================================
def band_edges(phi: NDArray) -> Tuple[NDArray, NDArray]:
    """Return (E_c, E_v) in Joules given the electrostatic potential phi [V].

    E_c(x) = -q phi(x)                (referenced to vacuum at phi = 0)
    E_v(x) = E_c(x) - E_g

    We set the reference such that E_c = 0 when phi = 0.
    """
    E_c = -E_CHARGE * phi
    E_v = E_c - BAND_GAP_J
    return E_c, E_v


# ============================================================================
# Boltzmann carrier densities
# ============================================================================
def carrier_densities_boltzmann(phi: NDArray,
                                E_Fn: float, E_Fp: float
                                ) -> Tuple[NDArray, NDArray]:
    """Compute n(x), p(x) from Boltzmann statistics.
    E_Fn, E_Fp are the electron and hole quasi-Fermi levels [J].
    """
    E_c, E_v = band_edges(phi)
    n = DOS_CONDUCTION * np.exp(-(E_c - E_Fn) / (K_B_T()))
    p = DOS_VALENCE * np.exp(-(E_Fp - E_v) / (K_B_T()))
    return n, p


def K_B_T() -> float:
    """kT in Joules."""
    from perovskite_constants import K_B, T_K
    return K_B * T_K


def intrinsic_density() -> float:
    """n_i = sqrt(N_c N_v) exp(-E_g / 2 kT)."""
    return math.sqrt(DOS_CONDUCTION * DOS_VALENCE) * math.exp(
        -BAND_GAP_J / (2.0 * K_B_T()))


# ============================================================================
# SRH recombination and defect occupancy
# ============================================================================
def srh_recombination(n: NDArray, p: NDArray,
                      N_t: NDArray,
                      E_t_J: float,
                      sigma_n: float = DEFECT_CAPTURE_E,
                      sigma_p: float = DEFECT_CAPTURE_H
                      ) -> Tuple[NDArray, NDArray]:
    """Compute the SRH recombination rate R_SRH and defect occupancy f_t.

    Returns:
        R_SRH : array of recombination rates [1/(m^3 s)]
        f_t   : defect occupancy (fraction filled by electrons)

    n_1, p_1:
        n_1 = N_c exp(-(E_c - E_t) / kT)
        p_1 = N_v exp(-(E_t - E_v) / kT)
    We place E_t at mid-gap for simplicity: E_c - E_t = E_g/2 - DEFECT_DEPTH_EV.
    """
    kT = K_B_T()
    E_t_from_Ec = BAND_GAP_J / 2.0 - DEFECT_DEPTH_EV * E_CHARGE
    n_1 = DOS_CONDUCTION * math.exp(-E_t_from_Ec / kT)
    p_1 = DOS_VALENCE * math.exp(-(BAND_GAP_J - E_t_from_Ec) / kT)
    n_i = intrinsic_density()
    tau_n0 = 1.0 / (V_TH_ELECTRON * sigma_n * np.maximum(N_t, 1.0))
    tau_p0 = 1.0 / (V_TH_HOLE * sigma_p * np.maximum(N_t, 1.0))
    denom = tau_p0 * (n + n_1) + tau_n0 * (p + p_1)
    # Clip denominator to avoid division by zero
    denom = np.maximum(denom, 1.0e-30)
    R_SRH = (n * p - n_i ** 2) / denom
    # Occupancy
    f_t = (sigma_n * n + sigma_p * p_1) / (
        sigma_n * (n + n_1) + sigma_p * (p + p_1))
    return R_SRH, f_t


def defect_charge_density(f_t: NDArray, N_t: NDArray,
                          f0: float = 0.5) -> NDArray:
    """Compute the defect charge density rho_def = q N_t (f_t - f0).
    f0 is the dark equilibrium occupancy (0.5 for mid-gap defect)."""
    return E_CHARGE * N_t * (f_t - f0)


# ============================================================================
# Self-consistent Poisson + SRH (Gummel iteration)
# ============================================================================
def gummel_poisson_srh(nx: int = DEFAULT_NX,
                       p_fd: int = FD_ORDER,
                       N_t_profile: Optional[NDArray] = None,
                       max_iter: int = 60,
                       tol: float = 1e-6,
                       omega: float = 0.3) -> dict:
    """Run Gummel iteration for the coupled Poisson + SRH system.

    Parameters
    ----------
    N_t_profile : array of defect densities [1/m^3] at grid points
    max_iter    : max Gummel iterations
    tol         : convergence tolerance on max |delta phi|
    omega       : under-relaxation parameter

    Returns
    -------
    dict with keys:
        'x', 'phi', 'n', 'p', 'f_t', 'R_SRH', 'rho_def',
        'converged', 'iterations', 'residual_history'
    """
    L = DEVICE_LENGTH_M
    dx = L / (nx - 1)
    x = np.linspace(0.0, L, nx)
    # Quasi-Fermi levels: flat under low injection
    # E_c at x=0: -q V_NET, E_c at x=L: 0
    # We set E_Fn = E_c(x) + 0.3 eV (n-type character)
    # and E_Fp = E_v(x) - 0.3 eV (p-type character)
    phi = np.linspace(V_NET, 0.0, nx)
    E_Fn = 0.3 * E_CHARGE  # constant across slab (flat quasi-Fermi)
    E_Fp = BAND_GAP_J - 0.3 * E_CHARGE
    if N_t_profile is None:
        # Gaussian defect cluster in the middle
        xc = 0.5 * L
        sigma_x = 0.1 * L
        N_t_profile = DEFECT_DENSITY_DEFAULT * np.exp(
            -((x - xc) ** 2) / (2.0 * sigma_x ** 2))
    residuals = []
    converged = False
    it = 0
    for it in range(1, max_iter + 1):
        n, p = carrier_densities_boltzmann(phi, E_Fn, E_Fp)
        R, f_t = srh_recombination(n, p, N_t_profile,
                                   BAND_GAP_J / 2.0)
        rho_def = defect_charge_density(f_t, N_t_profile, f0=0.5)
        # Total charge: defects only (we ignore free carriers in this
        # small-signal defect-state calculation)
        source = rho_def / EPS_PERP
        phi_new = solve_poisson_1d(source, dx,
                                   bc_left=V_NET, bc_right=0.0, p=p_fd)
        delta = np.max(np.abs(phi_new - phi))
        residuals.append(delta)
        phi = (1.0 - omega) * phi + omega * phi_new
        # Clamp phi to physical range
        phi = np.clip(phi, -1.0, V_NET + 1.0)
        if delta < tol:
            converged = True
            break
    # Final state
    n, p = carrier_densities_boltzmann(phi, E_Fn, E_Fp)
    R, f_t = srh_recombination(n, p, N_t_profile, BAND_GAP_J / 2.0)
    rho_def = defect_charge_density(f_t, N_t_profile, f0=0.5)
    return {
        "x": x,
        "phi": phi,
        "n": n,
        "p": p,
        "f_t": f_t,
        "R_SRH": R,
        "rho_def": rho_def,
        "N_t": N_t_profile,
        "converged": converged,
        "iterations": it,
        "residual_history": residuals,
    }


# ============================================================================
# Diagnostic: compute the total recombination current
# ============================================================================
def recombination_current(result: dict) -> float:
    """Integrate R_SRH across the slab to get J_rec = q Integral R_SRH dx."""
    x = result["x"]
    R = result["R_SRH"]
    J = E_CHARGE * np.trapz(R, x)
    return float(J)


# ============================================================================
# Diagnostic: built-in field and Debye length
# ============================================================================
def debye_length(n0: float) -> float:
    """Debye length L_D = sqrt(eps kT / (q^2 n0))."""
    return math.sqrt(EPS_PERP * K_B_T() / (E_CHARGE ** 2 * max(n0, 1.0)))


def electric_field(phi: NDArray, dx: float,
                   p: int = FD_ORDER) -> NDArray:
    """Compute E = -d phi/dx using the 2p-th order centered first derivative.

    The first-derivative stencil of order 2p is obtained from the
    Vandermonde system (see high_order_fd.vandermonde_derivative_weights).
    """
    from high_order_fd import vandermonde_derivative_weights
    pts = np.arange(-p, p + 1, dtype=float)
    w = vandermonde_derivative_weights(pts, 0.0, deriv_order=1)
    Nx = len(phi)
    E = np.zeros(Nx)
    for i in range(p, Nx - p):
        s = 0.0
        for k in range(-p, p + 1):
            s += w[k + p] * phi[i + k]
        E[i] = -s / dx
    # Boundaries: one-sided extrapolation
    for i in range(p):
        E[i] = E[p]
    for i in range(Nx - p, Nx):
        E[i] = E[Nx - p - 1]
    return E
