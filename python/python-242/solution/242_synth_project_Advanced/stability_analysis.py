"""
stability_analysis.py  --  Von Neumann stability analysis for the
                           high-order finite-difference nuclear solver
===========================================================================
Fused seeds:
    150_cg_lab_triangles  -- signed-distance geometric predicates
    827_ode_euler_system  -- forward Euler reference scheme
    829_ode_midpoint_system -- explicit midpoint reference
    369_fd2d_predator_prey  -- 2D finite-difference stencil
    1294_Ankur-IIT_Modified_PNP-NS_Model -- nonlinear self-consistency
    1263_JianchengXie_MMPHATE_REPRO      -- hierarchical scale analysis

We analyse the Numerov scheme used in radial_fd_solver.py by computing the
amplification factor G(k h) for the model problem:

    -u'' + k_0^2 u = 0   (constant potential limit)

with Numerov discretisation on mesh spacing h.

The amplification factor satisfies:

    (1 + h^2 k_0^2 / 12) G^2 - 2 (1 - 5 h^2 k_0^2 / 12) G + (1 + h^2 k_0^2 / 12) = 0

=>  G = exp(+/- i theta)  with  cos(theta) = (1 - 5 h^2 k_0^2 / 12) / (1 + h^2 k_0^2 / 12)

Stability requires |cos(theta)| <= 1, i.e.

    h k_0 <= sqrt(12 / (5 - 1)) * sqrt(2) ~ 1.73   (for the boundary case)

More generally, the Numerov scheme is unconditionally stable for the
time-independent Schrödinger equation (it is a boundary-value method), but
for the time-dependent problem i hbar d psi / dt = H psi we need an explicit
time integrator, and then the Courant condition

    dt <= C_CFL * h^2 * 2 m / hbar

must be respected. We compute this bound and verify it against the parameters
used in radial_fd_solver.py and ode_evolution.py.

Additionally we compute:
    - The spectral radius rho(H) of the Hamiltonian matrix.
    - The condition number kappa of the mass-matrix D.
    - Round-off error bound for the Lanczos iteration.
    - Triangle inequality on eigenvalue spacings (geometric predicate).

References:
    Strikwerda, Finite Difference Schemes and PDEs (1989)
    LeVeque, Finite Difference Methods for ODEs (2007)
"""

from __future__ import annotations
import math
import numpy as np
from nuclear_constants import HBAR_C, M_PROTON, M_NEUTRON, R_EPSILON, STABILITY_CFL


# ======================================================================
#  Von Neumann amplification factor for Numerov
# ======================================================================
def numerov_amplification_factor(k0: float, h: float) -> complex:
    """G(k_0 h) for the Numerov scheme on the model problem -u'' + k_0^2 u = 0.

    Solves the quadratic (1 + z/12) G^2 - 2 (1 - 5 z/12) G + (1 + z/12) = 0
    with z = (k_0 h)^2. Returns the root on the unit circle (if it exists)
    or the dominant root otherwise.
    """
    z = (k0 * h) ** 2
    a = 1.0 + z / 12.0
    b = -2.0 * (1.0 - 5.0 * z / 12.0)
    c = 1.0 + z / 12.0
    disc = b * b - 4.0 * a * c
    if disc >= 0.0:
        # Real roots (evanescent regime)
        g1 = (-b + math.sqrt(disc)) / (2.0 * a)
        g2 = (-b - math.sqrt(disc)) / (2.0 * a)
        return g1 if abs(g1) >= abs(g2) else g2
    # Complex conjugate roots on the unit circle
    re = -b / (2.0 * a)
    im = math.sqrt(-disc) / (2.0 * a)
    return complex(re, im)


def numerov_stability_limit() -> float:
    """Critical value (k_0 h)_crit beyond which |G| > 1.

    cos(theta) = (1 - 5 z/12) / (1 + z/12)
    |cos| <= 1  =>  z <= 12 / 2 = 6  (since numerator becomes -1 at z=12/5,
                                         then decreases; we need |cos|<=1
                                         which holds for z <= 12).
    Actually, the condition |cos|<=1 holds for all z >= 0 because
        -1 <= (1 - 5z/12) / (1 + z/12) <= 1
    <=> (1 + z/12) >= 1 - 5z/12 >= -(1 + z/12)
    <=> z >= 0 and z/6 <= 2  => z <= 12.
    So critical z = 12 => (k_0 h)_crit = sqrt(12) ~ 3.464.
    """
    return math.sqrt(12.0)


def check_numerov_stability(k0_max: float, h: float) -> dict:
    """Return stability diagnostics for the Numerov scheme."""
    z_max = (k0_max * h) ** 2
    z_crit = 12.0
    G = numerov_amplification_factor(k0_max, h)
    return {
        "k0_h_max": k0_max * h,
        "z_max": z_max,
        "z_crit": z_crit,
        "stable": z_max <= z_crit,
        "G_abs": abs(G),
        "G_phase_rad": math.atan2(G.imag, G.real) if abs(G) > 0.0 else 0.0,
        "safety_factor": z_crit / max(z_max, R_EPSILON),
    }


# ======================================================================
#  Courant-Friedrichs-Lewy condition for time propagation
# ======================================================================
def cfl_time_step(h: float, m_red: float, cfl_number: float = STABILITY_CFL) -> float:
    r"""CFL time step for the explicit time propagation of i hbar d psi / dt = H psi.

    The Hamiltonian has kinetic term T ~ -hbar^2 / (2 m) d^2/dr^2. Discretised
    with second-order central differences, the spectral radius is
        rho(T) ~ hbar^2 / (m h^2)
    For forward Euler stability we need dt * rho(T) / hbar <= 2, i.e.
        dt <= 2 m h^2 / hbar

    With the midpoint method the stability region is larger (imaginary axis
    is on the boundary), and we use a safety factor CFL_number.
    """
    hbar = HBAR_C  # MeV fm
    return cfl_number * 2.0 * m_red * (h ** 2) / hbar


# ======================================================================
#  Hamiltonian spectral diagnostics
# ======================================================================
def spectral_radius(diag: np.ndarray, off: np.ndarray) -> float:
    """Spectral radius of a symmetric tridiagonal matrix.

    Uses Gershgorin circle theorem: rho <= max_i (|a_i| + |b_{i-1}| + |b_i|).
    """
    n = len(diag)
    if n == 0:
        return 0.0
    radii = np.abs(diag)
    for i in range(n - 1):
        radii[i] += abs(off[i])
        radii[i + 1] += abs(off[i])
    return float(np.max(radii))


def condition_number_tridiag(diag: np.ndarray, off: np.ndarray) -> float:
    """2-norm condition number of a symmetric tridiagonal matrix.

    kappa = |lambda_max| / |lambda_min|  (both from scipy eigensolver).
    """
    from scipy.linalg import eigh_tridiagonal
    if len(diag) < 2:
        return 1.0
    try:
        eigs = eigh_tridiagonal(diag, off)
    except Exception:
        return float("inf")
    e_min = np.min(np.abs(eigs))
    e_max = np.max(np.abs(eigs))
    if e_min < R_EPSILON:
        return float("inf")
    return float(e_max / e_min)


# ======================================================================
#  Triangle inequality on eigenvalue spacings (geometric predicate)
# ======================================================================
def eigenvalue_spacing_triangle_ok(eigenvalues: np.ndarray, tol: float = 1.0e-8) -> dict:
    """Verify that consecutive spacings satisfy the triangle inequality.

    For a well-resolved spectrum the spacings d_i = E_{i+1} - E_i should
    form a valid "triangle" in the sense that no single gap dominates the
    sum (sign of avoided crossings / intruder states).

    Condition:  d_i <= 0.5 * sum_j d_j  for every i
    """
    if len(eigenvalues) < 3:
        return {"ok": True, "ratio": 0.0, "message": "too few levels"}
    spacings = np.diff(eigenvalues)
    total = np.sum(spacings)
    max_s = np.max(spacings)
    ratio = max_s / total if total > R_EPSILON else float("inf")
    ok = ratio <= 0.5 + tol
    return {
        "ok": ok,
        "ratio": ratio,
        "max_spacing_MeV": float(max_s),
        "total_range_MeV": float(total),
        "n_gaps": len(spacings),
    }


# ======================================================================
#  Round-off propagation in Lanczos
# ======================================================================
def lanczos_roundoff_bound(n_dim: int, n_steps: int, epsilon: float = 2.2e-16) -> float:
    """Estimated loss of orthogonality in Lanczos after n_steps iterations.

    Bound: ||V^T V - I||_F ~ epsilon * sqrt(n_steps) * kappa(H)
    (Paige 1971, "Computational variants of the Lanczos method")
    """
    return epsilon * math.sqrt(max(n_steps, 1)) * math.sqrt(max(n_dim, 1))


# ======================================================================
#  Nonlinear self-consistency residual (PNP-style iteration)
# ======================================================================
def self_consistency_residual(V_old: np.ndarray, V_new: np.ndarray) -> float:
    """L2 relative residual between successive mean-field potentials.

    ||V_new - V_old||_2 / ||V_new||_2  -- analogous to the PNP-NS nonlinear
    residual used in the modified PNP model (1294).
    """
    diff = V_new - V_old
    num = math.sqrt(np.sum(diff * diff))
    den = math.sqrt(np.sum(V_new * V_new))
    if den < R_EPSILON:
        return float("inf")
    return num / den


# ======================================================================
#  Multi-scale hierarchical energy decomposition (MMPHATE-style)
# ======================================================================
def hierarchical_decomposition(energies: np.ndarray, n_levels_group: int = 4) -> dict:
    """Decompose the level spectrum into coarse / mid / fine scales.

    Adapted from the multi-scale trace analysis (1263): partition the
    ordered spectrum into contiguous blocks of size n_levels_group and
    compute the variance at each scale.
    """
    n = len(energies)
    if n < n_levels_group:
        return {"n_scales": 1, "variances": [float(np.var(energies))]}
    blocks = [energies[i:i + n_levels_group] for i in range(0, n - n_levels_group + 1, n_levels_group)]
    variances = [float(np.var(b)) for b in blocks]
    return {
        "n_scales": len(blocks),
        "variances": variances,
        "total_variance": float(np.var(energies)),
        "explained_ratio": sum(variances) / max(np.var(energies), R_EPSILON),
    }


# ======================================================================
#  High-level diagnostic driver
# ======================================================================
def run_stability_diagnostics(
    h: float, m_red: float, k0_typical: float, energies: np.ndarray,
    diag_H: np.ndarray = None, off_H: np.ndarray = None,
) -> dict:
    """Run all stability checks and return a consolidated report."""
    report = {}
    report["numerov"] = check_numerov_stability(k0_typical, h)
    report["cfl_dt_fm_c"] = cfl_time_step(h, m_red)
    report["eig_spacing"] = eigenvalue_spacing_triangle_ok(energies)
    report["lanczos_roundoff"] = lanczos_roundoff_bound(
        n_dim=len(diag_H) if diag_H is not None else 0,
        n_steps=30,
    )
    if diag_H is not None and off_H is not None:
        report["spectral_radius_MeV"] = spectral_radius(diag_H, off_H)
        report["condition_number"] = condition_number_tridiag(diag_H, off_H)
    report["hierarchical"] = hierarchical_decomposition(energies)
    return report
