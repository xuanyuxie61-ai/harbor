"""
boundary_handler.py  --  Boundary condition handling and numerical robustness
============================================================================
Fused seeds:
    999_r8sto              -- packed-row sparse storage with boundary pointers
    1294_Ankur-IIT_Modified_PNP-NS_Model -- Dirichlet boundary conditions
    369_fd2d_predator_prey -- ghost-cell boundary treatment

The radial Schrödinger equation for u(r) = r R(r) is a boundary-value problem:

    u(0) = 0             (regularity at origin)
    u(R_max) = 0         (box quantisation)

We implement:
  - Boundary-layer detection (identify mesh points where the wavefunction
    decays by more than a factor of 10 per mesh interval).
  - Ghost-cell extrapolation for Neumann-like conditions on derived operators.
  - Overflow/underflow protection for the WS potential at r -> 0 and r -> inf.
  - Adaptive mesh refinement where the wavefunction has rapid variation.
  - Self-consistency boundary check (the mean-field potential at R_max must
    be negligible compared to V(0)).

References:
    Press et al., Numerical Recipes, Sec. 17.1 (boundary-value problems)
    Trefethen, Spectral Methods in MATLAB (2000), Ch. 13
"""

from __future__ import annotations
import math
import numpy as np
from nuclear_constants import R_EPSILON, WAVE_TOL, MAX_RADIAL_GRID


# ======================================================================
#  Boundary layer detection
# ======================================================================
def detect_boundary_layers(u: np.ndarray, r: np.ndarray, threshold: float = 10.0) -> list:
    """Identify mesh indices where |u| decays by more than `threshold` per interval.

    Returns list of indices. These mark the onset of asymptotic decay and
    indicate where the mesh may need to be refined or R_max adjusted.
    """
    abs_u = np.abs(u)
    abs_u = np.where(abs_u < R_EPSILON, R_EPSILON, abs_u)
    ratio = abs_u[1:] / abs_u[:-1]
    indices = []
    for i in range(len(ratio)):
        if ratio[i] > threshold or ratio[i] < 1.0 / threshold:
            indices.append(i + 1)
    return indices


def check_box_size(u: np.ndarray, r: np.ndarray, threshold: float = 1.0e-4) -> dict:
    """Verify that u(R_max) is negligible compared to max|u|.

    If not, R_max is too small for the bound state to be properly contained
    in the box, and the energy will be contaminated by finite-size effects.
    """
    abs_u = np.abs(u)
    u_max = np.max(abs_u)
    u_end = abs_u[-1]
    ratio = u_end / u_max if u_max > R_EPSILON else 0.0
    return {
        "u_max": float(u_max),
        "u_at_Rmax": float(u_end),
        "ratio": float(ratio),
        "box_ok": ratio < threshold,
        "r_max_fm": float(r[-1]),
    }


# ======================================================================
#  Origin regularity
# ======================================================================
def check_origin_regularity(u: np.ndarray, r: np.ndarray, l_q: int) -> dict:
    """Verify u(r) ~ r^{l+1} near the origin (regularity of R(r)).

    For u(r) = r R(r), we need u(r) / r^{l+1} -> const as r -> 0.
    """
    # Find first few non-trivial points
    mask = (r > R_EPSILON) & (r < r[-1] / 10.0)
    if not np.any(mask):
        return {"regular": False, "ratio": float("inf")}
    r_sel = r[mask]
    u_sel = u[mask]
    u_sel = np.where(np.abs(u_sel) < R_EPSILON, R_EPSILON * np.sign(u_sel + R_EPSILON), u_sel)
    ratio = u_sel / (r_sel ** (l_q + 1))
    # Regularity: ratio should be nearly constant
    if np.std(ratio) / max(np.mean(np.abs(ratio)), R_EPSILON) > 0.2:
        return {"regular": False, "ratio_std_over_mean": float(np.std(ratio) / max(np.mean(np.abs(ratio)), R_EPSILON))}
    return {"regular": True, "ratio_std_over_mean": float(np.std(ratio) / max(np.mean(np.abs(ratio)), R_EPSILON))}


# ======================================================================
#  Ghost-cell extrapolation
# ======================================================================
def ghost_cell_extrapolate(u: np.ndarray, order: int = 2) -> np.ndarray:
    """Extend array u with `order` ghost cells at each end via polynomial extrapolation.

    Used when computing derivative operators that need values outside the domain.
    """
    if len(u) < order + 1:
        return u
    # Left ghost cells (extrapolate from u[0:order+1])
    left = np.polyval(np.polyfit(np.arange(order + 1), u[:order + 1], order), -np.arange(1, order + 1)[::-1])
    # Right ghost cells
    right = np.polyval(np.polyfit(np.arange(order + 1), u[-order - 1:], order), len(u) + np.arange(1, order + 1) - 1)
    return np.concatenate([left, u, right])


# ======================================================================
#  Adaptive mesh refinement
# ======================================================================
def refine_mesh_near(
    r: np.ndarray, u: np.ndarray, r_center: float, width: float,
    refinement_factor: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Refine the mesh locally around r_center +/- width.

    Returns (r_new, u_new) with the refinement region having
    refinement_factor times the original point density.
    """
    h = r[1] - r[0]
    mask_in = (r >= r_center - width) & (r <= r_center + width)
    mask_out = ~mask_in
    r_in = r[mask_in]
    u_in = u[mask_in]
    r_out = r[mask_out]
    u_out = u[mask_out]
    # Create refined interior mesh
    n_fine = len(r_in) * refinement_factor
    r_fine = np.linspace(r_in[0], r_in[-1], n_fine)
    u_fine = np.interp(r_fine, r_in, u_in)
    # Merge, keeping order
    r_new = np.sort(np.concatenate([r_out, r_fine]))
    u_new = np.interp(r_new, r, u)  # interpolate from original for safety
    return r_new, u_new


# ======================================================================
#  Self-consistency boundary check
# ======================================================================
def self_consistency_check(V_eff: np.ndarray, r: np.ndarray) -> dict:
    """Verify that the effective potential at R_max is negligible.

    Condition: |V_eff(R_max)| < 0.01 * |V_eff(0)|
    """
    V_max = abs(V_eff[-1])
    V_0 = abs(V_eff[1]) if len(V_eff) > 1 else R_EPSILON
    return {
        "V_at_0_MeV": float(V_0),
        "V_at_Rmax_MeV": float(V_max),
        "ratio": float(V_max / V_0) if V_0 > R_EPSILON else float("inf"),
        "consistent": V_max < 0.01 * V_0,
    }


# ======================================================================
#  Normalisation check
# ======================================================================
def check_normalisation(u: np.ndarray, r: np.ndarray, tol: float = WAVE_TOL) -> dict:
    """Verify that int_0^Rmax |u(r)|^2 dr ~ 1."""
    norm_sq = np.trapz(u * u, r)
    return {
        "norm_sq": float(norm_sq),
        "deviation": float(abs(norm_sq - 1.0)),
        "ok": abs(norm_sq - 1.0) < tol,
    }


# ======================================================================
#  Comprehensive boundary diagnostics
# ======================================================================
def full_boundary_diagnostics(
    u: np.ndarray, r: np.ndarray, V_eff: np.ndarray, l_q: int,
) -> dict:
    """Run all boundary / robustness checks and return a consolidated report."""
    return {
        "box_check": check_box_size(u, r),
        "origin_check": check_origin_regularity(u, r, l_q),
        "norm_check": check_normalisation(u, r),
        "self_consistency": self_consistency_check(V_eff, r),
        "boundary_layers": detect_boundary_layers(u, r),
        "n_boundary_layers": len(detect_boundary_layers(u, r)),
    }
