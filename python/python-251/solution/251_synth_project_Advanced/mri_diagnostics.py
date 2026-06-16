"""
mri_diagnostics.py
==================
Diagnostic quantities that characterise the nonlinear saturation of the
magnetorotational instability (MRI) in the shearing-box simulation.

The following volume-averaged stresses are computed at every history
output:

    Reynolds stress   <R_xy> = < rho v_x' v_y' >
    Maxwell stress    <M_xy> = < - B_x B_y / (4 pi) >
    alpha_SS          = (<R_xy> + <M_xy>) / < p >

where angle brackets denote a volume average over the box and
primed velocities are deviations from the box-averaged flow.

The module also computes the *turbulence coherence angle* of the
Maxwell stress tensor, adapting the positional-angle statistical
framework of davfer12 (1245) -- originally developed for AGN jet /
host-galaxy alignment studies -- to the MHD stress tensor.  The idea
is to treat the (B_x, B_y) components at each cell as defining a
local "polarisation angle"

    phi_B = 0.5 * atan2(2 B_x B_y, B_x^2 - B_y^2)

and to ask whether the distribution of phi_B over the box is
isotropic (unstructured turbulence) or concentrated near a preferred
angle (channel-flow dominance).  The diagnostic is the circular
variance

    V = 1 - |< exp(2 i phi_B) >|

which is zero for a perfectly aligned channel flow and one for an
isotropic distribution.
"""

from __future__ import annotations
import math
import numpy as np
from typing import Dict, List

import physical_constants as pc
import boundary_conditions as bc
import mhd_equations as mhd


# ---------------------------------------------------------------------------
#                   Volume averages
# ---------------------------------------------------------------------------
def volume_average(field: np.ndarray, g) -> float:
    """Return the cell-volume-weighted average of ``field``."""
    vol = float(np.mean(g.dx) * np.mean(g.dy) * np.mean(g.dz))
    return float(np.sum(field) * vol / max(g.Nx * g.Ny * g.Nz * vol, 1.0e-30))


def reynolds_stress(U: np.ndarray, g) -> float:
    """Return the volume-averaged Reynolds stress < rho vx' vy' >."""
    ng = 2
    rho = U[bc.ConsIdx.rho, ng:-ng, ng:-ng, ng:-ng]
    vx, vy, _ = mhd.velocity(U[:, ng:-ng, ng:-ng, ng:-ng])
    vx_m = volume_average(rho * vx, g)
    vy_m = volume_average(rho * vy, g)
    rho_m = volume_average(rho, g)
    vx_prime = vx - vx_m / (rho_m + 1.0e-30)
    vy_prime = vy - vy_m / (rho_m + 1.0e-30)
    R_xy = rho * vx_prime * vy_prime
    return volume_average(R_xy, g)


def maxwell_stress(U: np.ndarray, g) -> float:
    """Return the volume-averaged Maxwell stress < -Bx By / (4 pi) >."""
    ng = 2
    Bx = U[bc.ConsIdx.Bx, ng:-ng, ng:-ng, ng:-ng]
    By = U[bc.ConsIdx.By, ng:-ng, ng:-ng, ng:-ng]
    M_xy = - Bx * By / (4.0 * math.pi)
    return volume_average(M_xy, g)


def effective_alpha(U: np.ndarray, g) -> float:
    """Return alpha_SS = (<R_xy> + <M_xy>) / < p >."""
    ng = 2
    p = mhd.pressure(U[:, ng:-ng, ng:-ng, ng:-ng])
    p_mean = volume_average(p, g)
    R = reynolds_stress(U, g)
    M = maxwell_stress(U, g)
    return (R + M) / max(p_mean, 1.0e-30)


# ---------------------------------------------------------------------------
#              Turbulence coherence angle (from 1245_davfer12)
# ---------------------------------------------------------------------------
def magnetic_angle_field(U: np.ndarray, g) -> np.ndarray:
    """Return the local magnetic "polarisation" angle phi_B(x, y, z).

    phi_B = 0.5 atan2(2 Bx By, Bx^2 - By^2)

    The factor 1/2 follows from the spin-2 nature of the stress tensor
    (a 180-degree rotation of the field leaves the stress unchanged).
    """
    ng = 2
    Bx = U[bc.ConsIdx.Bx, ng:-ng, ng:-ng, ng:-ng]
    By = U[bc.ConsIdx.By, ng:-ng, ng:-ng, ng:-ng]
    return 0.5 * np.arctan2(2.0 * Bx * By, Bx**2 - By**2 + 1.0e-30)


def circular_variance_of_angles(U: np.ndarray, g) -> float:
    """Circular variance V = 1 - |< exp(2 i phi_B) >|.

    This is the standard measure of angular concentration used in
    directional statistics (Mardia & Jupp 2000).  A perfectly
    bimodal distribution concentrated at phi_0 and phi_0 + pi gives
    V = 0; an isotropic distribution gives V = 1.
    """
    phi = magnetic_angle_field(U, g)
    z = np.exp(2j * phi)
    R = np.abs(np.mean(z))
    return float(1.0 - R)


def mean_magnetic_angle(U: np.ndarray, g) -> float:
    """Return the mean direction <phi_B> from the first circular moment."""
    phi = magnetic_angle_field(U, g)
    z = np.exp(2j * phi)
    return float(0.5 * np.angle(np.mean(z)))


def angle_histogram(U: np.ndarray, g, n_bins: int = 18) -> np.ndarray:
    """Return the normalised histogram of phi_B in n_bins bins over
    [-pi/2, pi/2]."""
    phi = magnetic_angle_field(U, g).ravel()
    hist, _ = np.histogram(phi, bins=n_bins, range=(-math.pi / 2, math.pi / 2),
                           density=True)
    return hist


# ---------------------------------------------------------------------------
#              Weighted multi-survey style combination (from 1245)
# ---------------------------------------------------------------------------
def weighted_mean_stress(U_list: List[np.ndarray], g,
                         err_list: List[float] = None) -> Dict[str, float]:
    """Combine Maxwell stress from multiple snapshots using inverse-
    variance weighting, following the Astrogeo catalogue combination
    in davfer12 (1245).

    If ``err_list`` is not provided, the errors are estimated as the
    standard deviation across the list.
    """
    stresses = np.array([maxwell_stress(U, g) for U in U_list])
    if err_list is None:
        err = np.std(stresses) + 1.0e-30
        weights = np.ones_like(stresses) / (err**2)
    else:
        err = np.array(err_list) + 1.0e-30
        weights = 1.0 / err**2
    weights /= weights.sum()
    mean = float(np.sum(weights * stresses))
    err_w = float(1.0 / math.sqrt(np.sum(1.0 / err**2)))
    return {"M_xy_weighted": mean, "M_xy_err": err_w, "n_snaps": len(U_list)}


# ---------------------------------------------------------------------------
#                          Diagnostic bundle
# ---------------------------------------------------------------------------
def collect_diagnostics(U: np.ndarray, g, t: float) -> Dict[str, float]:
    """Return a dictionary of all scalar diagnostics at time t."""
    ng = 2
    U_int = U[:, ng:-ng, ng:-ng, ng:-ng]
    p = mhd.pressure(U_int)
    B2 = mhd.total_B2(U_int)
    rho = U_int[bc.ConsIdx.rho]
    return {
        "t":                  t,
        "mean_rho":           volume_average(rho, g),
        "mean_p":             volume_average(p, g),
        "mean_B2":            volume_average(B2, g),
        "mean_beta":          float(np.mean(pc.plasma_beta(p, B2))),
        "R_xy":               reynolds_stress(U, g),
        "M_xy":               maxwell_stress(U, g),
        "alpha_SS":           effective_alpha(U, g),
        "V_circ":             circular_variance_of_angles(U, g),
        "mean_phi_B":         mean_magnetic_angle(U, g),
        "divB_max":           float(_divB_max(U, g)),
    }


def _divB_max(U: np.ndarray, g) -> float:
    """Internal helper: compute max |div B| in the interior."""
    ng = 2
    Bx = U[bc.ConsIdx.Bx, ng:-ng, ng:-ng, ng:-ng]
    By = U[bc.ConsIdx.By, ng:-ng, ng:-ng, ng:-ng]
    Bz = U[bc.ConsIdx.Bz, ng:-ng, ng:-ng, ng:-ng]
    dx_mean = float(np.mean(g.dx))
    dy_mean = float(np.mean(g.dy))
    dz_mean = float(np.mean(g.dz))
    dBx_dx = np.gradient(Bx, dx_mean, axis=0)
    dBy_dy = np.gradient(By, dy_mean, axis=1)
    dBz_dz = np.gradient(Bz, dz_mean, axis=2)
    return float(np.max(np.abs(dBx_dx + dBy_dy + dBz_dz)))
