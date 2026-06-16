"""
boundary_reconstruction.py — Equilibrium boundary and geometric-quantity
reconstruction, inspired by the lightning-pose keypoint-detection framework
(1036_themattinthehatt_lightning-pose-2024-nat-methods).

Scientific background
---------------------
We treat the magnetic axis, X-point(s) and a set of boundary keypoints as
"pose keypoints" to be estimated from the poloidal-flux field ψ(R,Z).  The
LCFS (last closed flux surface) is the outermost closed contour of ψ passing
through the X-point (if present) or defined by a user-supplied ψ_bnd.

Keypoint regression:
    We fit a 2-D quadratic around each detected extremum / saddle of ψ
    (from Hessian analysis in tokamak_geometry) to sub-grid refinement:

        ψ(R,Z) ≈ ψ₀ + a (R - R₀)² + 2 c (R - R₀)(Z - Z₀) + b (Z - Z₀)²

    The refined position is
        (R*, Z*) = (R₀, Z₀) - H^{-1} ∇ψ   (one Newton step)               (1)

Geometric outputs
    • R_axis, Z_axis           — magnetic axis
    • R_X, Z_X                 — lower X-point (if diverted)
    • a_geo, κ_geo, δ_geo      — geometric minor radius, elongation, triangularity
    • volume, surface_area     — plasma volume and LCFS area
    • q_cyl                    — cylindrical safety factor q_cyl = (a² B₀)/(μ₀ I_p R₀)
"""

from __future__ import annotations
import math
from dataclasses import dataclass


@dataclass
class EquilibriumGeometry:
    R_axis: float
    Z_axis: float
    psi_axis: float
    R_X: float | None
    Z_X: float | None
    a_geo: float
    kappa_geo: float
    delta_geo: float
    volume: float
    surface_area: float
    q_cyl: float


def refine_keypoint(psi: list[list[float]], dR: float, dZ: float,
                    R_grid: list[float], Z_grid: list[float],
                    i: int, j: int) -> tuple[float, float, float]:
    """Newton-refined location of a critical point of ψ.  Uses central
    differences for gradient and Hessian."""
    Nr = len(psi); Nz = len(psi[0])
    if i <= 0 or i >= Nr - 1 or j <= 0 or j >= Nz - 1:
        return R_grid[i], Z_grid[j], psi[i][j]
    pR = (psi[i + 1][j] - psi[i - 1][j]) / (2.0 * dR)
    pZ = (psi[i][j + 1] - psi[i][j - 1]) / (2.0 * dZ)
    pRR = (psi[i + 1][j] - 2.0 * psi[i][j] + psi[i - 1][j]) / (dR * dR)
    pZZ = (psi[i][j + 1] - 2.0 * psi[i][j] + psi[i][j - 1]) / (dZ * dZ)
    pRZ = (psi[i + 1][j + 1] - psi[i + 1][j - 1]
           - psi[i - 1][j + 1] + psi[i - 1][j - 1]) / (4.0 * dR * dZ)
    det = pRR * pZZ - pRZ * pRZ
    if abs(det) < 1e-15:
        return R_grid[i], Z_grid[j], psi[i][j]
    dR_step = -(pZZ * pR - pRZ * pZ) / det
    dZ_step = -(pRR * pZ - pRZ * pR) / det
    # Clamp step to half a cell to avoid runaway
    dR_step = max(-0.5 * dR, min(0.5 * dR, dR_step))
    dZ_step = max(-0.5 * dZ, min(0.5 * dZ, dZ_step))
    R_new = R_grid[i] + dR_step
    Z_new = Z_grid[j] + dZ_step
    # Quadratic interpolation of ψ at the new point
    psi_new = psi[i][j] + 0.5 * (pR * dR_step + pZ * dZ_step)
    return R_new, Z_new, psi_new


def reconstruct_geometry(psi: list[list[float]], jt: list[list[float]],
                         R_grid: list[float], Z_grid: list[float],
                         R0_input: float, B0: float,
                         Ip: float) -> EquilibriumGeometry:
    """Build a complete EquilibriumGeometry from ψ and j_φ."""
    try:
        from .tokamak_geometry import find_keypoints, TokamakGeometry
    except ImportError:
        from tokamak_geometry import find_keypoints, TokamakGeometry
    dR = R_grid[1] - R_grid[0]
    dZ = Z_grid[1] - Z_grid[0]
    Nr = len(R_grid); Nz = len(Z_grid)
    geom = TokamakGeometry(R0=R0_input, R_min=R_grid[0], R_max=R_grid[-1],
                           Z_min=Z_grid[0], Z_max=Z_grid[-1],
                           Nr=Nr, Nz=Nz)
    keys = find_keypoints(psi, geom)
    axis = keys["axis"]
    if axis is None:
        # Fallback: take maximum of ψ
        imax = jmax = 0
        vmax = psi[0][0]
        for i in range(Nr):
            for j in range(Nz):
                if psi[i][j] > vmax:
                    vmax = psi[i][j]; imax = i; jmax = j
        axis = (R_grid[imax], Z_grid[jmax], vmax)
    R_axis, Z_axis, psi_axis = refine_keypoint(psi, dR, dZ, R_grid, Z_grid,
                                               int((axis[0] - R_grid[0]) / dR),
                                               int((axis[1] - Z_grid[0]) / dZ))
    # X-point
    R_X = Z_X = None
    if keys["xpoints"]:
        xpt = keys["xpoints"][0]
        R_X, Z_X, _ = refine_keypoint(psi, dR, dZ, R_grid, Z_grid,
                                      int((xpt[0] - R_grid[0]) / dR),
                                      int((xpt[1] - Z_grid[0]) / dZ))
    # Geometric a, κ, δ from the ψ = 0.05 ψ_axis contour
    target = 0.05 * psi_axis
    R_out = R_grid[0]; R_in = R_grid[-1]
    Z_top = 0.0
    for i in range(Nr - 1):
        for j in range(Nz - 1):
            v00 = psi[i][j]; v10 = psi[i + 1][j]
            v01 = psi[i][j + 1]; v11 = psi[i + 1][j + 1]
            if (v00 - target) * (v10 - target) < 0:
                t = (target - v00) / (v10 - v00 + 1e-30)
                R = R_grid[i] + t * (R_grid[i + 1] - R_grid[i])
                Z = Z_grid[j]
                if R > R_out: R_out = R
                if R < R_in: R_in = R
            if (v00 - target) * (v01 - target) < 0:
                t = (target - v00) / (v01 - v00 + 1e-30)
                R = R_grid[i]
                Z = Z_grid[j] + t * (Z_grid[j + 1] - Z_grid[j])
                if Z > Z_top: Z_top = Z
    a_geo = 0.5 * (R_out - R_in)
    R_geo = 0.5 * (R_out + R_in)
    kappa_geo = Z_top / a_geo if a_geo > 1e-6 else 1.0
    # Triangularity: the R-shift of the Z_top point relative to R_geo, divided by a
    # Find the R coordinate at the highest Z on the contour
    R_top = R_geo
    best_Z = -1.0
    for i in range(Nr - 1):
        for j in range(Nz - 1):
            v00 = psi[i][j]; v01 = psi[i][j + 1]
            if (v00 - target) * (v01 - target) < 0:
                t = (target - v00) / (v01 - v00 + 1e-30)
                R = R_grid[i]
                Z = Z_grid[j] + t * (Z_grid[j + 1] - Z_grid[j])
                if Z > best_Z:
                    best_Z = Z
                    R_top = R
    delta_geo = (R_top - R_geo) / max(a_geo, 1e-6)
    # Volume and area (approximate)
    volume = math.pi * 2.0 * math.pi * R_geo * a_geo ** 2 * kappa_geo
    surface_area = 4.0 * math.pi ** 2 * R_geo * a_geo * math.sqrt(0.5 * (1 + kappa_geo ** 2))
    # Cylindrical safety factor
    mu0 = 4.0 * math.pi * 1e-7
    q_cyl = (a_geo ** 2 * B0) / (mu0 * abs(Ip) * R_geo + 1e-30) if abs(Ip) > 1e-10 else 1.0
    return EquilibriumGeometry(
        R_axis=R_axis, Z_axis=Z_axis, psi_axis=psi_axis,
        R_X=R_X, Z_X=Z_X,
        a_geo=a_geo, kappa_geo=kappa_geo, delta_geo=delta_geo,
        volume=volume, surface_area=surface_area, q_cyl=q_cyl)
