"""
tokamak_geometry.py — Toroidal (R,Z) grid, flux-surface shaping, magnetic
axis / X-point keypoint detection.

Scientific background
---------------------
A tokamak equilibrium is axisymmetric about the toroidal angle φ, so the
computational domain is a rectangle Ω = [R_min, R_max] × [Z_min, Z_max] in
the poloidal plane. The magnetic axis (R_0, Z_0) is the O-point of ψ where
∇ψ = 0 and ψ is extremal; the X-point(s) are saddle points. We follow the
lightning-pose project (themattinthehatt) and treat both as *keypoints*
detected from the Hessian of ψ:

    H(ψ) = [ ψ_{RR}  ψ_{RZ} ]
           [ ψ_{ZR}  ψ_{ZZ} ]

det H > 0  → O-point (magnetic axis)
det H < 0  → X-point

The boundary shape is prescribed by the Miller parametrisation:

    R(θ) = R_0 + a cos(θ + δ sin θ)
    Z(θ) = κ a sin θ

with inverse aspect ratio ε = a/R_0, triangularity δ, elongation κ.
"""

from __future__ import annotations
import math
from dataclasses import dataclass


@dataclass
class TokamakGeometry:
    """Axisymmetric (R,Z) computational domain and shaping parameters."""
    R0: float = 1.0                 # major radius [m]
    a:  float = 0.33                # minor radius [m]
    kappa: float = 1.7              # elongation
    delta: float = 0.33             # triangularity
    R_min: float = 0.5
    R_max: float = 1.6
    Z_min: float = -0.6
    Z_max: float = 0.6
    Nr: int = 65                    # number of radial grid points (odd → clean symmetry)
    Nz: int = 65                    # number of vertical grid points

    @property
    def eps(self) -> float:
        return self.a / self.R0

    @property
    def dR(self) -> float:
        return (self.R_max - self.R_min) / max(1, self.Nr - 1)

    @property
    def dZ(self) -> float:
        return (self.Z_max - self.Z_min) / max(1, self.Nz - 1)

    def R_grid(self) -> list[float]:
        return [self.R_min + i * self.dR for i in range(self.Nr)]

    def Z_grid(self) -> list[float]:
        return [self.Z_min + j * self.dZ for j in range(self.Nz)]

    # ------------------------------------------------------------------
    # Miller equilibrium boundary: r(θ) = (R(θ), Z(θ))
    # ------------------------------------------------------------------
    def boundary_points(self, n_theta: int = 128) -> list[tuple[float, float]]:
        pts = []
        for k in range(n_theta):
            theta = 2.0 * math.pi * k / n_theta
            R = self.R0 + self.a * math.cos(theta + self.delta * math.sin(theta))
            Z = self.kappa * self.a * math.sin(theta)
            pts.append((R, Z))
        return pts

    # ------------------------------------------------------------------
    # Analytic seed for ψ — a Solov'ev-like starter field
    #   ψ(R,Z) = (R² - R0²)² / (8 R0²) + (Z² - (κ a)²) / (2 κ²)
    # scaled so ψ ≈ 0 on the LCFS, ψ > 0 inside. We use a sign-flipped form
    # so that the axis is the maximum.
    # ------------------------------------------------------------------
    def seed_psi(self, R: float, Z: float) -> float:
        rho2 = ((R - self.R0) / self.a) ** 2
        zeta2 = (Z / (self.kappa * self.a)) ** 2
        s = rho2 + zeta2
        # ψ decreases away from axis (axis = maximum)
        return 1.0 - s + 0.2 * (rho2 - zeta2) * math.cos(0.0) - 0.05 * (rho2 ** 2 + zeta2 ** 2)

    def seed_psi_field(self) -> list[list[float]]:
        Rg = self.R_grid(); Zg = self.Z_grid()
        return [[self.seed_psi(R, Z) for Z in Zg] for R in Rg]


# ---------------------------------------------------------------------------
# Keypoint detection (lightning-pose analog): Hessian-based O/X-point finder
# ---------------------------------------------------------------------------

def hessian(psi: list[list[float]], dR: float, dZ: float,
            i: int, j: int) -> tuple[float, float, float]:
    """Return (ψ_RR, ψ_ZZ, ψ_RZ) at (i,j) using central differences.
    Boundary cells return zero determinant."""
    Nr = len(psi); Nz = len(psi[0])
    if i <= 0 or i >= Nr - 1 or j <= 0 or j >= Nz - 1:
        return 0.0, 0.0, 0.0
    psiRR = (psi[i + 1][j] - 2.0 * psi[i][j] + psi[i - 1][j]) / (dR * dR)
    psiZZ = (psi[i][j + 1] - 2.0 * psi[i][j] + psi[i][j - 1]) / (dZ * dZ)
    psiRZ = (psi[i + 1][j + 1] - psi[i + 1][j - 1]
             - psi[i - 1][j + 1] + psi[i - 1][j - 1]) / (4.0 * dR * dZ)
    return psiRR, psiZZ, psiRZ


def find_keypoints(psi: list[list[float]], geom: TokamakGeometry
                   ) -> dict:
    """Detect magnetic axis (O-point) and X-point(s) via Hessian criterion.
    Returns dict with 'axis' = (R, Z, ψ), 'xpoints' = [(R, Z, ψ), ...]."""
    Rg = geom.R_grid(); Zg = geom.Z_grid()
    dR, dZ = geom.dR, geom.dZ
    axis = None
    xpoints: list[tuple[float, float, float]] = []
    # Restrict search to interior cells within a box near the expected axis
    R_lo = geom.R0 - 0.6 * geom.a
    R_hi = geom.R0 + 0.6 * geom.a
    for i, R in enumerate(Rg):
        if R < R_lo or R > R_hi:
            continue
        for j, Z in enumerate(Zg):
            if abs(Z) > 0.8 * geom.kappa * geom.a:
                continue
            psiRR, psiZZ, psiRZ = hessian(psi, dR, dZ, i, j)
            det = psiRR * psiZZ - psiRZ * psiRZ
            val = psi[i][j]
            if det > 0.0 and psiRR < 0.0:
                # O-point candidate (local maximum)
                if axis is None or val > axis[2]:
                    axis = (R, Z, val)
            elif det < 0.0:
                xpoints.append((R, Z, val))
    # De-duplicate close X-points
    if xpoints:
        xpoints.sort(key=lambda p: p[2])
        filtered = [xpoints[0]]
        for p in xpoints[1:]:
            if abs(p[0] - filtered[-1][0]) + abs(p[1] - filtered[-1][1]) > 0.05:
                filtered.append(p)
        xpoints = filtered
    return {"axis": axis, "xpoints": xpoints}
