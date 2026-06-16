"""
pic_reconstruction.py — Particle-in-Cell reconstruction of the toroidal
current density j_φ and charge density ρ on the (R,Z) grid, mapped from the
electrostatic PIC code of the 869_pic project.

Scientific background
---------------------
Once ψ(R,Z) is known, the MHD force balance gives

    j_φ = - (1/μ₀) Δ* ψ / R                              (1)

We treat j_φ as the "charge density" in a PIC analogy: each computational
particle carries a weight w_k = j_φ(R_k, Z_k) ΔV_k and deposits onto the
grid via bilinear shape functions S_i(R) S_j(Z), exactly as in the ES-PIC
method of Brieda (2013).  This is useful for (i) verifying consistency of
the GS solution, (ii) providing the source for a coupled kinetic model, and
(iii) estimating the fraction of bootstrap vs. Ohmic current.
"""

from __future__ import annotations
import math
from dataclasses import dataclass


MU0 = 4.0 * math.pi * 1e-7


@dataclass
class PICGrid:
    """Deposition grid mirroring the (R,Z) mesh."""
    R_min: float
    R_max: float
    Z_min: float
    Z_max: float
    Nr: int
    Nz: int

    @property
    def dR(self) -> float:
        return (self.R_max - self.R_min) / max(1, self.Nr - 1)

    @property
    def dZ(self) -> float:
        return (self.Z_max - self.Z_min) / max(1, self.Nz - 1)


def laplacian_star(psi: list[list[float]], dR: float, dZ: float,
                   R: float, i: int, j: int) -> float:
    """Compute Δ*ψ at (i,j) using 2nd-order FD:
    Δ*ψ = ψ_RR - ψ_R / R + ψ_ZZ."""
    psiRR = (psi[i + 1][j] - 2.0 * psi[i][j] + psi[i - 1][j]) / (dR * dR)
    psiZZ = (psi[i][j + 1] - 2.0 * psi[i][j] + psi[i][j - 1]) / (dZ * dZ)
    psiR = (psi[i + 1][j] - psi[i - 1][j]) / (2.0 * dR)
    return psiRR - psiR / max(R, 1e-8) + psiZZ


def compute_jtor(psi: list[list[float]], R_grid: list[float], dR: float, dZ: float
                 ) -> list[list[float]]:
    """Return j_φ(R,Z) = -Δ*ψ / (μ₀ R)."""
    Nr = len(psi); Nz = len(psi[0])
    jt = [[0.0] * Nz for _ in range(Nr)]
    for i in range(1, Nr - 1):
        R = R_grid[i]
        for j in range(1, Nz - 1):
            L = laplacian_star(psi, dR, dZ, R, i, j)
            jt[i][j] = -L / (MU0 * max(R, 1e-8))
    return jt


# ---------------------------------------------------------------------------
# PIC-like deposition from "particles" sampled on the plasma
# ---------------------------------------------------------------------------

def sample_particles(jt: list[list[float]], R_grid: list[float], Z_grid: list[float],
                     n_particles: int = 5000) -> list[tuple[float, float, float]]:
    """Draw (R, Z, weight) particles proportional to |j_φ|.
    We use deterministic low-discrepancy sampling with rejection."""
    import random
    random.seed(42)
    R_min, R_max = R_grid[0], R_grid[-1]
    Z_min, Z_max = Z_grid[0], Z_grid[-1]
    Nr = len(R_grid); Nz = len(Z_grid)
    jt_max = max(abs(jt[i][j]) for i in range(Nr) for j in range(Nz)) or 1.0
    pts = []
    attempts = 0
    max_attempts = 50 * n_particles
    while len(pts) < n_particles and attempts < max_attempts:
        attempts += 1
        R = R_min + random.random() * (R_max - R_min)
        Z = Z_min + random.random() * (Z_max - Z_min)
        # Find cell
        i = int((R - R_min) / ((R_max - R_min) / (Nr - 1)))
        j = int((Z - Z_min) / ((Z_max - Z_min) / (Nz - 1)))
        i = max(1, min(Nr - 2, i))
        j = max(1, min(Nz - 2, j))
        w = abs(jt[i][j]) / jt_max
        if random.random() < w:
            pts.append((R, Z, jt[i][j]))
    return pts


def deposit_particles(particles: list[tuple[float, float, float]],
                      grid: PICGrid) -> list[list[float]]:
    """Bilinear PIC deposition: ρ_{ij} = Σ_k w_k S_i(R_k) S_j(Z_k) / ΔV.
    ΔV = 2π R ΔR ΔZ (toroidal volume of a cell)."""
    rho = [[0.0] * grid.Nz for _ in range(grid.Nr)]
    dR, dZ = grid.dR, grid.dZ
    for (R, Z, w) in particles:
        # Bi-linear weights
        u = (R - grid.R_min) / dR
        v = (Z - grid.Z_min) / dZ
        i = int(u); j = int(v)
        i = max(0, min(grid.Nr - 2, i))
        j = max(0, min(grid.Nz - 2, j))
        du = u - i; dv = v - j
        Rmid = grid.R_min + (i + 0.5) * dR
        dV = 2.0 * math.pi * Rmid * dR * dZ
        weights = [(1 - du) * (1 - dv), du * (1 - dv),
                   (1 - du) * dv,       du * dv]
        idx = [(i, j), (i + 1, j), (i, j + 1), (i + 1, j + 1)]
        for (ii, jj), ww in zip(idx, weights):
            if 0 <= ii < grid.Nr and 0 <= jj < grid.Nz:
                rho[ii][jj] += w * ww / max(dV, 1e-30)
    return rho


def bootstrap_fraction(jt: list[list[float]], R_grid: list[float], Z_grid: list[float],
                       f_bs: float = 0.55) -> float:
    """Estimate I_bs / I_total ≈ ∫ j_bs dA / ∫ j_φ dA.
    We assume a model where j_bs = f_bs · j_φ inside the plasma."""
    dR = R_grid[1] - R_grid[0]
    dZ = Z_grid[1] - Z_grid[0]
    I_total = 0.0
    I_bs = 0.0
    Nr = len(R_grid); Nz = len(Z_grid)
    for i in range(1, Nr - 1):
        R = R_grid[i]
        for j in range(1, Nz - 1):
            if abs(jt[i][j]) > 1e-4 * max(abs(x) for row in jt for x in row) \
                    if max(abs(x) for row in jt for x in row) > 0 else False:
                dA = 2.0 * math.pi * R * dR * dZ
                I_total += jt[i][j] * dA
                I_bs += f_bs * jt[i][j] * dA
    return I_bs / I_total if abs(I_total) > 1e-30 else 0.0
