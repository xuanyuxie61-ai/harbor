"""
flux_surface_integrals.py — Surface and volume integrals over flux surfaces
and the toroidal domain, mapped from the 234_cube_integrals project (monomial
integrals over [0,1]³).

Scientific background
---------------------
In toroidal geometry the natural integration variables are (ψ, θ, φ) where
θ is a poloidal angle.  The Jacobian of the transformation (R,Z,φ) → (ψ,θ,φ)
is

    J = (∂(R,Z)/∂(ψ,θ)) R = R / |∇ψ|²                            (1)

after choosing the equal-arc poloidal angle.  A flux-surface average of a
quantity Q is

    <Q> = (1/V') ∫∫ Q / |∇ψ|²  dθ                                (2)

with V'(ψ) = ∮ R / |∇ψ| dθ the flux-surface volume derivative.  The total
toroidal volume integral is

    I = ∫_0^{ψ_a} dψ ∫_0^{2π} dφ ∫_0^{2π} dθ  J(ψ,θ) Q(ψ,θ)     (3)

which reduces to a sum of 1-D monomial-like integrals along θ at each ψ.
"""

from __future__ import annotations
import math


def grad_psi_sq(psi: list[list[float]], dR: float, dZ: float,
                i: int, j: int) -> float:
    """|∇ψ|² at (i,j) via central differences."""
    pR = (psi[i + 1][j] - psi[i - 1][j]) / (2.0 * dR)
    pZ = (psi[i][j + 1] - psi[i][j - 1]) / (2.0 * dZ)
    return pR * pR + pZ * pZ


def flux_surface_integral(psi: list[list[float]], Q: list[list[float]],
                          R_grid: list[float], Z_grid: list[float],
                          psi_level: float, tol: float = 0.02) -> dict:
    """Compute ∮_{ψ=const} Q / |∇ψ|² dθ on the level set ψ = psi_level
    using linear interpolation between grid values.
    Returns {'integral': float, 'length': float, 'n_points': int}."""
    dR = R_grid[1] - R_grid[0]
    dZ = Z_grid[1] - Z_grid[0]
    Nr = len(R_grid); Nz = len(Z_grid)
    # Collect all cell-edge crossings where ψ crosses psi_level
    crossings: list[tuple[float, float, float]] = []   # (R, Z, integrand)
    for i in range(Nr - 1):
        for j in range(Nz - 1):
            corners = [(i, j), (i + 1, j), (i, j + 1), (i + 1, j + 1)]
            vals = [psi[ii][jj] for (ii, jj) in corners]
            # Check sign changes along edges
            edges = [(0, 1), (1, 3), (3, 2), (2, 0)]
            for (a, b) in edges:
                va, vb = vals[a], vals[b]
                if (va - psi_level) * (vb - psi_level) >= 0:
                    continue
                t = (psi_level - va) / (vb - va + 1e-30)
                ia, ja = corners[a]; ib, jb = corners[b]
                Ra = R_grid[ia]; Rb = R_grid[ib]
                Za = Z_grid[ja]; Zb = Z_grid[jb]
                R = Ra + t * (Rb - Ra)
                Z = Za + t * (Zb - Za)
                # Interpolate Q
                Qa = Q[ia][ja]; Qb = Q[ib][jb]
                Qv = Qa + t * (Qb - Qa)
                # |∇ψ|² at crossing: average of adjacent nodes
                i_mid = max(1, min(Nr - 2, (ia + ib) // 2))
                j_mid = max(1, min(Nz - 2, (ja + jb) // 2))
                g2 = max(grad_psi_sq(psi, dR, dZ, i_mid, j_mid), 1e-20)
                crossings.append((R, Z, Qv / g2))
    if len(crossings) < 4:
        return {"integral": 0.0, "length": 0.0, "n_points": 0}
    # Sort crossings by angle around the centroid
    Rc = sum(c[0] for c in crossings) / len(crossings)
    Zc = sum(c[1] for c in crossings) / len(crossings)
    crossings.sort(key=lambda c: math.atan2(c[1] - Zc, c[0] - Rc))
    # Trapezoidal line integral: ∮ (Q/|∇ψ|²) |dx|
    integral = 0.0; length = 0.0
    n = len(crossings)
    for k in range(n):
        R1, Z1, I1 = crossings[k]
        R2, Z2, I2 = crossings[(k + 1) % n]
        dl = math.sqrt((R2 - R1) ** 2 + (Z2 - Z1) ** 2)
        integral += 0.5 * (I1 + I2) * dl
        length += dl
    return {"integral": integral, "length": length, "n_points": n}


def volume_integral(psi: list[list[float]], Q: list[list[float]],
                    R_grid: list[float], Z_grid: list[float],
                    n_levels: int = 20) -> float:
    """Compute the toroidal volume integral ∫∫∫ Q(R,Z) R dR dZ dφ = 2π ∫∫ Q R dR dZ
    on the region ψ > 0 (inside the LCFS)."""
    dR = R_grid[1] - R_grid[0]
    dZ = Z_grid[1] - Z_grid[0]
    Nr = len(R_grid); Nz = len(Z_grid)
    I = 0.0
    for i in range(1, Nr - 1):
        R = R_grid[i]
        for j in range(1, Nz - 1):
            if psi[i][j] > 0.0:
                I += Q[i][j] * R * dR * dZ
    return 2.0 * math.pi * I


def monomial_torus_integral(p: int, q: int, r: int) -> float:
    """Analytic toroidal monomial integral on the unit cube (0,1)³,
    mirroring the cube01_monomial_integral of the seed project:
    ∫_0^1 ∫_0^1 ∫_0^1 x^p y^q z^r dx dy dz = 1/((p+1)(q+1)(r+1)).
    In the tokamak context this is the normalised (ρ, θ̃, φ̃) integral of
    ρ^p θ̃^q φ̃^r used to benchmark the numerical flux-surface routines."""
    return 1.0 / ((p + 1) * (q + 1) * (r + 1))


def plasma_volume(psi: list[list[float]], R_grid: list[float], Z_grid: list[float]) -> float:
    """Plasma volume V = 2π ∫∫_{ψ>0} R dR dZ."""
    Q = [[1.0] * len(Z_grid) for _ in range(len(R_grid))]
    return volume_integral(psi, Q, R_grid, Z_grid)


def plasma_current(jt: list[list[float]], R_grid: list[float], Z_grid: list[float]) -> float:
    """Total toroidal plasma current I_p = ∫∫ j_φ dR dZ."""
    dR = R_grid[1] - R_grid[0]
    dZ = Z_grid[1] - Z_grid[0]
    Nr = len(R_grid); Nz = len(Z_grid)
    Ip = 0.0
    for i in range(1, Nr - 1):
        for j in range(1, Nz - 1):
            Ip += jt[i][j] * dR * dZ
    return Ip
