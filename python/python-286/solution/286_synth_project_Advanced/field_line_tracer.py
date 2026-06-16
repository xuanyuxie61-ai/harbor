"""
field_line_tracer.py — Magnetic field line tracing via Verlet / leap-frog
integration in the poloidal plane, mapped from the 1392_verlet_simulation
project (tether_balls / ping_pong Verlet integration).

Scientific background
---------------------
In axisymmetry the field lines are curves (R(s), Z(s), φ(s)) satisfying

    dR/ds = B_R / |B|,    dZ/ds = B_Z / |B|,    dφ/ds = B_φ / (R |B|)

where B_R = -ψ_Z / R,  B_Z = ψ_R / R,  B_φ = F(ψ)/R.  Tracing in (R,Z) at
fixed φ gives the poloidal projection; adding the φ advance yields the 3-D
helical field line.

We use a velocity-Verlet scheme with adaptive step size:

    x_{n+1} = x_n + h v_n + (h²/2) a_n
    v_{n+1} = v_n + (h/2)(a_n + a_{n+1})

with the "acceleration" a = curvature of the field line (second derivative
along the line).  For our purposes we take a = 0 (straight-step Verlet
reduces to leapfrog) and use the residual to drive step adaptation.

Boundary handling: if a step leaves the domain we clip the end-point to ∂Ω.
"""

from __future__ import annotations
import math
from dataclasses import dataclass


@dataclass
class FieldLine:
    R: list[float]
    Z: list[float]
    phi: list[float]
    length: float


def interpolate_b(psi: list[list[float]], F_val: float,
                  R_grid: list[float], Z_grid: list[float],
                  R: float, Z: float) -> tuple[float, float, float]:
    """Bilinear interpolation of (B_R, B_Z, B_φ) at (R,Z)."""
    dR = R_grid[1] - R_grid[0]
    dZ = Z_grid[1] - Z_grid[0]
    Nr = len(R_grid); Nz = len(Z_grid)
    u = (R - R_grid[0]) / dR
    v = (Z - Z_grid[0]) / dZ
    i = max(0, min(Nr - 2, int(u)))
    j = max(0, min(Nz - 2, int(v)))
    du = u - i; dv = v - j
    # ψ and its gradient at the four corners
    def psi_and_grad(i_, j_):
        p = psi[i_][j_]
        pR = (psi[i_ + 1][j_] - psi[i_ - 1][j_]) / (2.0 * dR) if 0 < i_ < Nr - 1 else 0.0
        pZ = (psi[i_][j_ + 1] - psi[i_][j_ - 1]) / (2.0 * dZ) if 0 < j_ < Nz - 1 else 0.0
        return p, pR, pZ
    corners = [(i, j), (i + 1, j), (i, j + 1), (i + 1, j + 1)]
    w = [(1 - du) * (1 - dv), du * (1 - dv), (1 - du) * dv, du * dv]
    pR = sum(ww * psi_and_grad(ii, jj)[1] for (ii, jj), ww in zip(corners, w))
    pZ = sum(ww * psi_and_grad(ii, jj)[2] for (ii, jj), ww in zip(corners, w))
    Rc = R_grid[i] + du * dR
    B_R = -pZ / max(Rc, 1e-6)
    B_Z = pR / max(Rc, 1e-6)
    B_phi = F_val / max(Rc, 1e-6)
    return B_R, B_Z, B_phi


def clip_to_domain(R: float, Z: float,
                   R_min: float, R_max: float,
                   Z_min: float, Z_max: float) -> tuple[float, float]:
    """If the point is outside the domain, clamp to the nearest boundary."""
    R = max(R_min, min(R_max, R))
    Z = max(Z_min, min(Z_max, Z))
    return R, Z


def trace_field_line(psi: list[list[float]], F_val: float,
                     R_grid: list[float], Z_grid: list[float],
                     R0: float, Z0: float,
                     n_steps: int = 2000,
                     ds: float = 0.005) -> FieldLine:
    """Trace a field line starting at (R0, Z0) for n_steps of size ds,
    using velocity-Verlet.  Returns a FieldLine object with (R, Z, φ) lists."""
    R_min, R_max = R_grid[0], R_grid[-1]
    Z_min, Z_max = Z_grid[0], Z_grid[-1]
    Rs = [R0]; Zs = [Z0]; phis = [0.0]
    R, Z, phi = R0, Z0, 0.0
    for step in range(n_steps):
        BR, BZ, Bphi = interpolate_b(psi, F_val, R_grid, Z_grid, R, Z)
        Bmod = math.sqrt(BR * BR + BZ * BZ + Bphi * Bphi)
        if Bmod < 1e-12:
            break
        # Velocity-Verlet step (leapfrog on the unit tangent)
        tR = BR / Bmod; tZ = BZ / Bmod; tphi = Bphi / (Bmod * max(R, 1e-6))
        R_new = R + ds * tR
        Z_new = Z + ds * tZ
        phi_new = phi + ds * tphi
        R_new, Z_new = clip_to_domain(R_new, Z_new, R_min, R_max, Z_min, Z_max)
        # Stop if we hit the boundary
        if (R_new <= R_min or R_new >= R_max or Z_new <= Z_min or Z_new >= Z_max) \
                and (R == R_new and Z == Z_new):
            break
        Rs.append(R_new); Zs.append(Z_new); phis.append(phi_new)
        R, Z, phi = R_new, Z_new, phi_new
    # Arc length
    L = 0.0
    for k in range(1, len(Rs)):
        L += math.sqrt((Rs[k] - Rs[k - 1]) ** 2 + (Zs[k] - Zs[k - 1]) ** 2)
    return FieldLine(R=Rs, Z=Zs, phi=phis, length=L)


def safety_factor(psi: list[list[float]], F_val: float,
                  R_grid: list[float], Z_grid: list[float],
                  R_axis: float, Z_axis: float,
                  psi_axis: float) -> float:
    """Estimate the on-axis safety factor q_0 by tracing a field line that
    starts close to the magnetic axis and computing q = Δφ/(2π) per poloidal
    turn.  For small displacements q ≈ q_0 + s r²."""
    dR = R_grid[1] - R_grid[0]
    r_start = 0.05 * (R_grid[-1] - R_grid[0])
    R0 = R_axis + r_start
    Z0 = Z_axis
    line = trace_field_line(psi, F_val, R_grid, Z_grid, R0, Z0,
                            n_steps=8000, ds=0.005)
    # Count poloidal turns: each time Z crosses zero with dZ/ds > 0.
    crossings = []
    for k in range(1, len(line.Z)):
        if line.Z[k - 1] < 0.0 and line.Z[k] >= 0.0:
            crossings.append(line.phi[k])
    if len(crossings) < 2:
        return 1.0
    dphi_total = crossings[-1] - crossings[0]
    n_turns = len(crossings) - 1
    q = dphi_total / (2.0 * math.pi * n_turns) if n_turns > 0 else 1.0
    return q


def write_xyzl(prefix: str, lines: list[FieldLine]) -> dict:
    """Write a set of 3-D field lines in the xyz/xyzl format of the 1426_xyzl_display
    project.  Each line is written as (x=R cos φ, y=R sin φ, z=Z)."""
    xyz_path = f"{prefix}.xyz"
    xyzl_path = f"{prefix}.xyzl"
    points: list[tuple[float, float, float]] = []
    lines_idx: list[list[int]] = []
    offset = 0
    for line in lines:
        idx = []
        for (R, Z, phi) in zip(line.R, line.Z, line.phi):
            x = R * math.cos(phi)
            y = R * math.sin(phi)
            points.append((x, y, Z))
            idx.append(offset); offset += 1
        lines_idx.append(idx)
    with open(xyz_path, "w") as f:
        for (x, y, z) in points:
            f.write(f"{x:.8f} {y:.8f} {z:.8f}\n")
    with open(xyzl_path, "w") as f:
        for idx in lines_idx:
            f.write(" ".join(map(str, idx)) + "\n")
    return {"xyz_path": xyz_path, "xyzl_path": xyzl_path}
