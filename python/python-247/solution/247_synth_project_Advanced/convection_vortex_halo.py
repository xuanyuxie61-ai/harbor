"""
convection_vortex_halo.py
=========================
Convection-driven secondary circulation in the hot halo gas, leading
to angular-momentum intensification of the central dark matter
distribution (Tseng & Shao 2024 analogy).

Physical model
--------------
The hot circum-galactic medium (CGM) inside a massive halo develops
a convection-driven secondary circulation when the radiative cooling
time becomes comparable to the dynamical time.  In cylindrical
coordinates (R, phi, z) the axisymmetric vorticity equation reads

    d omega/dt + (u . grad) omega = (omega . grad) u
                                    + nu laplacian omega
                                    + S_buoy,

where the buoyancy source

    S_buoy = (g / c_s^2) d <T>/dR

is proportional to the radial entropy gradient.  The secondary
circulation transports angular momentum inward, intensifying the
central vortex and indirectly steepening the dark matter density
cusp (adiabatic contraction).

We solve a reduced 2D (R, z) vorticity-streamfunction system on a
uniform grid using a high-order (5th-order) upwind advection scheme
for the nonlinear term and a standard Laplacian for viscosity.
"""

from __future__ import annotations
import math
from typing import Tuple

import numpy as np


# ---------- Grid and parameters ----------------------------------------------

class VortexGrid:
    """Uniform (R, z) grid for the axisymmetric halo cross-section."""

    def __init__(self, nR: int = 33, nz: int = 33,
                 Rmax: float = 1.0, zmax: float = 1.0):
        self.nR = nR
        self.nz = nz
        self.Rmax = Rmax
        self.zmax = zmax
        self.dR = Rmax / max(nR - 1, 1)
        self.dz = zmax / max(nz - 1, 1)
        self.R = np.linspace(0.0, Rmax, nR)
        self.z = np.linspace(-zmax, zmax, nz)


class VortexParams:
    def __init__(self, nu: float = 1e-3, g_eff: float = 1.0,
                 cs2: float = 1.0, dT_dR: float = -0.5):
        self.nu = nu
        self.g_eff = g_eff
        self.cs2 = cs2
        self.dT_dR = dT_dR


# ---------- High-order upwind advection --------------------------------------

def upwind_5th(f: np.ndarray, axis: int, dx: float,
               velocity_sign: int) -> np.ndarray:
    """5th-order upwind finite-difference derivative along ``axis``.

    For velocity_sign > 0 (flow in +x direction) we use the
    backward-biased stencil

        f' ~= ( 2 f_{i-3} - 15 f_{i-2} + 60 f_{i-1}
                - 20 f_i - 30 f_{i+1} + 3 f_{i+2} ) / 60 dx

    and its mirror for velocity_sign < 0.  Near boundaries we fall
    back to lower-order one-sided differences."""
    n = f.shape[axis]
    out = np.zeros_like(f)
    slc = [slice(None)] * f.ndim
    for i in range(n):
        if velocity_sign >= 0:
            # backward-biased 5th order if possible
            im3 = max(i - 3, 0); im2 = max(i - 2, 0); im1 = max(i - 1, 0)
            ip1 = min(i + 1, n - 1); ip2 = min(i + 2, n - 1)
            slc[axis] = i
            num = (2 * _take(f, axis, im3) - 15 * _take(f, axis, im2)
                   + 60 * _take(f, axis, im1) - 20 * _take(f, axis, i)
                   - 30 * _take(f, axis, ip1) + 3 * _take(f, axis, ip2))
            out[tuple(slc)] = num / (60.0 * dx)
        else:
            im2 = min(i + 2, n - 1); im1 = min(i + 1, n - 1)
            ip1 = max(i - 1, 0); ip2 = max(i - 2, 0); ip3 = max(i - 3, 0)
            slc[axis] = i
            num = (-2 * _take(f, axis, im2) + 15 * _take(f, axis, im1)
                   - 60 * _take(f, axis, i) + 20 * _take(f, axis, ip1)
                   + 30 * _take(f, axis, ip2) - 3 * _take(f, axis, ip3))
            out[tuple(slc)] = num / (60.0 * dx)
    return out


def _take(a: np.ndarray, axis: int, i: int) -> np.ndarray:
    slc = [slice(None)] * a.ndim
    slc[axis] = i
    return a[tuple(slc)]


# ---------- Laplacian --------------------------------------------------------

def laplacian_2d(f: np.ndarray, dR: float, dz: float) -> np.ndarray:
    """Standard 5-point Laplacian on a uniform (R, z) grid."""
    lap = np.zeros_like(f)
    lap[1:-1, 1:-1] = (
        (f[2:, 1:-1] - 2 * f[1:-1, 1:-1] + f[:-2, 1:-1]) / dR ** 2 +
        (f[1:-1, 2:] - 2 * f[1:-1, 1:-1] + f[1:-1, :-2]) / dz ** 2
    )
    return lap


# ---------- Buoyancy source --------------------------------------------------

def buoyancy_source(grid: VortexGrid, params: VortexParams) -> np.ndarray:
    S = np.zeros((grid.nR, grid.nz))
    S[:, :] = (params.g_eff / params.cs2) * params.dT_dR
    # vorticity must vanish on axis R = 0
    S[0, :] = 0.0
    return S


# ---------- Time integrator --------------------------------------------------

def advance_vortex(grid: VortexGrid, params: VortexParams,
                   omega: np.ndarray, dt: float,
                   n_steps: int) -> np.ndarray:
    """Integrate the vorticity equation forward by n_steps explicit
    Euler steps (with a CFL limiter)."""
    S = buoyancy_source(grid, params)
    u_R = np.zeros_like(omega)
    u_z = np.zeros_like(omega)
    # crude streamfunction proxy: u_R = - dPsi/dz, u_z = (1/R) dPsi/dR
    # we approximate Psi ~ omega so u_R ~ -omega, u_z ~ 0
    u_R = -omega
    for _ in range(n_steps):
        # CFL limiter
        umax = max(float(np.max(np.abs(u_R))) / grid.dR,
                   1e-9)
        dt_safe = min(dt, 0.4 / (umax + params.nu / grid.dR ** 2 + 1e-9))
        advR = u_R * upwind_5th(omega, axis=0, dx=grid.dR,
                                velocity_sign=np.sign(u_R.mean()) or 1)
        adpz = np.zeros_like(omega)   # u_z = 0 in this proxy model
        visc = params.nu * laplacian_2d(omega, grid.dR, grid.dz)
        omega = omega + dt_safe * (-advR - adpz + visc + S)
        omega[0, :] = 0.0             # regularity on axis
    return omega


# ---------- Self-check --------------------------------------------------------

def self_check() -> dict:
    grid = VortexGrid(nR=17, nz=17)
    params = VortexParams()
    R, Z = np.meshgrid(grid.R, grid.z, indexing="ij")
    omega0 = 0.1 * np.exp(-(R - 0.5) ** 2 / 0.1 - Z ** 2 / 0.2)
    omega1 = advance_vortex(grid, params, omega0, dt=1e-3, n_steps=10)
    return dict(mean_initial=float(omega0.mean()),
                mean_final=float(omega1.mean()),
                max_final=float(omega1.max()))
