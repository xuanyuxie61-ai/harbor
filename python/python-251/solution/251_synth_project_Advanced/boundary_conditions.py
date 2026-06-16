"""
boundary_conditions.py
======================
Application of boundary conditions to the conserved-variable array of the
shearing-box MHD solver.

Three boundary types are implemented, one per box face:

    * azimuthal (y)  -- strictly periodic, implemented as a modular
      index shift directly inspired by the Caesar cipher (132_caesar);
    * vertical  (z)  -- reflecting for rho, v_y, B_y and outflow for
      everything else (standard stratified-disk setup);
    * radial    (x)  -- shearing-periodic (HD) plus a linearised
      epicyclic correction that swaps the x-boundary values with a
      velocity shift  +- 2 A Lx where A = - q Omega / 2 is the Oort
      constant.  The shift is the discrete analogue of the Hill's-frame
      shear and is what distinguishes a shearing box from a plain
      periodic box.

The implementation follows the layout used in the Athena / Athena++
shearing box: two ghost layers on every face (compatible with WENO5
and 4th-order compact finite differences).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
import numpy as np

import physical_constants as pc


# ---------------------------------------------------------------------------
#                Conserved-variable layout (index mnemonics)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ConsIdx:
    """Integer indices into the conserved-variable array U.

    The ordering follows the canonical compressible-MHD layout so that
    hydrodynamic and magnetic routines share the same memory order.
    """
    rho: int = 0     # density
    mx:  int = 1     # x-momentum
    my:  int = 2     # y-momentum
    mz:  int = 3     # z-momentum
    E:   int = 4     # total energy
    Bx:  int = 5     # magnetic x (face centred in x)
    By:  int = 6     # magnetic y
    Bz:  int = 7     # magnetic z
    psi: int = 8     # divergence-cleaning scalar (Dedner et al. 2002)


NCONS = 9  # total number of conserved fields


# ---------------------------------------------------------------------------
#               Caesar-style modular shift (from 132_caesar)
# ---------------------------------------------------------------------------
def caesar_roll(arr: np.ndarray, k: int, axis: int) -> np.ndarray:
    """Roll ``arr`` by ``k`` slots along ``axis`` using modular indexing.

    This is the direct ndarray analogue of s_to_caesar (132_caesar):
    just as the cipher performs

        c_i  = (c_i + k) mod 26

    on each character index, we perform

        out[..., j, ...] = arr[..., (j + k) mod N, ...]

    on every grid index along the chosen axis.  The function is the
    workhorse of the azimuthal periodic boundary and of the y-shift
    inside the shearing-periodic radial boundary.
    """
    return np.roll(arr, shift=-k, axis=axis)


# ---------------------------------------------------------------------------
#                Radial shearing-periodic boundary
# ---------------------------------------------------------------------------
def apply_shearing_x(U: np.ndarray,
                     g,
                     sign: int) -> np.ndarray:
    """Apply shearing-periodic BCs in x (the radial direction).

    The shearing-box shear is v_y(x) = -q Omega x so the two x-faces
    see a relative velocity shift

        Delta_vy = - q Omega Lx

    which, for a Keplerian disk (q = 3/2), becomes  - (3/2) Omega Lx.

    In the frame corotating with R0 the two x-faces must be matched
    after shifting one face by Delta_vy and by an integer number of
    cells in y so that the simulation box remains a consistent patch
    of the disk.  The integer y-shift is

        j_shift = round( Delta_vy * t / Ly * Ny )

    but for a single boundary application at fixed time we use the
    instantaneous shift appropriate for the current linearised shear.
    We here apply the *homogeneous* part (j_shift = 0) plus the
    velocity kick; the time-dependent shift is handled at the integrator
    level (see time_integration.shearing_source).

    Parameters
    ----------
    U : ndarray of shape (NCONS, Nx+2*nghost, Ny+2*nghost, Nz+2*nghost)
    g : ShearingBoxGrid
    sign : +1  ->  fill +x ghost from -x physical (and vice-versa)
    """
    ng = 2  # ghost layers, matches WENO5 stencil
    Nx, Ny, Nz = g.Nx, g.Ny, g.Nz

    scales = pc.derived_scales()
    Omega0 = scales["Omega0"]
    q      = scales["q_shear"]
    Lx_code = g.Lx

    # Shear velocity jump across the box: Delta_vy = - q * Omega * Lx
    dvy = - q * Omega0 * Lx_code  # in code units (c_s0)
    # Half-jump applied to each face
    half_dvy = 0.5 * dvy * sign

    U_out = U.copy()
    # ---- +x face ghosts filled from -x physical side, shifted in vy ----
    for k in range(ng):
        # my  <-  my + rho * half_dvy  (momentum kick from the shear)
        src_rho = U[ConsIdx.rho, ng + k, ng:-ng, ng:-ng]
        src_my  = U[ConsIdx.my,  ng + k, ng:-ng, ng:-ng]
        # +x ghost slot: index Nx + ng + k
        U_out[ConsIdx.rho, Nx + ng + k, ng:-ng, ng:-ng] = src_rho
        U_out[ConsIdx.my,  Nx + ng + k, ng:-ng, ng:-ng] = src_my + src_rho * half_dvy
        # Copy remaining fields unchanged
        for f in (ConsIdx.mx, ConsIdx.mz, ConsIdx.E,
                  ConsIdx.Bx, ConsIdx.By, ConsIdx.Bz, ConsIdx.psi):
            U_out[f, Nx + ng + k, ng:-ng, ng:-ng] = U[f, ng + k, ng:-ng, ng:-ng]
    # ---- -x face ghosts filled from +x physical side, opposite kick ----
    for k in range(ng):
        src_rho = U[ConsIdx.rho, Nx - 1 - k, ng:-ng, ng:-ng]
        src_my  = U[ConsIdx.my,  Nx - 1 - k, ng:-ng, ng:-ng]
        U_out[ConsIdx.rho, ng - 1 - k, ng:-ng, ng:-ng] = src_rho
        U_out[ConsIdx.my,  ng - 1 - k, ng:-ng, ng:-ng] = src_my - src_rho * half_dvy
        for f in (ConsIdx.mx, ConsIdx.mz, ConsIdx.E,
                  ConsIdx.Bx, ConsIdx.By, ConsIdx.Bz, ConsIdx.psi):
            U_out[f, ng - 1 - k, ng:-ng, ng:-ng] = U[f, Nx - 1 - k, ng:-ng, ng:-ng]
    return U_out


# ---------------------------------------------------------------------------
#                 Azimuthal periodic boundary (Caesar roll)
# ---------------------------------------------------------------------------
def apply_periodic_y(U: np.ndarray, g) -> np.ndarray:
    """Wrap the y-boundary ghost cells using a Caesar-style modular roll.

    For a strictly periodic direction the ghost slots mirror the
    interior on the opposite side: the left ghost gets the last ng
    interior slices (in order), and the right ghost gets the first ng
    interior slices.  This is equivalent to the character wrap-around
    in caesar.m where 'Z' + 1 -> 'A'.
    """
    ng = 2
    U_out = U.copy()
    # Left ghosts (indices 0..ng-1) <- last ng interior slices (indices -ng:-ng+ng)
    U_out[:, :, :ng, :] = U_out[:, :, -2 * ng:-ng, :]
    # Right ghosts (indices -ng:) <- first ng interior slices (indices ng:2*ng)
    U_out[:, :, -ng:, :] = U_out[:, :, ng:2 * ng, :]
    return U_out


# ---------------------------------------------------------------------------
#                 Vertical reflecting / outflow boundary
# ---------------------------------------------------------------------------
def apply_vertical_z(U: np.ndarray, g) -> np.ndarray:
    """Reflecting for (rho, my, By), outflow for (mx, mz, E, Bx, Bz, psi).

    The split is the standard one for a stratified disk: the mid-plane
    is a symmetry plane, so odd-parity quantities change sign across it
    while even-parity quantities do not.
    """
    ng = 2
    U_out = U.copy()
    Nz = g.Nz
    # Odd parity fields: reflect with sign flip
    for f in (ConsIdx.my, ConsIdx.By):
        for k in range(ng):
            U_out[f, :, :, ng - 1 - k] = -U_out[f, :, :, ng + k]
            U_out[f, :, :, Nz + ng + k] = -U_out[f, :, :, Nz + ng - 1 - k]
    # Even parity fields: reflect without sign flip (outflow-style copy)
    for f in (ConsIdx.rho, ConsIdx.mx, ConsIdx.mz, ConsIdx.E,
              ConsIdx.Bx, ConsIdx.Bz, ConsIdx.psi):
        for k in range(ng):
            U_out[f, :, :, ng - 1 - k] = U_out[f, :, :, ng + k]
            U_out[f, :, :, Nz + ng + k] = U_out[f, :, :, Nz + ng - 1 - k]
    return U_out


# ---------------------------------------------------------------------------
#                          Unified driver
# ---------------------------------------------------------------------------
def apply_all(U: np.ndarray, g) -> np.ndarray:
    """Apply x (shear), y (periodic), z (vertical) boundaries in order."""
    U = apply_shearing_x(U, g, sign=+1)
    U = apply_periodic_y(U, g)
    U = apply_vertical_z(U, g)
    return U


# ---------------------------------------------------------------------------
#                        Sanity check utilities
# ---------------------------------------------------------------------------
def check_periodic_conservation(U: np.ndarray, g,
                                tol: float = 1.0e-10) -> bool:
    """Return True if the y-periodic boundaries match to ``tol``.

    Useful in the test harness to verify that the Caesar-roll wrapper
    has been applied correctly and that no one-cell offset has crept
    in through off-by-one errors in ghost indexing.
    """
    ng = 2
    interior = U[:, :, ng:-ng, :]      # shape (NCONS, Nx+2ng, Ny, Nz+2ng)
    left     = U[:, :, :ng, :]         # shape (NCONS, Nx+2ng, ng,  Nz+2ng)
    right    = U[:, :, -ng:, :]        # shape (NCONS, Nx+2ng, ng,  Nz+2ng)
    # The left ghost (slot 0, 1) should equal the last ng interior slices
    # (slot -ng-1, -ng) which is interior[:, :, -ng:, :]
    err_L = np.max(np.abs(interior[:, :, -ng:, :] - left))
    # The right ghost (slot -ng, -ng+1) should equal the first ng interior slices
    err_R = np.max(np.abs(interior[:, :, :ng, :] - right))
    return bool(max(err_L, err_R) < tol)
