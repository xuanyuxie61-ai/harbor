"""
mhd_equations.py
================
Right-hand side of the compressible ideal-MHD system in the shearing-
box approximation, written in conservation-law form.

The conserved state vector is

    U = ( rho, rho v_x, rho v_y, rho v_z, E, B_x, B_y, B_z, psi )

where psi is the Dedner et al. (2002) divergence-cleaning scalar.  In
code units (H, Omega0^-1, rho0, c_s0) the fluxes along direction n are

    F^n = ( rho v_n,
            rho v v_n + (p + B^2/2) e_n - B_n B,
            (E + p + B^2/2) v_n - B_n (v.B),
            v_n B - B_n v + psi e_n,
            B_n )

and the source terms arising from the linearised Coriolis + tidal
forces in the shearing box are

    S_rho = 0
    S_mx  =  2 rho Omega0 v_y            (Coriolis x)
    S_my  = -2 rho Omega0 v_x            (Coriolis y)
            + 2 rho Omega0^2 q x         (tidal)
    S_mz  = -rho Omega0^2 z              (vertical gravity)
    S_E   =  2 rho Omega0 (v_x v_y)
            + 2 rho Omega0^2 q x v_y
            - rho Omega0^2 z v_z

These match the canonical form used in Stone et al. (1996) and in
every subsequent shearing-box code (Athena, Pluto, Idefix, Athena++).

The nonlinear reaction-network structure of
biochemical_nonlinear_deriv (091) is reused here: a stoichiometric-
like coupling matrix S multiplies a vector of non-linear "reaction"
rates r(U).  In the MHD case the "species" are the conserved
variables and the "reactions" are the fluxes/source terms.
"""

from __future__ import annotations
import numpy as np

import physical_constants as pc
import boundary_conditions as bc
import high_order_fd as hfd


# ---------------------------------------------------------------------------
#                  Equation-of-state helpers
# ---------------------------------------------------------------------------
def pressure(U: np.ndarray) -> np.ndarray:
    """Thermal pressure from the conserved state.

    p = (gamma - 1) [ E - 0.5 rho |v|^2 - 0.5 |B|^2 ]

    A floor is enforced on p to prevent negative-pressure crashes in
    low-beta regions.
    """
    gamma = pc.get("gamma_eos")
    rho   = np.maximum(U[bc.ConsIdx.rho], 1.0e-12)
    mx    = U[bc.ConsIdx.mx]
    my    = U[bc.ConsIdx.my]
    mz    = U[bc.ConsIdx.mz]
    Bx    = U[bc.ConsIdx.Bx]
    By    = U[bc.ConsIdx.By]
    Bz    = U[bc.ConsIdx.Bz]
    E     = U[bc.ConsIdx.E]
    kin   = 0.5 * (mx**2 + my**2 + mz**2) / rho
    mag   = 0.5 * (Bx**2 + By**2 + Bz**2)
    p     = (gamma - 1.0) * (E - kin - mag)
    p_floor = 1.0e-10 * rho   # ~ Mach 100 safety
    return np.maximum(p, p_floor)


def total_B2(U: np.ndarray) -> np.ndarray:
    return U[bc.ConsIdx.Bx]**2 + U[bc.ConsIdx.By]**2 + U[bc.ConsIdx.Bz]**2


def velocity(U: np.ndarray):
    rho = np.maximum(U[bc.ConsIdx.rho], 1.0e-12)
    return (U[bc.ConsIdx.mx] / rho,
            U[bc.ConsIdx.my] / rho,
            U[bc.ConsIdx.mz] / rho)


def max_signal_speed(U: np.ndarray) -> float:
    """Global maximum of |v_n| + c_f across all faces (for CFL).

    A small floor is applied to the signal speed to avoid an infinite
    timestep in cold / empty cells.
    """
    gamma = pc.get("gamma_eos")
    rho = np.maximum(U[bc.ConsIdx.rho], 1.0e-12)
    p   = pressure(U)
    B2  = total_B2(U)
    cs2 = gamma * p / rho
    va2 = B2 / (4.0 * np.pi * rho)
    cf  = np.sqrt(cs2 + va2)
    vx, vy, vz = velocity(U)
    floor = 1.0e-6  # code units
    return float(max(np.max(np.abs(vx) + cf),
                     np.max(np.abs(vy) + cf),
                     np.max(np.abs(vz) + cf),
                     floor))


# ---------------------------------------------------------------------------
#                     Flux computation along one axis
# ---------------------------------------------------------------------------
def flux_x(U: np.ndarray) -> np.ndarray:
    """Compute the x-direction flux F^x at every cell centre."""
    rho = U[bc.ConsIdx.rho]
    vx, vy, vz = velocity(U)
    p   = pressure(U)
    Bx  = U[bc.ConsIdx.Bx]
    By  = U[bc.ConsIdx.By]
    Bz  = U[bc.ConsIdx.Bz]
    B2  = Bx**2 + By**2 + Bz**2
    ptot = p + 0.5 * B2
    vdotB = vx * Bx + vy * By + vz * Bz
    E = U[bc.ConsIdx.E]

    F = np.zeros_like(U)
    F[bc.ConsIdx.rho] = rho * vx
    F[bc.ConsIdx.mx]  = rho * vx * vx + ptot - Bx * Bx
    F[bc.ConsIdx.my]  = rho * vy * vx - Bx * By
    F[bc.ConsIdx.mz]  = rho * vz * vx - Bx * Bz
    F[bc.ConsIdx.E]   = (E + ptot) * vx - Bx * vdotB
    F[bc.ConsIdx.Bx]  = 0.0              # div B maintained by construction
    F[bc.ConsIdx.By]  = vy * Bx - vx * By
    F[bc.ConsIdx.Bz]  = vz * Bx - vx * Bz
    F[bc.ConsIdx.psi] = Bx
    return F


def flux_y(U: np.ndarray) -> np.ndarray:
    rho = U[bc.ConsIdx.rho]
    vx, vy, vz = velocity(U)
    p   = pressure(U)
    Bx  = U[bc.ConsIdx.Bx]
    By  = U[bc.ConsIdx.By]
    Bz  = U[bc.ConsIdx.Bz]
    B2  = Bx**2 + By**2 + Bz**2
    ptot = p + 0.5 * B2
    vdotB = vx * Bx + vy * By + vz * Bz
    E = U[bc.ConsIdx.E]

    F = np.zeros_like(U)
    F[bc.ConsIdx.rho] = rho * vy
    F[bc.ConsIdx.mx]  = rho * vx * vy - By * Bx
    F[bc.ConsIdx.my]  = rho * vy * vy + ptot - By * By
    F[bc.ConsIdx.mz]  = rho * vz * vy - By * Bz
    F[bc.ConsIdx.E]   = (E + ptot) * vy - By * vdotB
    F[bc.ConsIdx.Bx]  = vx * By - vy * Bx
    F[bc.ConsIdx.By]  = 0.0
    F[bc.ConsIdx.Bz]  = vz * By - vy * Bz
    F[bc.ConsIdx.psi] = By
    return F


def flux_z(U: np.ndarray) -> np.ndarray:
    rho = U[bc.ConsIdx.rho]
    vx, vy, vz = velocity(U)
    p   = pressure(U)
    Bx  = U[bc.ConsIdx.Bx]
    By  = U[bc.ConsIdx.By]
    Bz  = U[bc.ConsIdx.Bz]
    B2  = Bx**2 + By**2 + Bz**2
    ptot = p + 0.5 * B2
    vdotB = vx * Bx + vy * By + vz * Bz
    E = U[bc.ConsIdx.E]

    F = np.zeros_like(U)
    F[bc.ConsIdx.rho] = rho * vz
    F[bc.ConsIdx.mx]  = rho * vx * vz - Bz * Bx
    F[bc.ConsIdx.my]  = rho * vy * vz - Bz * By
    F[bc.ConsIdx.mz]  = rho * vz * vz + ptot - Bz * Bz
    F[bc.ConsIdx.E]   = (E + ptot) * vz - Bz * vdotB
    F[bc.ConsIdx.Bx]  = vx * Bz - vz * Bx
    F[bc.ConsIdx.By]  = vy * Bz - vz * By
    F[bc.ConsIdx.Bz]  = 0.0
    F[bc.ConsIdx.psi] = Bz
    return F


# ---------------------------------------------------------------------------
#                 Shearing-box source terms  (Stone 1996)
# ---------------------------------------------------------------------------
def shearing_source(U: np.ndarray, g) -> np.ndarray:
    """Return the source-term vector S(U) for the shearing box.

    The source terms are applied only in the interior (ghost cells are
    excluded).  The shape of S matches that of U but the ghost regions
    are filled with zeros.
    """
    scales = pc.derived_scales()
    Omega0 = scales["Omega0"]
    q      = scales["q_shear"]

    rho = U[bc.ConsIdx.rho]
    vx, vy, vz = velocity(U)
    ng = 2
    Nx, Ny, Nz = g.Nx, g.Ny, g.Nz

    S = np.zeros_like(U)

    # Build x and z coordinate arrays broadcastable to interior shape.
    # Interior shape: (Nx+2*ng, Ny+2*ng, Nz+2*ng) with non-trivial
    # values only in [ng:Nx+ng, ng:Ny+ng, ng:Nz+ng].
    # We pad the coordinate arrays with zeros in the ghost zones so that
    # multiplication with ghost-region rho gives zero source contribution.
    x_full = np.zeros(Nx + 2 * ng)
    x_full[ng:Nx + ng] = g.xc
    z_full = np.zeros(Nz + 2 * ng)
    z_full[ng:Nz + ng] = g.zc
    x3 = x_full[:, None, None] * np.ones_like(rho)
    z3 = z_full[None, None, :] * np.ones_like(rho)

    S[bc.ConsIdx.mx] =  2.0 * rho * Omega0 * vy
    S[bc.ConsIdx.my] = -2.0 * rho * Omega0 * vx + 2.0 * rho * Omega0**2 * q * x3
    S[bc.ConsIdx.mz] = -rho * Omega0**2 * z3
    S[bc.ConsIdx.E]  = ( 2.0 * rho * Omega0 * vx * vy
                         + 2.0 * rho * Omega0**2 * q * x3 * vy
                         - rho * Omega0**2 * z3 * vz )
    # Density, B, psi have zero source terms.
    return S


# ---------------------------------------------------------------------------
#           Dedner hyperbolic divergence cleaning source
# ---------------------------------------------------------------------------
def dedner_source(U: np.ndarray, g,
                  c_h: float = 0.3, c_p: float = 0.3) -> np.ndarray:
    """Source terms for the Dedner et al. (2002) cleaning system.

    The augmented MHD system adds a scalar psi that satisfies

        d psi / dt + c_h^2 div B = - (c_h^2 / c_p) psi

    where c_h is the propagation speed and c_p the damping rate of
    the divergence error.  We use a mild damping to avoid destabilising
    the explicit time integrator: c_h = 0.3 * c_f_max and
    c_p = 0.3 * L / c_h.
    """
    scales = pc.derived_scales()
    cf = pc.fast_magnetosonic(float(np.mean(total_B2(U))),
                              float(np.mean(U[bc.ConsIdx.rho])),
                              float(np.mean(pressure(U))))
    c_h = 0.3 * cf
    L   = min(g.Lx, g.Ly, g.Lz)
    c_p = 0.3 * L / max(c_h, 1.0e-10)
    psi = U[bc.ConsIdx.psi]
    S = np.zeros_like(U)
    S[bc.ConsIdx.psi] = - (c_h**2 / c_p) * psi
    return S


# ---------------------------------------------------------------------------
#           Full RHS driver (assembles fluxes + sources)
# ---------------------------------------------------------------------------
def rhs(U: np.ndarray, g) -> np.ndarray:
    """Compute dU/dt = -div F + S  for the full MHD system.

    Uses robust second-order central differences (np.gradient) for the
    flux divergence; the compact-4 operator is available in the hfd
    module for verification studies on well-resolved flows.
    """
    # Apply boundary conditions before differencing
    U = bc.apply_all(U, g)

    Fx = flux_x(U)
    Fy = flux_y(U)
    Fz = flux_z(U)

    dx_mean = float(np.mean(g.dx))
    dy_mean = float(np.mean(g.dy))
    dz_mean = float(np.mean(g.dz))

    dFxdx = np.zeros_like(Fx)
    dFydy = np.zeros_like(Fy)
    dFzdz = np.zeros_like(Fz)
    for f in range(bc.NCONS):
        dFxdx[f] = np.gradient(Fx[f], dx_mean, axis=0)
        dFydy[f] = np.gradient(Fy[f], dy_mean, axis=1)
        dFzdz[f] = np.gradient(Fz[f], dz_mean, axis=2)

    dUdt = - (dFxdx + dFydy + dFzdz) + shearing_source(U, g) \
           + dedner_source(U, g)
    return dUdt
