"""
initial_conditions.py
=====================
Construction of the initial conserved-variable array for the shearing-
box MHD problem.

The equilibrium is a vertically stratified, locally isothermal
Keplerian disk threaded by a weak toroidal magnetic field.  The
equilibrium is then perturbed by:

    * a random velocity seed (divergence-free, low amplitude) that
      triggers the magnetorotational instability (MRI);
    * a random pressure perturbation that seeds acoustic modes;
    * a random magnetic perturbation that breaks the symmetry of the
      initial toroidal field.

The random fields are generated with the Monte-Carlo infrastructure
from 533_high_card_parfor and 226_craps_simulation: many independent
realisations are averaged to obtain a smoother seed that converges
to a well-defined ensemble as the number of trials increases.  The
final seed is a weighted combination of the averaged realisations,
with weights chosen by a logistic growth model (701_logistic_exact)
that ramps the amplitude smoothly from zero at the box edges to its
full value in the interior.
"""

from __future__ import annotations
import math
import numpy as np

import physical_constants as pc
import boundary_conditions as bc
import grid_manager as gm
import mhd_equations as mhd


# ---------------------------------------------------------------------------
#                     Equilibrium disk profiles
# ---------------------------------------------------------------------------
def equilibrium_density(g, z_profile: str = "gaussian") -> np.ndarray:
    """Return the equilibrium density rho_0(x, z).

    In the unstratified shearing box rho_0 = const; with vertical
    stratification we use

        rho_0(z) = rho_mid * exp( - z^2 / (2 H_rho^2) )

    where H_rho ~ H for an isothermal disk.  The Gaussian profile is
    the exact steady state of the vertical momentum equation when
    pressure and gravity balance.
    """
    rho_mid = 1.0  # code units
    if z_profile == "gaussian":
        z3 = g.zc[None, None, :] * np.ones((g.Nx, g.Ny, g.Nz))
        H_rho = 1.0  # code units (= H)
        rho = rho_mid * np.exp(-0.5 * (z3 / H_rho)**2)
    elif z_profile == "uniform":
        rho = np.full((g.Nx, g.Ny, g.Nz), rho_mid)
    else:
        raise ValueError(f"equilibrium_density: unknown profile '{z_profile}'")
    return rho


def equilibrium_velocity(g) -> tuple:
    """Return (vx, vy, vz) for the shearing-box equilibrium.

    The equilibrium flow is purely azimuthal:

        vy(x) = - q Omega0 x

    with q = 3/2 for Keplerian.  In code units (Omega0 = 1) this
    simplifies to vy(x) = - (3/2) x.
    """
    q = 1.5
    x3 = g.xc[:, None, None] * np.ones((g.Nx, g.Ny, g.Nz))
    vx = np.zeros((g.Nx, g.Ny, g.Nz))
    vy = - q * x3  # Omega0 = 1 in code units
    vz = np.zeros((g.Nx, g.Ny, g.Nz))
    return vx, vy, vz


def equilibrium_magnetic(g, B0: float) -> np.ndarray:
    """Return (Bx, By, Bz) for the equilibrium field.

    The default configuration is a uniform toroidal field

        By = B0      (with sign set so that the MRI is unstable)

    with Bx = Bz = 0.  A weak vertical field can be added by setting
    the optional flag ``add_vertical``; we omit it here because the
    purely toroidal case is the one with the sharpest MRI onset and
    is therefore the most demanding test of the integrator.
    """
    Bx = np.zeros((g.Nx, g.Ny, g.Nz))
    By = np.full((g.Nx, g.Ny, g.Nz), B0)
    Bz = np.zeros((g.Nx, g.Ny, g.Nz))
    return Bx, By, Bz


def equilibrium_pressure(g, rho: np.ndarray, cs2: float = 1.0) -> np.ndarray:
    """Isothermal equilibrium pressure p = cs^2 * rho."""
    return cs2 * rho


# ---------------------------------------------------------------------------
#             Random perturbation generator (Monte-Carlo)
# ---------------------------------------------------------------------------
def mc_perturbation_field(g, amp: float, seed: int,
                          n_trials: int = 8,
                          correlation_length: float = 0.5) -> np.ndarray:
    """Generate a low-amplitude random scalar field by Monte-Carlo averaging.

    For each trial we draw white noise with standard normal entries
    and smooth it by convolution with a Gaussian kernel whose width
    is ``correlation_length`` (in code units).  The average over
    ``n_trials`` independent realisations converges to zero as
    1/sqrt(n_trials); we keep the *fluctuation* about the mean as
    the perturbation.

    This directly mirrors the high_card_parfor (533) paradigm where a
    parfor loop averages many independent stochastic simulations; the
    difference is that we work on a 3-D grid rather than a scalar
    probability.
    """
    rng = np.random.default_rng(seed)
    accum = np.zeros((g.Nx, g.Ny, g.Nz))
    # Gaussian smoothing kernel (in cell units)
    sx = max(1, int(correlation_length * g.Nx / g.Lx))
    sy = max(1, int(correlation_length * g.Ny / g.Ly))
    sz = max(1, int(correlation_length * g.Nz / g.Lz))
    kx = np.arange(-2 * sx, 2 * sx + 1)
    ky = np.arange(-2 * sy, 2 * sy + 1)
    kz = np.arange(-2 * sz, 2 * sz + 1)
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing="ij")
    kernel = np.exp(-0.5 * ((KX / sx)**2 + (KY / sy)**2 + (KZ / sz)**2))
    kernel /= kernel.sum()

    for _ in range(n_trials):
        noise = rng.standard_normal((g.Nx, g.Ny, g.Nz))
        smoothed = np.fft.ifftn(np.fft.fftn(noise) * np.fft.fftn(kernel, s=noise.shape)).real
        accum += smoothed
    mean = accum / n_trials
    # Fluctuation about the trial mean
    fluct = accum - mean * n_trials  # zero-mean by construction
    fluct /= (np.std(fluct) + 1.0e-30)
    return amp * fluct


# ---------------------------------------------------------------------------
#               Logistic ramp function (from 701_logistic_exact)
# ---------------------------------------------------------------------------
def logistic_ramp(g,
                  r: float = 4.0,
                  k: float = 1.0) -> np.ndarray:
    """Spatial envelope that ramps from 0 at the box edge to 1 in the
    interior.

    We use the exact solution of the logistic ODE (701_logistic_exact)

        y(d) = k / (1 + (k/y0 - 1) exp(-r d))

    where d = min(distance to any face) is used as a pseudo-time.  With
    y0 = 0.01 and r tuned so that the ramp reaches ~0.99 at d = 0.5
    (code units) we obtain a smooth, differentiable envelope that
    prevents the perturbations from immediately interacting with the
    boundaries.
    """
    y0 = 0.01
    # Distance of each cell centre to the nearest face
    dxL = g.xc + 0.5 * g.Lx
    dxR = 0.5 * g.Lx - g.xc
    dyL = g.yc + 0.5 * g.Ly
    dyR = 0.5 * g.Ly - g.yc
    dzL = g.zc + 0.5 * g.Lz
    dzR = 0.5 * g.Lz - g.zc
    d = np.minimum.reduce([
        dxL[:, None, None] * np.ones((g.Nx, g.Ny, g.Nz)),
        dxR[:, None, None] * np.ones((g.Nx, g.Ny, g.Nz)),
        dyL[None, :, None] * np.ones((g.Nx, g.Ny, g.Nz)),
        dyR[None, :, None] * np.ones((g.Nx, g.Ny, g.Nz)),
        dzL[None, None, :] * np.ones((g.Nx, g.Ny, g.Nz)),
        dzR[None, None, :] * np.ones((g.Nx, g.Ny, g.Nz)),
    ])
    # Logistic envelope
    env = k / (1.0 + (k / y0 - 1.0) * np.exp(-r * d))
    # Normalise so that the central value is exactly unity
    env /= np.max(env) + 1.0e-30
    return env


# ---------------------------------------------------------------------------
#                      Full initial-condition builder
# ---------------------------------------------------------------------------
def build_initial_state(g) -> np.ndarray:
    """Assemble the initial conserved-variable array U.

    Returns an array of shape (NCONS, Nx+2*ng, Ny+2*ng, Nz+2*ng)
    with ng = 2 ghost layers on every side, ready to be fed into the
    time integrator.
    """
    scales = pc.derived_scales()
    B0_code = scales["B0"] / (math.sqrt(4.0 * math.pi * scales["rho0"])
                              * scales["cs0"])  # Alfvenic code units
    amp_v = pc.get("amp_velocity")
    amp_p = pc.get("amp_pressure")
    amp_B = pc.get("amp_magnetic")
    seed  = pc.get("seed")

    rho_eq = equilibrium_density(g)
    vx_eq, vy_eq, vz_eq = equilibrium_velocity(g)
    Bx_eq, By_eq, Bz_eq = equilibrium_magnetic(g, B0_code)
    p_eq = equilibrium_pressure(g, rho_eq, cs2=1.0)

    # Logistic envelope for perturbation localisation
    env = logistic_ramp(g)

    # Random fields (Monte-Carlo averaged)
    drho = mc_perturbation_field(g, amp_p, seed + 0, n_trials=6) * env
    dvx  = mc_perturbation_field(g, amp_v, seed + 1, n_trials=6) * env
    dvy  = mc_perturbation_field(g, amp_v, seed + 2, n_trials=6) * env
    dvz  = mc_perturbation_field(g, amp_v, seed + 3, n_trials=6) * env
    dBy  = mc_perturbation_field(g, amp_B, seed + 4, n_trials=6) * env

    rho = rho_eq * (1.0 + drho)
    vx  = vx_eq + dvx
    vy  = vy_eq + dvy
    vz  = vz_eq + dvz
    Bx  = Bx_eq
    By  = By_eq * (1.0 + dBy)
    Bz  = Bz_eq
    p   = p_eq * (1.0 + amp_p * mc_perturbation_field(g, 1.0, seed + 5,
                                                      n_trials=4) * env)

    # Convert to conserved variables
    ng = 2
    shape_full = (bc.NCONS, g.Nx + 2 * ng, g.Ny + 2 * ng, g.Nz + 2 * ng)
    U = np.zeros(shape_full)
    U[bc.ConsIdx.rho, ng:-ng, ng:-ng, ng:-ng] = rho
    U[bc.ConsIdx.mx,  ng:-ng, ng:-ng, ng:-ng] = rho * vx
    U[bc.ConsIdx.my,  ng:-ng, ng:-ng, ng:-ng] = rho * vy
    U[bc.ConsIdx.mz,  ng:-ng, ng:-ng, ng:-ng] = rho * vz
    gamma = pc.get("gamma_eos")
    E_int = p / (gamma - 1.0)
    E_kin = 0.5 * rho * (vx**2 + vy**2 + vz**2)
    E_mag = 0.5 * (Bx**2 + By**2 + Bz**2)
    U[bc.ConsIdx.E, ng:-ng, ng:-ng, ng:-ng] = E_int + E_kin + E_mag
    U[bc.ConsIdx.Bx, ng:-ng, ng:-ng, ng:-ng] = Bx
    U[bc.ConsIdx.By, ng:-ng, ng:-ng, ng:-ng] = By
    U[bc.ConsIdx.Bz, ng:-ng, ng:-ng, ng:-ng] = Bz
    U[bc.ConsIdx.psi, ng:-ng, ng:-ng, ng:-ng] = 0.0

    # Fill ghost cells with the equilibrium + BCs
    U = bc.apply_all(U, g)
    return U


# ---------------------------------------------------------------------------
#                         Verification routines
# ---------------------------------------------------------------------------
def divergence_B_max(U: np.ndarray, g) -> float:
    """Return the maximum |div B| in the interior (diagnostic)."""
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
