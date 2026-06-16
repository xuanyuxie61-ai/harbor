"""
molecular_dynamics.py
=====================

Metadynamics-inspired molecular-dynamics potential for the dust-grain
surface chemistry in the protoplanetary disk.  Adapted from the
DeepMD + MetaD workflow of project 1183.

Background
----------
Project 1183 couples a Deep Neural Network MD (DeepMD) potential with
metadynamics (MetaD) to sample rare events (e.g. molecule desorption
from grain surfaces) in atomistic simulation.  The MetaD bias potential
is

    V_bias(s, t) = sum_{t' < t, t' = tau, 2 tau, ...}
                     W * exp( - (s - s(t'))^2 / (2 sigma^2) )

where s(q) is a collective variable (CV) of the atomic coordinates q,
W is the Gaussian height, sigma the Gaussian width, and tau the
deposition stride.

In our multi-fidelity UQ context, we use a simplified 1-D meta-potential
as a high-fidelity correction term: the CV is the dust-grain surface
binding energy E_b, and the meta-potential provides a non-parametric
correction to the analytic binding energy:

    E_b^{HF}(E_b^{LF}) = E_b^{LF} + V_bias(E_b^{LF})

This correction is trained on-the-fly as the multi-fidelity sampler
collects data, forming a metadynamics-style exploration of the
binding-energy landscape.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# ----------------------------------------------------------------------
# Metadynamics bias potential.
# ----------------------------------------------------------------------
@dataclass
class MetaDConfig:
    """Configuration for the metadynamics bias potential."""
    sigma: float = 0.1        # Gaussian width in CV space
    W: float = 0.01           # Gaussian height
    tau: int = 5              # deposition stride (in samples)
    n_max: int = 200          # maximum number of Gaussians
    cv_min: float = -3.0      # minimum CV value
    cv_max: float = 3.0       # maximum CV value


@dataclass
class MetaDState:
    """State of the metadynamics bias potential."""
    config: MetaDConfig = field(default_factory=MetaDConfig)
    centers: List[float] = field(default_factory=list)   # deposited CV values
    n_deposited: int = 0


def bias_potential(s: float, state: MetaDState) -> float:
    """Evaluate the metadynamics bias V_bias(s) at CV value s.

    V_bias(s) = sum_i W * exp(- (s - s_i)^2 / (2 sigma^2))
    """
    if not state.centers:
        return 0.0
    sig2 = 2.0 * state.config.sigma ** 2
    total = 0.0
    for c in state.centers:
        total += state.config.W * math.exp(-((s - c) ** 2) / max(sig2, 1.0e-30))
    return total


def bias_derivative(s: float, state: MetaDState) -> float:
    """d V_bias / ds at CV value s."""
    if not state.centers:
        return 0.0
    sig2 = 2.0 * state.config.sigma ** 2
    total = 0.0
    for c in state.centers:
        d = s - c
        gauss = math.exp(-(d * d) / max(sig2, 1.0e-30))
        total += state.config.W * (-2.0 * d / max(sig2, 1.0e-30)) * gauss
    return total


def maybe_deposit(s: float, state: MetaDState, step: int) -> bool:
    """Deposit a Gaussian at s if step is a multiple of tau."""
    if step <= 0 or step % state.config.tau != 0:
        return False
    if state.n_deposited >= state.config.n_max:
        return False
    state.centers.append(s)
    state.n_deposited += 1
    return True


# ----------------------------------------------------------------------
# DeepMD-style local potential (simplified).
# ----------------------------------------------------------------------
def deepmd_local_potential(
    r: float, epsilon: float = 1.0, sigma: float = 1.0,
    cutoff: float = 2.5,
) -> float:
    """Lennard-Jones-style pair potential with smooth cutoff.

    V(r) = 4 epsilon [ (sigma/r)^12 - (sigma/r)^6 ] * f_cut(r)
    where f_cut is a cosine cutoff going to 0 at r = cutoff.
    """
    if r <= 0.0:
        raise ValueError("deepmd_local_potential: r must be positive.")
    if r >= cutoff:
        return 0.0
    sr = sigma / r
    sr6 = sr ** 6
    sr12 = sr6 * sr6
    v_lj = 4.0 * epsilon * (sr12 - sr6)
    # Cosine cutoff.
    f_cut = 0.5 * (1.0 + math.cos(math.pi * r / cutoff))
    return v_lj * f_cut


def deepmd_force(
    r: float, epsilon: float = 1.0, sigma: float = 1.0,
    cutoff: float = 2.5,
) -> float:
    """Force -dV/dr for the cutoff LJ potential."""
    if r <= 0.0 or r >= cutoff:
        return 0.0
    sr = sigma / r
    sr6 = sr ** 6
    sr12 = sr6 * sr6
    f_lj = 4.0 * epsilon * (12.0 * sr12 - 6.0 * sr6) / r
    # Cutoff derivative.
    f_cut = 0.5 * (1.0 + math.cos(math.pi * r / cutoff))
    df_cut = -0.5 * (math.pi / cutoff) * math.sin(math.pi * r / cutoff)
    v_lj = 4.0 * epsilon * (sr12 - sr6)
    return f_lj * f_cut + v_lj * df_cut


# ----------------------------------------------------------------------
# Combined high-fidelity potential: DeepMD + MetaD bias.
# ----------------------------------------------------------------------
def high_fidelity_potential(
    r: float, s: float,
    epsilon: float, sigma_lj: float, cutoff: float,
    metad_state: MetaDState,
) -> float:
    """V^{HF}(r, s) = V^{DeepMD}(r) + V_bias(s)."""
    return deepmd_local_potential(r, epsilon, sigma_lj, cutoff) \
        + bias_potential(s, metad_state)


# ----------------------------------------------------------------------
# Minimal MD integrator (velocity Verlet) for diagnostic use.
# ----------------------------------------------------------------------
def velocity_verlet_step(
    r: float, v: float, mass: float, dt: float,
    epsilon: float, sigma_lj: float, cutoff: float,
    metad_state: MetaDState,
) -> Tuple[float, float]:
    """Single velocity-Verlet step for a 1-D particle in V(r) + V_bias(s=r)."""
    f = deepmd_force(r, epsilon, sigma_lj, cutoff) + bias_derivative(r, metad_state)
    a = f / mass
    r_new = r + v * dt + 0.5 * a * dt * dt
    f_new = deepmd_force(r_new, epsilon, sigma_lj, cutoff) \
        + bias_derivative(r_new, metad_state)
    a_new = f_new / mass
    v_new = v + 0.5 * (a + a_new) * dt
    return r_new, v_new


def run_md_relaxation(
    r0: float, v0: float, mass: float, dt: float, n_steps: int,
    epsilon: float, sigma_lj: float, cutoff: float,
    metad_state: MetaDState,
) -> Tuple[List[float], List[float]]:
    """Run n_steps of velocity-Verlet MD and return trajectories."""
    r, v = r0, v0
    r_traj: List[float] = [r]
    v_traj: List[float] = [v]
    for step in range(1, n_steps + 1):
        r, v = velocity_verlet_step(
            r, v, mass, dt, epsilon, sigma_lj, cutoff, metad_state
        )
        maybe_deposit(r, metad_state, step)
        r_traj.append(r)
        v_traj.append(v)
    return r_traj, v_traj
