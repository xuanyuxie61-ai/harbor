"""
stochastic_li_transport.py
================================================================================
Stochastic lithium-ion transport modeling via Feynman-Kac formula and
Brownian motion simulation in electrode particles and electrolyte.

Injects core algorithms from:
  - 424_feynman_kac_3d  (Brownian motion, Monte Carlo PDE solver)
  - 209_conte_deboor     (random number utilities, ODE integration)

Scientific role:
  At the microscale, lithium-ion transport is fundamentally stochastic.
  This module:
    1. Simulates Brownian trajectories of Li+ in spherical active particles
       using the Feynman-Kac formula for concentration expectation.
    2. Estimates first-passage times for solid-state diffusion.
    3. Provides Monte Carlo estimates of local concentration fluctuations
       which feed into the Butler-Volmer reaction rate variance.
================================================================================
"""

import numpy as np
from typing import Tuple, Callable


# ==============================================================================
# Brownian motion in spherical particles
# ==============================================================================

def brownian_step_3d(pos: np.ndarray, dt: float, D: float) -> np.ndarray:
    """
    Single step of 3D isotropic Brownian motion:
    X(t+dt) = X(t) + sqrt(2*D*dt) * Z  where Z ~ N(0, I_3).
    Maps from 424_feynman_kac_3d.
    """
    return pos + np.sqrt(2.0 * D * dt) * np.random.randn(3)


def reflect_sphere(pos: np.ndarray, center: np.ndarray, radius: float) -> np.ndarray:
    """
    Reflect a particle back inside a sphere if it crossed the boundary.
    """
    vec = pos - center
    dist = np.linalg.norm(vec)
    if dist > radius:
        vec = vec / (dist + 1e-18) * (2.0 * radius - dist)
        pos = center + vec
    return pos


def feynman_kac_particle_diffusion(
    radius: float,
    D_s: float,
    surface_concentration: float,
    n_paths: int = 5000,
    dt: float = 1e-4,
    t_max: float = 1.0,
    potential_func: Callable[[np.ndarray], float] = None
) -> Tuple[float, float]:
    """
    Estimate the average lithium concentration inside a spherical particle
    at time t_max using the Feynman-Kac formula:

        u(x,t) = E[ exp(-integral_0^tau V(X_s) ds) * g(X_tau) ]

    For pure diffusion (V=0), this reduces to the expected value of the
    boundary condition along Brownian paths.

    Parameters
    ----------
    radius : particle radius (m)
    D_s : solid diffusivity (m^2/s)
    surface_concentration : boundary concentration (mol/m^3)
    n_paths : number of Monte Carlo sample paths
    dt : time step
    t_max : simulation end time
    potential_func : optional potential V(x)

    Returns
    -------
    mean_concentration, std_concentration
    """
    if potential_func is None:
        def potential_func(x):
            return 0.0

    center = np.zeros(3)
    n_steps = int(t_max / dt)
    values = np.zeros(n_paths, dtype=float)

    for p in range(n_paths):
        # Start uniformly inside sphere
        r = radius * np.random.rand() ** (1.0 / 3.0)
        theta = np.arccos(2.0 * np.random.rand() - 1.0)
        phi = 2.0 * np.pi * np.random.rand()
        pos = np.array([
            r * np.sin(theta) * np.cos(phi),
            r * np.sin(theta) * np.sin(phi),
            r * np.cos(theta)
        ])
        integral_V = 0.0
        alive = True
        for step in range(n_steps):
            pos = brownian_step_3d(pos, dt, D_s)
            pos = reflect_sphere(pos, center, radius)
            V = potential_func(pos)
            integral_V += V * dt
            # Check if path hit surface (absorption)
            if np.linalg.norm(pos) >= 0.99 * radius:
                alive = False
                break
        W = np.exp(-integral_V)
        values[p] = W * surface_concentration if alive else surface_concentration * 0.5

    return float(np.mean(values)), float(np.std(values))


# ==============================================================================
# First passage time estimation
# ==============================================================================

def first_passage_time_monte_carlo(
    radius: float,
    D_s: float,
    start_radius: float = 0.0,
    n_paths: int = 2000,
    dt: float = 1e-5,
    max_steps: int = 500000
) -> Tuple[float, float]:
    """
    Estimate mean first-passage time (MFPT) for a diffusing particle
    starting at start_radius to reach the surface of a sphere of radius R.
    The exact MFPT for radial diffusion from center is R^2 / (6*D).
    """
    fpt_samples = np.zeros(n_paths, dtype=float)
    center = np.zeros(3)
    for p in range(n_paths):
        r0 = start_radius if start_radius > 0 else radius * np.random.rand() ** (1.0 / 3.0)
        theta = np.arccos(2.0 * np.random.rand() - 1.0)
        phi = 2.0 * np.pi * np.random.rand()
        pos = np.array([
            r0 * np.sin(theta) * np.cos(phi),
            r0 * np.sin(theta) * np.sin(phi),
            r0 * np.cos(theta)
        ])
        t = 0.0
        step_count = 0
        while step_count < max_steps:
            pos = brownian_step_3d(pos, dt, D_s)
            t += dt
            step_count += 1
            dist = np.linalg.norm(pos)
            if dist >= radius:
                break
        fpt_samples[p] = t
    return float(np.mean(fpt_samples)), float(np.std(fpt_samples))


# ==============================================================================
# Stochastic electrolyte transport (1D projection)
# ==============================================================================

def stochastic_electrolyte_walk(
    n_particles: int,
    length: float,
    D_e: float,
    dt: float = 1e-4,
    n_steps: int = 1000,
    left_flux: float = 0.0,
    right_flux: float = 0.0
) -> np.ndarray:
    """
    Simulate 1D random walk of Li+ in electrolyte with reflecting boundaries.
    Returns final particle positions.
    """
    positions = np.random.uniform(0.0, length, size=n_particles)
    step_std = np.sqrt(2.0 * D_e * dt)
    for _ in range(n_steps):
        positions += step_std * np.random.randn(n_particles)
        # Reflecting boundaries
        positions = np.abs(positions)
        mask = positions > length
        positions[mask] = 2.0 * length - positions[mask]
        # Small drift from flux
        positions += (left_flux - right_flux) * dt / n_particles
    return positions


def concentration_variance_from_walk(positions: np.ndarray, length: float,
                                     n_bins: int = 20) -> float:
    """Compute concentration variance across bins as a measure of mixing."""
    hist, _ = np.histogram(positions, bins=n_bins, range=(0.0, length))
    concentrations = hist / (len(positions) / n_bins + 1e-18)
    return float(np.var(concentrations))
