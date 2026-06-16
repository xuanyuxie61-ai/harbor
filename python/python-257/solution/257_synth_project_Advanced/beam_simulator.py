"""
beam_simulator.py
=================
Simulation of the CMB telescope beam using molecular-dynamics-style
velocity-Verlet time-stepping (adapted from md_parfor.m).

The beam is modelled as N "photons" on the focal plane with a
harmonic-well pair potential:
    V(r) = sin^2(min(r, pi/2))
whose gradient gives force:
    F(r) = sin(2 min(r, pi/2)) / r   (radial)

This is the Burkardt/Cliff "saturated harmonic" potential used in
md_parfor.m.  The time integration is velocity-Verlet:
    x(t+dt) = x(t) + v(t) dt + 0.5 a(t) dt^2
    v(t+dt) = v(t) + 0.5 (a(t) + a(t+dt)) dt

After the simulation we compute the beam window function B_l from
the radial density profile.
"""

from __future__ import annotations
import math
import random
from typing import List, Tuple, Dict


# ---------------------------------------------------------------------------
def initialize_photons(n_photons: int, n_dim: int, box_size: float,
                         seed: int = 123456789) -> Tuple[List[List[float]], List[List[float]]]:
    """Random positions in [0, box_size]^n_dim, zero initial velocity."""
    rng = random.Random(seed)
    pos = [[rng.random() * box_size for _ in range(n_dim)] for _ in range(n_photons)]
    vel = [[0.0] * n_dim for _ in range(n_photons)]
    return pos, vel


# ---------------------------------------------------------------------------
def compute_forces_and_energy(pos: List[List[float]], n_dim: int,
                                 mass: float, vel: List[List[float]]
                                 ) -> Tuple[List[List[float]], float, float]:
    """
    Pairwise saturated-harmonic force and potential/kinetic energy.
    """
    n = len(pos)
    force = [[0.0] * n_dim for _ in range(n)]
    pot = 0.0
    pi2 = math.pi / 2.0
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            r2 = sum((pos[i][d] - pos[j][d]) ** 2 for d in range(n_dim))
            if r2 < 1e-30:
                continue
            r = math.sqrt(r2)
            r_cut = min(r, pi2)
            pot += 0.5 * math.sin(r_cut) ** 2
            fmag = math.sin(2.0 * r_cut) / r
            for d in range(n_dim):
                force[i][d] += (pos[i][d] - pos[j][d]) * fmag
    kin = 0.5 * mass * sum(sum(vel[i][d] ** 2 for d in range(n_dim)) for i in range(n))
    return force, pot, kin


# ---------------------------------------------------------------------------
def update_verlet(pos: List[List[float]], vel: List[List[float]],
                    acc: List[List[float]], force: List[List[float]],
                    mass: float, dt: float, n_dim: int
                    ) -> Tuple[List[List[float]], List[List[float]], List[List[float]]]:
    """Velocity-Verlet step."""
    n = len(pos)
    rmass = 1.0 / mass
    for i in range(n):
        for d in range(n_dim):
            pos[i][d] += vel[i][d] * dt + 0.5 * acc[i][d] * dt * dt
            vel[i][d] += 0.5 * dt * (force[i][d] * rmass + acc[i][d])
            acc[i][d] = force[i][d] * rmass
    return pos, vel, acc


# ---------------------------------------------------------------------------
def beam_simulation(n_photons: int = 50, n_dim: int = 2,
                      n_steps: int = 50, dt: float = 0.05,
                      box_size: float = 5.0, seed: int = 42) -> Dict[str, object]:
    """Full MD-like beam simulation."""
    pos, vel = initialize_photons(n_photons, n_dim, box_size, seed)
    acc = [[0.0] * n_dim for _ in range(n_photons)]
    force, pe, ke = compute_forces_and_energy(pos, n_dim, 1.0, vel)
    e0 = pe + ke
    e_drift_max = 0.0

    for step in range(n_steps):
        force, pe, ke = compute_forces_and_energy(pos, n_dim, 1.0, vel)
        drift = abs(pe + ke - e0) / max(1e-30, abs(e0))
        if drift > e_drift_max:
            e_drift_max = drift
        pos, vel, acc = update_verlet(pos, vel, acc, force, 1.0, dt, n_dim)

    # Build radial histogram around centroid
    cx = sum(p[0] for p in pos) / n_photons
    cy = sum(p[1] for p in pos) / n_photons if n_dim >= 2 else 0.0
    n_bins = 20
    r_max = box_size * math.sqrt(n_dim)
    dr = r_max / n_bins
    hist = [0] * n_bins
    for p in pos:
        r = math.sqrt((p[0] - cx) ** 2 + ((p[1] if n_dim >= 2 else 0.0) - cy) ** 2)
        b = min(n_bins - 1, int(r / dr))
        hist[b] += 1
    max_bin = max(hist) if hist else 1
    fwhm_r = r_max
    for b in range(1, n_bins):
        if hist[b] < max_bin / 2.0:
            fwhm_r = b * dr
            break
    sigma_b = fwhm_r / (2.0 * math.sqrt(2.0 * math.log(2.0))) if fwhm_r < r_max else 0.1
    return {
        "fwhm": fwhm_r / box_size * 0.01,
        "sigma_beam": sigma_b / box_size * 0.01,
        "energy_drift_max": e_drift_max,
        "final_positions": pos,
    }


# ---------------------------------------------------------------------------
def beam_window_bl(ell_max: int, sigma_b: float) -> List[float]:
    """Gaussian beam window function  B_l = exp(-l(l+1) sigma_b^2)."""
    return [math.exp(-ell * (ell + 1) * sigma_b * sigma_b) for ell in range(ell_max + 1)]


def apply_beam_window(cl: List[float], bl: List[float]) -> List[float]:
    """Observed C_l = C_l^true * B_l^2."""
    L = min(len(cl), len(bl))
    return [cl[l] * bl[l] ** 2 for l in range(L)]


if __name__ == "__main__":
    result = beam_simulation(n_photons=30, n_steps=30, dt=0.05)
    print(f"Beam FWHM = {result['fwhm']:.4e}")
    print(f"Max energy drift = {result['energy_drift_max']:.4e}")
