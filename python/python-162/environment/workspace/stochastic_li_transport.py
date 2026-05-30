
import numpy as np
from typing import Tuple, Callable






def brownian_step_3d(pos: np.ndarray, dt: float, D: float) -> np.ndarray:
    return pos + np.sqrt(2.0 * D * dt) * np.random.randn(3)


def reflect_sphere(pos: np.ndarray, center: np.ndarray, radius: float) -> np.ndarray:
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
    if potential_func is None:
        def potential_func(x):
            return 0.0

    center = np.zeros(3)
    n_steps = int(t_max / dt)
    values = np.zeros(n_paths, dtype=float)

    for p in range(n_paths):

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

            if np.linalg.norm(pos) >= 0.99 * radius:
                alive = False
                break
        W = np.exp(-integral_V)
        values[p] = W * surface_concentration if alive else surface_concentration * 0.5

    return float(np.mean(values)), float(np.std(values))






def first_passage_time_monte_carlo(
    radius: float,
    D_s: float,
    start_radius: float = 0.0,
    n_paths: int = 2000,
    dt: float = 1e-5,
    max_steps: int = 500000
) -> Tuple[float, float]:
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






def stochastic_electrolyte_walk(
    n_particles: int,
    length: float,
    D_e: float,
    dt: float = 1e-4,
    n_steps: int = 1000,
    left_flux: float = 0.0,
    right_flux: float = 0.0
) -> np.ndarray:
    positions = np.random.uniform(0.0, length, size=n_particles)
    step_std = np.sqrt(2.0 * D_e * dt)
    for _ in range(n_steps):
        positions += step_std * np.random.randn(n_particles)

        positions = np.abs(positions)
        mask = positions > length
        positions[mask] = 2.0 * length - positions[mask]

        positions += (left_flux - right_flux) * dt / n_particles
    return positions


def concentration_variance_from_walk(positions: np.ndarray, length: float,
                                     n_bins: int = 20) -> float:
    hist, _ = np.histogram(positions, bins=n_bins, range=(0.0, length))
    concentrations = hist / (len(positions) / n_bins + 1e-18)
    return float(np.var(concentrations))
