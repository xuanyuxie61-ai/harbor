
import numpy as np
from typing import Tuple, Optional


BOLTZMANN = 1.380649e-23
TEMPERATURE_BODY = 310.15
VISCOSITY_BLOOD = 3.5e-3


def diffusion_coefficient(radius: float, temperature: float = TEMPERATURE_BODY,
                          viscosity: float = VISCOSITY_BLOOD) -> float:
    if radius <= 0:
        raise ValueError("半径必须为正")
    if viscosity <= 0:
        raise ValueError("粘度必须为正")
    
    D = BOLTZMANN * temperature / (6.0 * np.pi * viscosity * radius)
    return D


def brownian_step(n_particles: int, dim: int, D: float, dt: float) -> np.ndarray:
    if dt < 0:
        raise ValueError("时间步长必须非负")
    
    sigma = np.sqrt(2.0 * D * dt)
    step = sigma * np.random.randn(n_particles, dim)
    return step


def simulate_microbubble_diffusion(n_particles: int = 1000,
                                   radius: float = 2.5e-6,
                                   n_steps: int = 1000,
                                   dt: float = 1e-6,
                                   domain_size: float = 5e-3,
                                   temperature: float = TEMPERATURE_BODY,
                                   viscosity: float = VISCOSITY_BLOOD,
                                   acoustic_force: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    dim = 2
    D = diffusion_coefficient(radius, temperature, viscosity)
    

    trajectory = np.zeros((n_steps + 1, n_particles, dim))
    trajectory[0] = np.random.uniform(0, domain_size, (n_particles, dim))
    

    gamma = 6.0 * np.pi * viscosity * radius
    

    if acoustic_force is None:
        force = np.zeros(dim)
    elif acoustic_force.ndim == 1:
        force = acoustic_force
    else:
        force = np.zeros(dim)
    
    for step in range(n_steps):

        brownian = brownian_step(n_particles, dim, D, dt)
        

        if acoustic_force is not None and acoustic_force.ndim == 2:
            force = acoustic_force[step % len(acoustic_force)]
        drift = (force / gamma) * dt
        

        trajectory[step + 1] = trajectory[step] + brownian + drift
        

        for d in range(dim):

            mask_low = trajectory[step + 1, :, d] < 0
            trajectory[step + 1, mask_low, d] = -trajectory[step + 1, mask_low, d]
            

            mask_high = trajectory[step + 1, :, d] > domain_size
            trajectory[step + 1, mask_high, d] = 2 * domain_size - trajectory[step + 1, mask_high, d]
    

    msd = np.zeros(n_steps + 1)
    initial_pos = trajectory[0]
    for step in range(n_steps + 1):
        displacements = trajectory[step] - initial_pos
        squared_displacements = np.sum(displacements**2, axis=1)
        msd[step] = np.mean(squared_displacements)
    
    return trajectory, msd, D


def acoustic_radiation_force(frequency: float = 5e6,
                              pressure_amplitude: float = 1e5,
                              bubble_radius: float = 2.5e-6,
                              c0: float = 1540.0,
                              rho0: float = 1000.0) -> float:
    k = 2.0 * np.pi * frequency / c0
    volume = 4.0 / 3.0 * np.pi * bubble_radius**3
    
    F = volume * k * pressure_amplitude**2 / (3.0 * rho0 * c0**2)
    return F


def concentration_profile(trajectory: np.ndarray, domain_size: float,
                          n_bins: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    final_positions = trajectory[-1]
    

    hist, xedges, yedges = np.histogram2d(
        final_positions[:, 0], final_positions[:, 1],
        bins=n_bins, range=[[0, domain_size], [0, domain_size]]
    )
    

    bin_area = (domain_size / n_bins)**2
    concentration = hist / bin_area
    
    return xedges, yedges, concentration
