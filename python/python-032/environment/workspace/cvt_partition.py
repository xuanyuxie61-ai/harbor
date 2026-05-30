
import numpy as np
from typing import Tuple


def cvt_iterate_2d(
    generators: np.ndarray,
    n_samples: int,
    density_func,
    bounds: dict,
    max_iter: int = 50,
    tol: float = 1e-4,
) -> Tuple[np.ndarray, float, float]:
    n_gen = len(generators)
    if n_gen < 1:
        raise ValueError("need at least one generator")
    
    ndim = 2
    z = generators.copy()
    b2_min, b2_max = bounds['beta2']
    b3_min, b3_max = bounds['beta3']
    
    for iteration in range(max_iter):

        samples = np.zeros((n_samples, ndim))
        accepted = 0
        max_attempts = n_samples * 100
        attempts = 0
        
        while accepted < n_samples and attempts < max_attempts:
            attempts += 1
            s = np.array([
                np.random.uniform(b2_min, b2_max),
                np.random.uniform(b3_min, b3_max),
            ])
            rho = density_func(s)

            if np.random.rand() < rho:
                samples[accepted] = s
                accepted += 1
        
        if accepted < n_samples // 2:

            samples = np.random.rand(n_samples, 2)
            samples[:, 0] = b2_min + samples[:, 0] * (b2_max - b2_min)
            samples[:, 1] = b3_min + samples[:, 1] * (b3_max - b3_min)
        else:
            samples = samples[:accepted]
        

        z_new = np.zeros_like(z)
        counts = np.zeros(n_gen)
        energy = 0.0
        
        for s in samples:

            dists = np.sum((z - s) ** 2, axis=1)
            nearest = np.argmin(dists)
            z_new[nearest] += s
            counts[nearest] += 1
            energy += dists[nearest]
        

        for j in range(n_gen):
            if counts[j] > 0:
                z_new[j] /= counts[j]
            else:
                z_new[j] = z[j]
        

        z_new[:, 0] = np.clip(z_new[:, 0], b2_min, b2_max)
        z_new[:, 1] = np.clip(z_new[:, 1], b3_min, b3_max)
        

        diff = np.sum(np.sqrt(np.sum((z_new - z) ** 2, axis=1)))
        energy = energy / len(samples) if len(samples) > 0 else 0.0
        
        z = z_new
        
        if diff < tol:
            break
    
    return z, diff, energy


def cvt_partition_fission_space(
    mass_number: int,
    charge_number: int,
    excitation_energy: float,
    n_generators: int = 20,
    n_samples: int = 5000,
) -> Tuple[np.ndarray, float, float]:
    from potential_energy_surface import potential_energy
    from diffusion_coefficient import nuclear_temperature
    from collective_coordinates import collective_coordinate_bounds
    
    T = nuclear_temperature(excitation_energy, mass_number)
    if T < 0.1:
        T = 0.1
    
    bounds = collective_coordinate_bounds(mass_number)
    

    b2_min, b2_max = bounds['beta2']
    b3_min, b3_max = bounds['beta3']
    
    generators = np.zeros((n_generators, 2))
    generators[:, 0] = np.random.uniform(b2_min, b2_max, n_generators)
    generators[:, 1] = np.random.uniform(b3_min, b3_max, n_generators)
    
    def density_func(s):
        q = np.array([s[0], s[1], 0.0, 0.0, 0.0])
        V = potential_energy(q, mass_number, charge_number)
        rho = np.exp(-V / T)
        return float(np.clip(rho, 0.0, 1.0))
    
    z_opt, diff, energy = cvt_iterate_2d(
        generators, n_samples, density_func, bounds, max_iter=30, tol=1e-3
    )
    
    return z_opt, diff, energy
