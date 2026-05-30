
import numpy as np
from typing import Callable, Tuple


def triangle01_area() -> float:
    return 0.5


def triangle01_sample(n: int) -> np.ndarray:
    samples = np.random.rand(n, 2)
    mask = samples[:, 0] + samples[:, 1] > 1.0
    samples[mask] = 1.0 - samples[mask]
    return samples


def triangle_monte_carlo_integral(
    f: Callable[[np.ndarray], np.ndarray],
    n_samples: int = 10000,
) -> float:
    area = triangle01_area()
    samples = triangle01_sample(n_samples)
    values = f(samples)
    return area * np.mean(values)


def gaussian_mass_distribution(
    mass_centers: np.ndarray,
    A_peak: float,
    sigma: float,
) -> np.ndarray:
    if sigma <= 0:
        sigma = 1.0
    return (1.0 / (sigma * np.sqrt(2.0 * np.pi))) * np.exp(
        -0.5 * ((mass_centers - A_peak) / sigma) ** 2
    )


def bimodal_mass_distribution(
    mass_centers: np.ndarray,
    A_light: float,
    A_heavy: float,
    sigma_light: float,
    sigma_heavy: float,
    weight_ratio: float = 1.0,
) -> np.ndarray:
    if sigma_light <= 0:
        sigma_light = 2.0
    if sigma_heavy <= 0:
        sigma_heavy = 2.0
    
    w_total = 1.0 + weight_ratio
    w_L = weight_ratio / w_total
    w_H = 1.0 / w_total
    
    Y_L = gaussian_mass_distribution(mass_centers, A_light, sigma_light)
    Y_H = gaussian_mass_distribution(mass_centers, A_heavy, sigma_heavy)
    
    return w_L * Y_L + w_H * Y_H


def importance_sampling_mc_yield(
    mass_number: int,
    charge_number: int,
    excitation_energy: float,
    n_samples: int = 50000,
) -> Tuple[np.ndarray, np.ndarray]:
    from potential_energy_surface import potential_energy
    from collective_coordinates import (
        mass_asymmetry_to_fragment_mass,
        collective_coordinate_bounds,
        clip_to_physical_domain,
    )
    from diffusion_coefficient import nuclear_temperature
    
    T = nuclear_temperature(excitation_energy, mass_number)
    if T < 0.1:
        T = 0.1
    
    bounds = collective_coordinate_bounds(mass_number)
    

    q_current = np.array([0.3, 0.0, 0.0, 0.0, 1.0])
    q_current = clip_to_physical_domain(q_current, bounds)
    V_current = potential_energy(q_current, mass_number, charge_number)
    
    masses = []
    n_accepted = 0
    n_burn = min(1000, n_samples // 10)
    total_steps = n_samples + n_burn
    
    step_size = np.array([0.1, 0.15, 0.08, 0.04, 0.3])
    
    for step in range(total_steps):
        q_proposal = q_current + step_size * np.random.randn(5)
        q_proposal = clip_to_physical_domain(q_proposal, bounds)
        V_proposal = potential_energy(q_proposal, mass_number, charge_number)
        
        delta_V = V_proposal - V_current

        accept = False
        if delta_V < 0:
            accept = True
        else:
            if np.random.rand() < np.exp(-delta_V / T):
                accept = True
        
        if accept:
            q_current = q_proposal
            V_current = V_proposal
            n_accepted += 1
        
        if step >= n_burn:
            beta3 = q_current[1]
            A_L, A_H = mass_asymmetry_to_fragment_mass(beta3, mass_number)

            if np.random.rand() < 0.5:
                masses.append(A_L)
            else:
                masses.append(A_H)
    
    masses = np.array(masses)
    

    A_min = max(1.0, mass_number * 0.25)
    A_max = mass_number * 0.75
    n_bins = min(80, n_samples // 200)
    n_bins = max(n_bins, 20)
    bins = np.linspace(A_min, A_max, n_bins + 1)
    counts, edges = np.histogram(masses, bins=bins)
    bin_width = edges[1] - edges[0]
    

    total_counts = np.sum(counts)
    if total_counts > 0:
        counts = counts / (total_counts * bin_width)
    
    mass_centers = 0.5 * (edges[:-1] + edges[1:])
    return mass_centers, counts


def scission_point_yield_model(
    mass_number: int,
    charge_number: int,
    excitation_energy: float,
) -> Tuple[np.ndarray, np.ndarray]:
    from potential_energy_surface import (
        potential_energy_1d,
        fission_barrier_height,
    )
    from diffusion_coefficient import nuclear_temperature
    from collective_coordinates import mass_asymmetry_to_fragment_mass
    
    T = nuclear_temperature(excitation_energy, mass_number)
    barrier = fission_barrier_height(mass_number, charge_number)
    

    beta3_grid = np.linspace(-1.0, 1.0, 200)
    

    def V_beta3(b3):
        from potential_energy_surface import potential_energy
        q = np.array([1.0, b3, 0.0, 0.0, 0.0])
        return potential_energy(q, mass_number, charge_number)
    
    V_vals = np.array([V_beta3(b3) for b3 in beta3_grid])
    

    dV = np.gradient(V_vals, beta3_grid)
    ddV = np.gradient(dV, beta3_grid)
    

    minima_idx = []
    for i in range(1, len(beta3_grid) - 1):
        if dV[i - 1] < 0 and dV[i + 1] > 0 and ddV[i] > 0:
            minima_idx.append(i)
    
    if len(minima_idx) >= 2:

        idx_L = minima_idx[0]
        idx_H = minima_idx[-1]
        beta3_L = beta3_grid[idx_L]
        beta3_H = beta3_grid[idx_H]
        A_L, _ = mass_asymmetry_to_fragment_mass(beta3_L, mass_number)
        _, A_H = mass_asymmetry_to_fragment_mass(beta3_H, mass_number)
        






        raise NotImplementedError("Hole_2: scission_point 质量宽度计算待修复")
        sigma_A_L = 3.0
        sigma_A_H = 3.0
    else:

        A_L = mass_number * 0.4
        A_H = mass_number * 0.6
        sigma_A_L = 3.0
        sigma_A_H = 3.0
    
    mass_centers = np.linspace(mass_number * 0.2, mass_number * 0.8, 100)
    yield_dist = bimodal_mass_distribution(
        mass_centers, A_L, A_H, sigma_A_L, sigma_A_H, weight_ratio=1.0
    )
    

    total = np.trapezoid(yield_dist, mass_centers)
    if total > 0:
        yield_dist = yield_dist / total
    
    return mass_centers, yield_dist
