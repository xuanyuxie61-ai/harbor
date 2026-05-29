"""
Pore-network dynamics and molecular trajectory modeling.

Adapted from:
  - kepler_ode (conservative trajectory in cylindrical pore)
  - quasiperiodic_ode (oscillatory diffusion)
  - runge (stiff interpolation test)
  - xyz_display (3D point coordinate handling)
"""

import numpy as np
from utils import generate_pore_coordinates, safe_sqrt
from time_integrator import runge_kutta4, conserved_quantity_kepler, compute_conservation_drift
from mass_transfer_ode import kepler_like_trajectory_deriv, kepler_parameters


def knudsen_diffusivity(r_pore, T, M_molar):
    """
    Knudsen diffusion coefficient in a cylindrical pore:
        D_K = (2 / 3) * r_pore * sqrt(8 R T / (pi M))
    where M_molar is in kg/mol.
    """
    R = 8.314
    return (2.0 / 3.0) * r_pore * np.sqrt(8.0 * R * T / (np.pi * M_molar))


def effective_diffusivity_support(D_knudsen, porosity, tortuosity):
    """
    Effective diffusion coefficient in porous support:
        D_eff = (epsilon / tau) * D_knudsen
    """
    return (porosity / tortuosity) * D_knudsen


def mean_free_path(T, P, d_collision):
    """
    Mean free path for hard-sphere molecules:
        lambda = k_B T / (sqrt(2) pi d^2 P)
    """
    k_B = 1.380649e-23
    return k_B * T / (np.sqrt(2.0) * np.pi * (d_collision ** 2) * P)


def pore_flux_maxwell_stefan(c1, c2, D_eff, L_pore):
    """
    Steady-state flux through a pore using the effective-medium approximation.
    """
    if L_pore <= 0:
        return 0.0
    return D_eff * (c1 - c2) / L_pore


def simulate_molecular_trajectory(n_steps=5000):
    """
    Integrate the Kepler-like trajectory for a representative molecule
    inside a cylindrical pore and report energy conservation.
    """
    params = kepler_parameters()
    f = lambda t, y: kepler_like_trajectory_deriv(t, y, mu=params["mu"])
    t, y = runge_kutta4(f, params["tspan"], params["y0"], n_steps)
    drift = compute_conservation_drift(t, y, conserved_quantity_kepler)
    return t, y, drift


def simulate_pore_network_flux(pore_count, mean_radius, std_radius,
                                length, T, M_co2, M_ch4,
                                porosity, tortuosity,
                                c_feed, c_perm):
    """
    Monte-Carlo style pore-network model: generate pore radii from a
    log-normal distribution, compute Knudsen diffusivity for each pore,
    and integrate total flux.
    """
    rng = np.random.default_rng(seed=139)
    # Log-normal radii
    sigma_ln = np.sqrt(np.log(1.0 + (std_radius / mean_radius) ** 2))
    mu_ln = np.log(mean_radius) - 0.5 * sigma_ln ** 2
    radii = rng.lognormal(mean=mu_ln, sigma=sigma_ln, size=pore_count)
    radii = np.clip(radii, 1e-10, 1e-6)

    D_k_co2 = knudsen_diffusivity(radii, T, M_co2)
    D_k_ch4 = knudsen_diffusivity(radii, T, M_ch4)
    D_eff_co2 = effective_diffusivity_support(D_k_co2, porosity, tortuosity)
    D_eff_ch4 = effective_diffusivity_support(D_k_ch4, porosity, tortuosity)

    J_co2 = D_eff_co2 * (c_feed["CO2"] - c_perm["CO2"]) / length
    J_ch4 = D_eff_ch4 * (c_feed["CH4"] - c_perm["CH4"]) / length

    total_J_co2 = float(np.mean(J_co2))
    total_J_ch4 = float(np.mean(J_ch4))
    return total_J_co2, total_J_ch4, radii


def pore_resistance_network(pore_radii, pore_lengths, D_knudsen_values):
    """
    Compute total resistance of pores in parallel:
        1/R_total = sum_i (1 / R_i)
        R_i = L_i / (A_i * D_i)
    Approximate cross-sectional area A_i = pi r_i^2.
    """
    resistances = pore_lengths / (np.pi * (pore_radii ** 2) * D_knudsen_values + 1e-30)
    R_total = 1.0 / np.sum(1.0 / resistances)
    return R_total


def runge_mesh_adaptation_nodes(n_base, L):
    """
    Generate non-uniform FEM nodes clustered near boundaries using the
    derivative of the Runge function as a monitor function.
    """
    # Uniform parameter xi in [-1,1]
    xi = np.linspace(-1.0, 1.0, n_base)
    # Monitor function based on |f''(xi)| of Runge function
    monitor = np.abs(-50.0 * (75.0 * xi ** 2 - 1.0) / ((1.0 + 25.0 * xi ** 2) ** 3))
    monitor = monitor + 1e-3  # regularization
    # Integrate monitor to get mapping
    cum = np.cumsum(monitor)
    cum = cum / cum[-1]
    # Map back to physical domain [0, L]
    x = cum * L
    x[0] = 0.0
    x[-1] = L
    return x
