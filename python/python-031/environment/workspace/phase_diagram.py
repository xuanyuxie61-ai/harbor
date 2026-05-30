# -*- coding: utf-8 -*-

import numpy as np
from nuclear_eos import nuclear_matter_properties, skyrme_energy_density
from geometry_pasta import create_pasta_phase, PastaPhase
from coulomb_solver import analytical_coulomb
from bessel_modes import pasta_deformation_energy


E_CHARGE = 1.43996448
K_B = 8.617333262e-11


def surface_tension(rho, proton_fraction, params=None):
    sigma_0 = 0.5
    kappa_s = 2.6
    rho_0 = 0.16
    p = 1.5
    I = 1.0 - 2.0 * proton_fraction
    sigma = sigma_0 * (1.0 - kappa_s * I**2) * (rho / rho_0)**p
    return max(0.05, sigma)


def lattice_energy(density, proton_fraction):
    R_WS = (3.0 / (4.0 * np.pi * density)) ** (1.0 / 3.0)
    Z_eff = proton_fraction
    E_lat = -0.9 * (Z_eff * E_CHARGE)**2 / (2.0 * R_WS)
    return E_lat


def total_energy_per_nucleon(phase_id, density, proton_fraction, temperature=0.0,
                              u=None, include_shell=False):
    if density <= 0.0 or proton_fraction < 0.0 or proton_fraction > 1.0:
        return np.inf, {}

    try:
        phase = create_pasta_phase(phase_id, density, proton_fraction, u)
    except ValueError:
        return np.inf, {}


    props = nuclear_matter_properties(density, proton_fraction)
    E_bulk = props['energy_per_nucleon']


    sigma = surface_tension(density, proton_fraction)
    E_surf = sigma * phase.surface_to_volume() / density


    E_coul = analytical_coulomb(phase_id, density, proton_fraction, u)


    E_lat = lattice_energy(density, proton_fraction)


    if temperature > 0.0:

        m_n = 939.565
        k_f = (3.0 * np.pi**2 * density) ** (1.0 / 3.0)
        eps_f = k_f**2 / (2.0 * m_n)
        N0 = 3.0 * density / (2.0 * eps_f)
        S = (np.pi**2 / 2.0) * N0 * K_B**2 * temperature
        E_thermal = -temperature * S / density
    else:
        E_thermal = 0.0


    E_shell = 0.0
    if include_shell:

        E_shell = 2.0 * np.sin(np.pi * proton_fraction * 100.0) / (proton_fraction * 100.0)




    shape_correction = 0.0
    rho_s = density / 0.16
    if phase_id == 1:
        shape_correction = -4.0 * np.exp(-rho_s / 0.08)
    elif phase_id == 2:
        shape_correction = -3.5 * np.exp(-(rho_s - 0.3)**2 / 0.02)
    elif phase_id == 3:
        shape_correction = -3.0 * np.exp(-(rho_s - 0.5)**2 / 0.02)
    elif phase_id == 4:
        shape_correction = -2.5 * np.exp(-(rho_s - 0.7)**2 / 0.02)
    elif phase_id == 5:
        shape_correction = -2.0 * np.exp(-(rho_s - 0.9)**2 / 0.02)

    E_total = E_bulk + E_surf + E_coul + E_lat + E_thermal + E_shell + shape_correction

    components = {
        'bulk': E_bulk,
        'surface': E_surf,
        'coulomb': E_coul,
        'lattice': E_lat,
        'thermal': E_thermal,
        'shell': E_shell,
        'shape_correction': shape_correction,
        'total': E_total,
    }

    return E_total, components


def optimal_filling(phase_id, density, proton_fraction, n_u=50):
    u_grid = np.linspace(0.05, 0.95, n_u)
    energies = []

    for u in u_grid:
        E, _ = total_energy_per_nucleon(phase_id, density, proton_fraction, u=u)
        energies.append(E)

    energies = np.array(energies)
    i_min = np.argmin(energies)
    u_opt = u_grid[i_min]
    E_min = energies[i_min]


    if u_opt <= 0.06:
        u_opt = 0.1
    if u_opt >= 0.94:
        u_opt = 0.9

    return u_opt, E_min


def compute_phase_diagram(density_range, temperature_range, proton_fraction=0.3):
    n_rho = len(density_range)
    n_T = len(temperature_range)

    phase_map = np.zeros((n_T, n_rho), dtype=int)
    energy_map = np.full((n_T, n_rho, 5), np.inf)

    for i_T, T in enumerate(temperature_range):
        for i_rho, rho in enumerate(density_range):
            E_min = np.inf
            best_phase = 0

            for pid in range(1, 6):
                try:
                    u_opt, E = optimal_filling(pid, rho, proton_fraction)
                    E_total, _ = total_energy_per_nucleon(
                        pid, rho, proton_fraction, temperature=T, u=u_opt
                    )
                    energy_map[i_T, i_rho, pid - 1] = E_total

                    if E_total < E_min:
                        E_min = E_total
                        best_phase = pid
                except Exception:
                    continue

            phase_map[i_T, i_rho] = best_phase

    return phase_map, energy_map


def stability_analysis(density, proton_fraction, phase_id, u=None,
                       temperature=0.0, n_modes=5):
    phase = create_pasta_phase(phase_id, density, proton_fraction, u)


    dr = 1e-4 * density
    _, P1 = skyrme_energy_density(
        density * (1.0 - dr) * (1.0 - proton_fraction),
        density * (1.0 - dr) * proton_fraction
    )
    _, P2 = skyrme_energy_density(
        density * (1.0 + dr) * (1.0 - proton_fraction),
        density * (1.0 + dr) * proton_fraction
    )
    dP_drho = (P2 - P1) / (2.0 * dr * density)
    mechanical_stable = dP_drho > 0.0


    dx = 1e-3
    E1, _ = total_energy_per_nucleon(phase_id, density, proton_fraction - dx, temperature, u)
    E0, _ = total_energy_per_nucleon(phase_id, density, proton_fraction, temperature, u)
    E2, _ = total_energy_per_nucleon(phase_id, density, proton_fraction + dx, temperature, u)
    d2E_dx2 = (E1 - 2.0 * E0 + E2) / dx**2
    chemical_stable = d2E_dx2 > 0.0


    sigma = surface_tension(density, proton_fraction)
    R = getattr(phase, 'R', phase.a_WS)
    modes = []
    for m in range(2, n_modes + 2):
        dE = pasta_deformation_energy(phase_id, R, 0.1, m, sigma)
        stable_mode = dE > 0.0
        modes.append({
            'mode': m,
            'deformation_energy': dE,
            'stable': stable_mode
        })

    stable = mechanical_stable and chemical_stable and all(m['stable'] for m in modes)

    return {
        'stable': stable,
        'mechanical_stable': mechanical_stable,
        'chemical_stable': chemical_stable,
        'dP_drho': dP_drho,
        'd2E_dx2': d2E_dx2,
        'modes': modes
    }


def transition_density(phase_id_1, phase_id_2, proton_fraction=0.3,
                       rho_min=0.01, rho_max=0.2, n_points=100):
    rho_grid = np.linspace(rho_min, rho_max, n_points)
    diff = []

    for rho in rho_grid:
        try:
            u1, _ = optimal_filling(phase_id_1, rho, proton_fraction)
            u2, _ = optimal_filling(phase_id_2, rho, proton_fraction)
            E1, _ = total_energy_per_nucleon(phase_id_1, rho, proton_fraction, u=u1)
            E2, _ = total_energy_per_nucleon(phase_id_2, rho, proton_fraction, u=u2)
            diff.append(E1 - E2)
        except Exception:
            diff.append(np.nan)

    diff = np.array(diff)

    for i in range(len(diff) - 1):
        if not (np.isfinite(diff[i]) and np.isfinite(diff[i + 1])):
            continue
        if diff[i] * diff[i + 1] < 0:

            rho_trans = rho_grid[i] + (rho_grid[i + 1] - rho_grid[i]) * abs(diff[i]) / (
                abs(diff[i]) + abs(diff[i + 1])
            )
            return rho_trans, True

    return None, False


if __name__ == '__main__':

    for pid in range(1, 6):
        E, comp = total_energy_per_nucleon(pid, 0.08, 0.3)
        print(f"Phase {pid}: E/A={E:.2f} MeV, components={comp}")

    rho_t, found = transition_density(1, 2, 0.3)
    if found:
        print(f"Gnocchi->Spaghetti transition at rho={rho_t:.4f} fm^-3")
