
import numpy as np
from numpy.linalg import norm


def bubble_energy_budget(R, dRdt, p_inf, p_v, sigma, rho, c_sound):

    E_k = 2.0 * np.pi * rho * (R ** 3) * (dRdt ** 2)


    E_p = (4.0 / 3.0) * np.pi * (R ** 3) * (p_inf - p_v)


    E_s = 4.0 * np.pi * sigma * (R ** 2)



    p_wall = p_v - 2.0 * sigma / R - 4.0 * 1.002e-3 * dRdt / R
    P_acoustic = (p_wall ** 2) * 4.0 * np.pi * (R ** 2) / (rho * c_sound + 1e-30)

    return {
        'kinetic': E_k,
        'potential': E_p,
        'surface': E_s,
        'acoustic_power': P_acoustic,
        'total': E_k + E_p + E_s
    }


def energy_dissipation_path(R_history, dRdt_history, dt, p_inf, p_v, sigma, rho, c_sound):
    energy_history = []
    cumulative_acoustic = 0.0

    for R, dRdt in zip(R_history, dRdt_history):
        energies = bubble_energy_budget(R, dRdt, p_inf, p_v, sigma, rho, c_sound)
        cumulative_acoustic += energies['acoustic_power'] * dt
        energies['cumulative_acoustic'] = cumulative_acoustic
        energy_history.append(energies)

    return energy_history


def collapse_efficiency(R0, R_min, p_inf, p_v, sigma, rho, c_sound):
    E_initial = bubble_energy_budget(R0, 0.0, p_inf, p_v, sigma, rho, c_sound)
    E_total_0 = E_initial['total']


    E_k_max_approx = (4.0 / 3.0) * np.pi * (R0 ** 3) * (p_inf - p_v)

    if E_total_0 > 1e-30:
        eta = E_k_max_approx / E_total_0
    else:
        eta = 0.0
    return min(eta, 1.0)


def random_search_energy_allocation(N_stages, energy_budget, efficiency_weights, num_samples=5000):
    w_k, w_a, w_h = efficiency_weights
    best_cost = -np.inf
    best_allocation = None

    for _ in range(num_samples):

        ratios = np.random.dirichlet(np.ones(3), size=N_stages)
        allocation = ratios * (energy_budget / N_stages)


        total_kinetic = np.sum(allocation[:, 0])
        total_acoustic = np.sum(allocation[:, 1])
        total_heat = np.sum(allocation[:, 2])


        efficiency = (w_k * total_kinetic + w_a * total_acoustic - w_h * total_heat)

        if np.any(allocation < 0):
            efficiency -= 1e10

        if efficiency > best_cost:
            best_cost = efficiency
            best_allocation = allocation.copy()

    return best_allocation, best_cost


def optimize_collapse_parameters(p_inf_range, R0_range, p_v, sigma, rho, c_sound,
                                  num_samples=1000):
    best_efficiency = -1.0
    best_params = None
    results = []

    for _ in range(num_samples):
        p_inf = np.random.uniform(p_inf_range[0], p_inf_range[1])
        R0 = np.random.uniform(R0_range[0], R0_range[1])

        eta = collapse_efficiency(R0, 1e-6, p_inf, p_v, sigma, rho, c_sound)
        results.append((p_inf, R0, eta))

        if eta > best_efficiency:
            best_efficiency = eta
            best_params = (p_inf, R0)

    return best_params, best_efficiency, np.array(results)


def energy_spectrum_analysis(R_history, dRdt_history, dt):
    N = len(R_history)
    if N < 4:
        return np.array([]), np.array([])

    freqs = np.fft.rfftfreq(N, d=dt)
    R_fft = np.abs(np.fft.rfft(R_history - np.mean(R_history)))
    v_fft = np.abs(np.fft.rfft(dRdt_history))

    return freqs, R_fft, v_fft
