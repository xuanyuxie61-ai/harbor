
import numpy as np


def integrate_over_spherical_mesh(values, faces, areas):
    integral = 0.0
    for i, tri in enumerate(faces):
        avg_val = np.mean(values[tri])
        integral += avg_val * areas[i]
    return integral


def pyramid_witherden_rule_3d(f_vals, volume):
    return np.mean(f_vals) * volume


def compute_global_energy(T, faces, areas, heat_capacity=2.5e8):
    energy_density = heat_capacity * np.asarray(T, dtype=np.float64)
    return integrate_over_spherical_mesh(energy_density, faces, areas)


def compute_radiative_imbalance(T, vertices, faces, areas, epsilon=0.6, Q_solar=340.25):
    from ebm_dynamics import ice_albedo_feedback, outgoing_longwave_radiation, SIGMA











    pass


def conservation_diagnostics(T_history, vertices, faces, dt_years, areas,
                             heat_capacity=2.5e8, epsilon=0.6, Q_solar=340.25):
    n_steps = len(T_history)
    energies = np.zeros(n_steps, dtype=np.float64)
    imbalances = np.zeros(n_steps, dtype=np.float64)

    for i in range(n_steps):
        energies[i] = compute_global_energy(T_history[i], faces, areas, heat_capacity)
        imbalances[i] = compute_radiative_imbalance(
            T_history[i], vertices, faces, areas, epsilon, Q_solar
        )


    dt_sec = dt_years * 365.25 * 24 * 3600
    dEdt = np.zeros(n_steps, dtype=np.float64)
    dEdt[1:-1] = (energies[2:] - energies[:-2]) / (2.0 * dt_sec)
    dEdt[0] = (energies[1] - energies[0]) / dt_sec
    dEdt[-1] = (energies[-1] - energies[-2]) / dt_sec

    residual = dEdt - imbalances
    return {
        'energies': energies,
        'imbalances': imbalances,
        'dEdt': dEdt,
        'residual': residual,
        'max_residual': float(np.max(np.abs(residual))),
        'mean_residual': float(np.mean(np.abs(residual)))
    }
