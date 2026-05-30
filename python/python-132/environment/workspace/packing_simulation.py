
import numpy as np
from utils import ensure_positive






def line_packing_simulation(x_min, x_max, seg_width, max_attempts=100000):
    seg_rad = seg_width / 2.0
    available_length = x_max - x_min

    if available_length <= seg_width:
        return 0, 0.0, 0.0, np.array([])

    positions = []
    n_parked = 0
    latest_success = 0

    for attempt in range(1, max_attempts + 1):
        r = np.random.rand()
        pos = x_min + seg_rad + r * (available_length - 2.0 * seg_rad)

        if len(positions) == 0:
            min_dist = 2.0 * seg_rad + 1e-12
        else:
            min_dist = np.min(np.abs(np.array(positions) - pos))

        if min_dist >= 2.0 * seg_rad:
            positions.append(pos)
            n_parked += 1
            latest_success = attempt
        elif attempt - latest_success > 10000:
            break

    positions = np.array(positions, dtype=float)
    density_max = 1.0 / (2.0 * seg_rad)
    density_obs = n_parked / available_length if available_length > 0 else 0.0

    return n_parked, density_obs, density_max, positions






def packing_void_fraction(n_packing, packing_diameter, column_diameter,
                          packing_height, packing_shape_factor=1.0):
    V_column = np.pi / 4.0 * (column_diameter ** 2) * packing_height
    V_single = np.pi / 4.0 * (packing_diameter ** 2) * packing_diameter * packing_shape_factor
    V_packing_total = n_packing * V_single

    epsilon = 1.0 - V_packing_total / ensure_positive(V_column, name="V_column")
    epsilon = float(np.clip(epsilon, 0.2, 0.98))
    return epsilon


def packing_efficiency_factor(epsilon, epsilon_ref=0.9, eta_ref=1.0):
    epsilon = np.clip(epsilon, 0.2, 0.98)
    epsilon_ref = max(epsilon_ref, 0.2)
    eta = eta_ref * (1.0 - epsilon) / (1.0 - epsilon_ref)
    eta = eta * np.exp(-2.0 * (epsilon - 0.7) ** 2)
    return float(np.clip(eta, 0.1, 2.0))


def ergun_pressure_drop(epsilon, mu, u, rho, d_p, L_packing):
    epsilon = max(epsilon, 0.2)
    mu = max(mu, 1e-6)
    u = max(u, 0.0)
    rho = max(rho, 0.01)
    d_p = max(d_p, 1e-6)
    L_packing = max(L_packing, 1e-6)

    term1 = 150.0 * ((1.0 - epsilon) ** 2) * mu * u / (epsilon ** 3 * d_p ** 2)
    term2 = 1.75 * (1.0 - epsilon) * rho * (u ** 2) / (epsilon ** 3 * d_p)

    dP = (term1 + term2) * L_packing
    return dP


def simulate_random_packing_column(column_diameter, packing_height,
                                    packing_diameter, packing_shape_factor,
                                    mu, u, rho, n_runs=10):
    epsilons = []
    etas = []
    dPs = []
    densities = []


    circumference = np.pi * column_diameter

    for _ in range(n_runs):
        n_parked, density_obs, density_max, positions = line_packing_simulation(
            0.0, circumference, packing_diameter
        )
        densities.append(density_obs)


        area_fraction = density_obs * packing_diameter
        n_packing_est = int(area_fraction * (column_diameter / packing_diameter) ** 2
                            * (packing_height / packing_diameter))
        n_packing_est = max(n_packing_est, 1)

        eps = packing_void_fraction(n_packing_est, packing_diameter,
                                     column_diameter, packing_height,
                                     packing_shape_factor)
        epsilons.append(eps)

        eta = packing_efficiency_factor(eps)
        etas.append(eta)

        dP = ergun_pressure_drop(eps, mu, u, rho, packing_diameter, packing_height)
        dPs.append(dP)

    results = {
        "epsilon_mean": float(np.mean(epsilons)),
        "epsilon_std": float(np.std(epsilons)),
        "eta_mean": float(np.mean(etas)),
        "eta_std": float(np.std(etas)),
        "dP_mean": float(np.mean(dPs)),
        "dP_std": float(np.std(dPs)),
        "density_mean": float(np.mean(densities)),
        "n_runs": n_runs
    }
    return results
