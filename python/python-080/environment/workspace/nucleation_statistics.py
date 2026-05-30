
import numpy as np
from utils import disk01_sample, ellipsoid_sample, BOLTZMANN




SURFACE_TENSION = 0.0728
VAPOR_PRESSURE = 2338.0
AMBIENT_TEMPERATURE = 293.15


def lognormal_nuclei_distribution(R, mu_R, sigma_R):
    ln_R = np.log(np.maximum(R, 1e-15))
    ln_mu = np.log(mu_R)
    coeff = 1.0 / (R * sigma_R * np.sqrt(2.0 * np.pi))
    exponent = -0.5 * ((ln_R - ln_mu) / sigma_R) ** 2
    return coeff * np.exp(exponent)


def nucleation_barrier_energy(p_inf, p_v, sigma):
    delta_p = p_v - p_inf
    if abs(delta_p) < 1.0:
        delta_p = np.sign(delta_p) * 1.0
    return 16.0 * np.pi * sigma**3 / (3.0 * delta_p**2)


def nucleation_rate(p_inf, p_v, sigma, T, J0=1e30):
    delta_G = nucleation_barrier_energy(p_inf, p_v, sigma)
    return J0 * np.exp(-delta_G / (BOLTZMANN * T + 1e-30))


def sample_nuclei_monte_carlo(num_samples, mu_R, sigma_R, method='disk'):

    u = np.random.uniform(0.0, 1.0, size=num_samples)
    radii = mu_R * np.exp(sigma_R * np.random.randn(num_samples))

    if method == 'disk':

        positions = disk01_sample(num_samples)
    elif method == 'ellipsoid':

        A = np.array([[2.0, 0.5], [0.5, 1.5]])
        v = np.array([0.0, 0.0])
        positions = ellipsoid_sample(2, num_samples, A, v, 1.0)
    else:
        positions = np.random.randn(2, num_samples)
        norms = np.sqrt(np.sum(positions**2, axis=0))
        norms = np.maximum(norms, 1e-15)
        positions = positions / norms * np.random.uniform(0.0, 1.0, size=num_samples)

    return positions, radii


def full_deck_nucleation_stats(num_experiments, p_inf_range, p_v, sigma, T, surface_area):
    nucleation_counts = []
    dt = 1e-6

    for _ in range(num_experiments):
        p_inf = np.random.choice(p_inf_range)
        J = nucleation_rate(p_inf, p_v, sigma, T)
        expected_nuclei = J * surface_area * dt

        num_nuclei = np.random.poisson(max(expected_nuclei, 0.0))
        nucleation_counts.append(num_nuclei)

    nucleation_counts = np.array(nucleation_counts, dtype=float)
    stats = {
        'min': int(np.min(nucleation_counts)),
        'max': int(np.max(nucleation_counts)),
        'mean': np.mean(nucleation_counts),
        'variance': np.var(nucleation_counts),
        'median': np.median(nucleation_counts),
    }
    return stats


def vacancy_activation_probability(num_sites, p_inf, p_v, sigma, T, span_years=1.0):
    J = nucleation_rate(p_inf, p_v, sigma, T)

    p_no_activate_single = np.exp(-J * span_years * 1e-6)

    p_no_activate_all = p_no_activate_single ** num_sites

    p_activation = 1.0 - p_no_activate_all
    return p_activation


def critical_nuclei_fraction(p_inf, p_v, sigma, T, R0, mu_R, sigma_R):
    delta_p = p_v - p_inf
    if delta_p >= 0:
        return 1.0
    R_crit = np.sqrt(2.0 * sigma / (3.0 * abs(delta_p)))


    from scipy.stats import lognorm
    cdf = lognorm.cdf(R_crit, s=sigma_R, scale=mu_R)
    return 1.0 - cdf


def surface_site_occupancy(num_sites, nuclei_positions, site_radius=1e-5):
    occupied = set()
    for pos in nuclei_positions.T:

        ix = int(pos[0] / site_radius)
        iy = int(pos[1] / site_radius)
        occupied.add((ix, iy))

    occupancy_rate = len(occupied) / max(num_sites, 1)
    return occupancy_rate


def nucleation_event_times(poisson_rate, t_max, num_realizations=10):
    events = []
    for _ in range(num_realizations):
        t = 0.0
        realization = []
        while t < t_max:
            dt = np.random.exponential(1.0 / max(poisson_rate, 1e-15))
            t += dt
            if t < t_max:
                realization.append(t)
        events.append(realization)
    return events
