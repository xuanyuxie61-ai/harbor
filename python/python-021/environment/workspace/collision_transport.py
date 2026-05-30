
import numpy as np
from parameters import (
    EPS0, QE, ME, KB, N_E_AXIS, T_E_AXIS, Z_EFF
)


def coulomb_logarithm(n_e, T_e_eV):
    n_e = np.asarray(n_e)
    T_e = np.asarray(T_e_eV)
    T_e_safe = np.where(T_e < 1.0, 1.0, T_e)
    lnL = 31.3 - np.log(np.sqrt(n_e) / T_e_safe)
    return np.clip(lnL, 5.0, 25.0)


def electron_ion_collision_frequency(n_e, T_e_eV, Z_eff=Z_EFF):
    n_e = np.asarray(n_e)
    T_e = np.asarray(T_e_eV) * QE
    T_e_safe = np.where(T_e < 1e-20, 1e-20, T_e)
    lnL = coulomb_logarithm(n_e, T_e_eV)

    numerator = n_e * Z_eff * (QE ** 4) * lnL
    denominator = 3.0 * (2.0 * np.pi) ** 1.5 * (EPS0 ** 2) * np.sqrt(ME) * (T_e_safe ** 1.5)
    nu = numerator / (denominator + 1e-50)
    return nu


def thermal_velocity_electron(T_e_eV):
    T_e_J = np.asarray(T_e_eV) * QE
    return np.sqrt(2.0 * KB * T_e_J / ME)


def mean_free_path(n_e, T_e_eV, Z_eff=Z_EFF):
    vth = thermal_velocity_electron(T_e_eV)
    nu = electron_ion_collision_frequency(n_e, T_e_eV, Z_eff)
    return vth / (nu + 1e-50)


def hypersphere_velocity_sampling(m_dim, n_samples, T_e_eV):
    if m_dim < 2:
        raise ValueError("维数必须 ≥ 2")

    costs = np.zeros(n_samples)
    for i in range(n_samples):

        p = np.random.randn(m_dim)
        p /= np.linalg.norm(p)
        q = np.random.randn(m_dim)
        q /= np.linalg.norm(q)
        costs[i] = abs(np.dot(p, q))

    costs = np.clip(costs, 0.0, 1.0)
    thetas = np.arccos(costs)

    stats = {
        "dim": m_dim,
        "temperature_eV": T_e_eV,
        "cos_mean": float(np.mean(costs)),
        "cos_std": float(np.std(costs)),
        "theta_mean_rad": float(np.mean(thetas)),
        "theta_std_rad": float(np.std(thetas)),
        "theta_mean_deg": float(np.degrees(np.mean(thetas))),
    }
    return stats


def rectangle_collision_distance_stats(a, b, n_samples=100000):
    if a <= 0 or b <= 0:
        raise ValueError("矩形边长必须为正")

    x1 = np.random.uniform(0, a, n_samples)
    y1 = np.random.uniform(0, b, n_samples)
    x2 = np.random.uniform(0, a, n_samples)
    y2 = np.random.uniform(0, b, n_samples)

    d = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    stats = {
        "a": a,
        "b": b,
        "mean_distance": float(np.mean(d)),
        "std_distance": float(np.std(d)),
        "max_distance": float(np.max(d)),
        "min_distance": float(np.min(d)),
        "median_distance": float(np.median(d)),
    }
    return stats


def compute_transport_coefficients(n_e, T_e_eV, B, Z_eff=Z_EFF, q=2.0, R0=6.2, a=2.0):
    from parameters import MD

    nu_ei = electron_ion_collision_frequency(n_e, T_e_eV, Z_eff)
    vth_e = thermal_velocity_electron(T_e_eV)
    rho_e = ME * vth_e / (QE * B + 1e-30)


    T_i_J = T_e_eV * QE
    vth_i = np.sqrt(2.0 * KB * T_i_J / MD)
    rho_i = MD * vth_i / (QE * B + 1e-30)

    epsilon = a / (R0 + 1e-20)
    epsilon_safe = max(epsilon, 1e-6)

    D_cl = nu_ei * rho_e ** 2
    D_neo = (q ** 2) * nu_ei * (rho_i ** 2) / (epsilon_safe ** 1.5)


    chi_e = D_cl * np.sqrt(MD / ME)
    chi_i = D_neo

    return {
        "nu_ei_Hz": float(nu_ei),
        "lambda_e_m": float(vth_e / (nu_ei + 1e-50)),
        "rho_e_m": float(rho_e),
        "rho_i_m": float(rho_i),
        "D_classical_m2s": float(D_cl),
        "D_neoclassical_m2s": float(D_neo),
        "chi_e_m2s": float(chi_e),
        "chi_i_m2s": float(chi_i),
    }
