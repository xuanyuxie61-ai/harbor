
import numpy as np


def biot_wave_velocities_low_freq(material):
    v_fast = material.V_p_fast
    v_slow = material.V_p_slow
    v_shear = material.V_s
    return v_fast, v_slow, v_shear


def biot_dispersion_relation(omega, material):
    omega = np.asarray(omega, dtype=float)


    omega_c = (material.eta * material.phi) / (material.kappa * material.rho_f)



    alpha_infty = material.tortuosity
    dynamic_tortuosity = alpha_infty * np.sqrt(1.0 + omega_c / (1j * omega + 1e-30))


    rho_f_eff = material.rho_f * dynamic_tortuosity


    K_d = material.K_d
    mu = material.mu
    alpha = material.alpha
    M = material.M
    rho_s = material.rho_s
    rho_f = material.rho_f
    phi = material.phi
    rho_bulk = material.rho_bulk


    K_u = K_d + alpha ** 2 * M


    H = K_d + 4.0 * mu / 3.0 + alpha ** 2 * M



    v_fast = np.sqrt((K_u + 4.0 * mu / 3.0) / rho_bulk)

    v_fast = v_fast * (1.0 - 0.5j * (omega / omega_c) / (1.0 + (omega / omega_c) ** 2))


    D_diff = material.D_diff
    v_slow = np.sqrt(2.0 * omega * D_diff) * (1.0 + 0.5j)


    alpha_fast = np.imag(omega / v_fast)
    alpha_slow = np.imag(omega / v_slow)

    return v_fast, v_slow, alpha_fast, alpha_slow


def compute_quality_factor(omega, material):
    v_fast, v_slow, alpha_fast, alpha_slow = biot_dispersion_relation(omega, material)
    k_fast = omega / v_fast
    k_slow = omega / v_slow

    Q_fast_inv = 2.0 * np.abs(np.imag(k_fast)) / (np.abs(np.real(k_fast)) + 1e-30)
    Q_slow_inv = 2.0 * np.abs(np.imag(k_slow)) / (np.abs(np.real(k_slow)) + 1e-30)

    Q_fast = 1.0 / (Q_fast_inv + 1e-30)
    Q_slow = 1.0 / (Q_slow_inv + 1e-30)

    return Q_fast, Q_slow


def separate_fast_slow_waves(pressure, displacement, nodes, material, dt, dx_est):
    p = np.asarray(pressure)
    u = np.asarray(displacement)
    u_mag = np.linalg.norm(u, axis=1) + 1e-30
    ratio = np.abs(p) / u_mag

    v_fast, v_slow, _ = biot_wave_velocities_low_freq(material)




    threshold = np.median(ratio)

    fast_mask = ratio < threshold
    slow_mask = ratio >= threshold

    return fast_mask, slow_mask, ratio


def compute_wave_energy_flux(pressure, displacement, velocity, material):
    n = len(pressure)


    J = np.zeros((n, 2))


    sigma_approx = material.lam * np.linalg.norm(displacement, axis=1) + \
                   2.0 * material.mu * np.linalg.norm(displacement, axis=1)
    J[:, 0] += sigma_approx * velocity[:, 0]
    J[:, 1] += sigma_approx * velocity[:, 1]


    kappa_eta = material.kappa / material.eta

    J[:, 0] += pressure * kappa_eta * pressure
    J[:, 1] += pressure * kappa_eta * pressure

    return J


def dispersion_error_analysis(v_num, v_exact):
    return np.abs(v_num - v_exact) / (np.abs(v_exact) + 1e-30)
