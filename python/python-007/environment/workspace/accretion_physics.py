import numpy as np






G_GRAV = 6.67430e-11
C_LIGHT = 2.99792458e8
M_SUN = 1.98847e30
SIGMA_SB = 5.670374419e-8
K_BOLTZMANN = 1.380649e-23
MP = 1.6726219e-27
MU = 0.6
GAMMA_AD = 5.0 / 3.0


def keplerian_angular_velocity(r, M_bh):
    r = np.asarray(r, dtype=np.float64)
    r = np.where(r < 1e-3, 1e-3, r)
    return np.sqrt(G_GRAV * M_bh / r ** 3)


def sound_speed(T, mu=MU, gamma=GAMMA_AD):
    return np.sqrt(gamma * K_BOLTZMANN * T / (mu * MP))


def scale_height(r, M_bh, T, mu=MU):
    cs = sound_speed(T, mu)
    omega = keplerian_angular_velocity(r, M_bh)
    return cs / omega


def shakura_sunyaev_sigma(r, M_dot, M_bh, alpha, mu=MU):
    r = np.asarray(r, dtype=np.float64)
    r = np.where(r < 1e-3, 1e-3, r)


    r_isco = 6.0 * G_GRAV * M_bh / C_LIGHT ** 2
    r_eff = np.maximum(r, r_isco * 1.001)


    factor = np.clip(1.0 - np.sqrt(r_isco / r_eff), 1e-15, 1.0)
    T = ((3.0 * G_GRAV * M_bh * M_dot) /
         (8.0 * np.pi * SIGMA_SB * r_eff ** 3) * factor) ** 0.25
    T = np.where(T < 1.0, 1.0, T)


    H = scale_height(r_eff, M_bh, T, mu)
    H = np.where(H < 1e-10, 1e-10, H)


    cs = sound_speed(T, mu)
    cs = np.where(cs < 1e-10, 1e-10, cs)


    nu = alpha * cs * H
    nu = np.where(nu < 1e-30, 1e-30, nu)


    Sigma = M_dot / (3.0 * np.pi * nu) * factor
    Sigma = np.where(Sigma < 0, 0.0, Sigma)

    return Sigma, T, H


def viscous_torque(r, Sigma, M_bh, alpha, mu=MU):
    r = np.asarray(r, dtype=np.float64)
    omega = keplerian_angular_velocity(r, M_bh)
    Sigma = np.asarray(Sigma)



    cs_approx = alpha * omega * r * 0.1
    T_approx = cs_approx ** 2 * mu * MP / (GAMMA_AD * K_BOLTZMANN)
    H = scale_height(r, M_bh, T_approx, mu)
    cs = sound_speed(T_approx, mu)
    nu = alpha * cs * H

    G = 3.0 * np.pi * nu * Sigma * r ** 2 * omega
    return G


def schwarzschild_potential(r, M_bh):
    r = np.asarray(r, dtype=np.float64)
    r = np.where(np.abs(r) < 1e-3, 1e-3, r)
    return -G_GRAV * M_bh / r


def schwarzschild_metric_correction(r, M_bh):
    r = np.asarray(r, dtype=np.float64)
    r = np.where(np.abs(r) < 1e-3, 1e-3, r)
    correction = 1.0 - 3.0 * G_GRAV * M_bh / (r * C_LIGHT ** 2)
    return correction


def paczynski_wiita_potential(r, M_bh):
    r_s = 2.0 * G_GRAV * M_bh / C_LIGHT ** 2
    r = np.asarray(r, dtype=np.float64)
    r_safe = np.maximum(r, r_s * 1.001)
    return -G_GRAV * M_bh / (r_safe - r_s)


def jet_launching_criterion(r, B_z, rho, M_bh):
    r = np.asarray(r, dtype=np.float64)
    r = np.where(r < 1e-3, 1e-3, r)



    v_A = None
    v_esc = np.sqrt(2.0 * G_GRAV * M_bh / r)

    launched = v_A > v_esc
    return launched, v_A, v_esc


def magnetic_braking_torque(r, B_phi, B_r, Sigma, M_bh):
    r = np.asarray(r, dtype=np.float64)
    return r ** 2 * B_r * B_phi / (2.0 * np.pi)


def disk_spectrum_nu(nu_freq, r_in, r_out, M_dot, M_bh):
    h_planck = 6.62607015e-34
    nu = np.asarray(nu_freq, dtype=np.float64)


    n_r = 100
    r = np.linspace(r_in, r_out, n_r)
    dr = r[1] - r[0]


    Sigma, T, H = shakura_sunyaev_sigma(r, M_dot, M_bh, alpha=0.1)


    L_nu = np.zeros_like(nu)
    for i in range(n_r):
        T_i = T[i]
        if T_i < 1.0 or np.isnan(T_i) or np.isinf(T_i):
            continue
        x = h_planck * nu / (K_BOLTZMANN * T_i)

        x = np.clip(x, 1e-10, 700.0)
        exp_x = np.exp(x)
        B_nu = (2.0 * h_planck * nu ** 3 / C_LIGHT ** 2) / (exp_x - 1.0)
        B_nu = np.where(np.isfinite(B_nu), B_nu, 0.0)
        L_nu += 2.0 * np.pi * r[i] * B_nu * dr

    L_nu = np.where(np.isfinite(L_nu), L_nu, 0.0)
    return L_nu


def disk_instability_criterion(Sigma, T_actual, r, M_bh, alpha, mu=MU):
    r = np.asarray(r, dtype=np.float64)
    Sigma = np.asarray(Sigma, dtype=np.float64)
    T = np.asarray(T_actual, dtype=np.float64)
    T = np.where(T < 1.0, 1.0, T)

    H = scale_height(r, M_bh, T, mu)
    cs = sound_speed(T, mu)
    nu = alpha * cs * H
    nu = np.where(nu < 1e-30, 1e-30, nu)

    t_visc = r ** 2 / nu
    t_visc = np.where(t_visc < 1e-10, 1e-10, t_visc)

    t_cool = Sigma * cs ** 2 / (2.0 * SIGMA_SB * T ** 4)
    t_cool = np.where(t_cool < 1e-10, 1e-10, t_cool)


    ratio = t_cool / t_visc
    unstable = ratio < 1e-4
    return unstable, t_visc, t_cool


def compute_radial_velocity(Sigma, r, M_bh, alpha, M_dot):
    r = np.asarray(r, dtype=np.float64)
    Sigma = np.asarray(Sigma, dtype=np.float64)

    r_isco = 6.0 * G_GRAV * M_bh / C_LIGHT ** 2
    r_eff = np.maximum(r, r_isco)


    omega = keplerian_angular_velocity(r_eff, M_bh)
    cs_approx = alpha * omega * r_eff * 0.1
    T_approx = cs_approx ** 2 * MU * MP / (GAMMA_AD * K_BOLTZMANN)
    H = scale_height(r_eff, M_bh, T_approx, MU)
    cs = sound_speed(T_approx, MU)
    nu = alpha * cs * H

    sqrt_ratio = np.sqrt(r_isco / r_eff)
    numerator = 1.0 - sqrt_ratio
    denominator = 1.0 - (2.0 / 3.0) * sqrt_ratio
    denominator = np.where(np.abs(denominator) < 1e-15, 1e-15, denominator)

    v_r = -1.5 * nu / r_eff * numerator / denominator
    return v_r
