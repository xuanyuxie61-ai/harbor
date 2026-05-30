
import numpy as np


E_CHARGE = 1.602176634e-19
M_U = 1.66053906660e-27
K_B = 1.380649e-23


Q_DT = 17.6e6 * E_CHARGE
E_ALPHA = 3.5e6 * E_CHARGE
E_NEUTRON = 14.1e6 * E_CHARGE


Q_DD = 4.0e6 * E_CHARGE


def bosch_hale_sigmav(T_keV, reaction='DT'):
    T = float(max(T_keV, 0.1))

    if reaction == 'DT':

        B_G = 34.3827
        m_rc2 = 1124656.0
        C1, C2, C3, C4, C5, C6, C7 = (1.17302e-9, 1.51361e-2, 7.51886e-2,
                                       4.60643e-3, 1.35000e-2, -1.06750e-4,
                                       1.36600e-5)
        theta = T / (1.0 - T * (C2 + T * (C4 + T * C6)) /
                     (1.0 + T * (C3 + T * (C5 + T * C7))))
        xi = (B_G**2 / (4.0 * theta)) ** (1.0 / 3.0)
        sigmav = C1 * theta * np.sqrt(xi / (m_rc2 * T**3)) * np.exp(-3.0 * xi)
    elif reaction == 'DD':

        sigmav = 3.0e-22 * T**2.0 * np.exp(-31.4 / T**0.5) if T > 0.0 else 0.0
    else:
        sigmav = 0.0

    return max(float(sigmav), 0.0)


def fusion_burn_rate(n_D, n_T, T_keV, reaction='DT'):
    sigmav = bosch_hale_sigmav(T_keV, reaction)
    if reaction == 'DT':
        return sigmav * n_D * n_T
    elif reaction == 'DD':
        return sigmav * n_D**2
    return 0.0


def alpha_deposition_range(rho, T_keV, Z_bar):
    if rho <= 0.0 or T_keV <= 0.0:
        return 1e10


    E_alpha_keV = 3500.0
    n_e = rho / (2.5 * M_U) * Z_bar
    ln_lambda = max(1.0, 23.5 - 0.5 * np.log(n_e * 1e-6) + 1.5 * np.log(T_keV * 1e3))

    range_m = (E_alpha_keV**1.5 * 1e-6) / (n_e * Z_bar * ln_lambda + 1e-30)
    return max(range_m, 1e-10)


def alpha_deposition_fraction(rho, R_zone, T_keV, Z_bar):
    range_alpha = alpha_deposition_range(rho, T_keV, Z_bar)
    if range_alpha <= 0.0:
        return 1.0



    frac = 1.0 - np.exp(-R_zone / (range_alpha + 1e-30))
    return np.clip(frac, 0.0, 1.0)


def species_evolution_markov(n_D, n_T, n_He3, n_p, n_n, dt, rho, T_keV):
    sigmav_DT = bosch_hale_sigmav(T_keV, 'DT')
    sigmav_DD = bosch_hale_sigmav(T_keV, 'DD')


    R_DT = sigmav_DT * n_D * n_T
    R_DD = sigmav_DD * n_D**2


    total_ions = n_D + n_T + n_He3 + n_p + 1e-30


    P = np.eye(5)


    dD_dt = -R_DT - 2.0 * R_DD

    dT_dt = -R_DT + R_DD

    dHe3_dt = R_DD

    dp_dt = R_DD

    dn_dt = R_DT


    if total_ions > 1e10:
        P[0, 0] = max(0.0, 1.0 + dD_dt * dt / (n_D + 1e-30))
        P[1, 1] = max(0.0, 1.0 + dT_dt * dt / (n_T + 1e-30))
        P[2, 2] = max(0.0, 1.0 + dHe3_dt * dt / (n_He3 + 1e-30))
        P[3, 3] = max(0.0, 1.0 + dp_dt * dt / (n_p + 1e-30))
        P[4, 4] = max(0.0, 1.0 + dn_dt * dt / (n_n + 1e-30))

    species = np.array([float(n_D), float(n_T), float(n_He3), float(n_p), float(n_n)])
    new_species = P @ species


    new_species = np.maximum(new_species, 0.0)

    return new_species


def reaction_diffusion_burn_step(Y, T, rho, R_zone, dt, D_diff=1e-6):
    n = len(Y)
    if n <= 1:
        return np.array(Y)

    Y = np.asarray(Y, dtype=float)
    T = np.asarray(T, dtype=float)
    rho = np.asarray(rho, dtype=float)
    R_zone = np.asarray(R_zone, dtype=float)

    dYdt = np.zeros(n)

    for i in range(n):

        rho_i = float(rho[i]) if len(rho) > 1 else float(rho[0])
        n_D = Y[i] * rho_i / (2.5 * M_U)
        n_T = n_D
        T_keV = float(T[i]) / 1e3 if float(T[i]) > 0.0 else 0.01

        R_fusion = fusion_burn_rate(float(n_D), float(n_T), float(T_keV), 'DT')


        n_total = rho_i / (2.5 * M_U)
        reaction_term = R_fusion / (n_total + 1e-30)


        R_i = float(R_zone[i])
        if i == 0:
            if n > 1 and R_i > 1e-15:
                laplace = 2.0 * (Y[1] - Y[0]) / R_i**2
            else:
                laplace = 0.0
        elif i == n - 1:
            if R_i > 1e-15:
                laplace = 2.0 * (Y[n - 2] - Y[n - 1]) / R_i**2
            else:
                laplace = 0.0
        else:
            dr = float(R_zone[i + 1]) - float(R_zone[i - 1])
            if abs(dr) < 1e-30:
                laplace = 0.0
            else:
                dY_dr = (Y[i + 1] - Y[i - 1]) / dr
                dr2 = float(R_zone[i + 1]) - float(R_zone[i])
                d2Y_dr2 = (Y[i + 1] - 2.0 * Y[i] + Y[i - 1]) / (dr2**2 + 1e-30)
                if R_i > 1e-15:
                    laplace = d2Y_dr2 + (2.0 / R_i) * dY_dr
                else:
                    laplace = d2Y_dr2

        dYdt[i] = float(D_diff) * float(laplace) + float(reaction_term)


    if n > 1:
        dr_vals = np.diff(R_zone)
        if len(dr_vals) > 0:
            dr_min = np.min(dr_vals)
            dt_diff = 0.5 * dr_min**2 / max(D_diff, 1e-30)
            if dt > dt_diff:
                dt = dt_diff

    Y_new = Y + dt * dYdt
    Y_new = np.clip(Y_new, 0.0, 1.0)

    return Y_new


def compute_burn_metrics(Y, rho, R_zone):
    n = len(Y)
    if n <= 1:
        return 0.0, 0.0, 0.0


    volumes = np.zeros(n)
    for i in range(n):
        r_inner = float(R_zone[i - 1]) if i > 0 else 0.0
        r_outer = float(R_zone[i])
        volumes[i] = (4.0 / 3.0) * np.pi * (r_outer**3 - r_inner**3)

    total_vol = np.sum(volumes)
    avg_burn = np.sum(Y * volumes) / (total_vol + 1e-30)


    n_total = rho / (2.5 * M_U)
    total_fusions = np.sum(Y * n_total * volumes)
    yield_J = total_fusions * Q_DT


    gain = yield_J / (1.0 + 1e-30)

    return float(avg_burn), float(yield_J), float(gain)
