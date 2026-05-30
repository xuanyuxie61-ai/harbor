
import numpy as np
from quadrature_utils import integrate_gauss_jacobi, gauss_legendre_rule


K_B = 1.380649e-23
H_PLANCK = 6.62607015e-34
C_LIGHT = 2.99792458e8
M_E = 9.10938356e-31
E_CHARGE = 1.602176634e-19
EPSILON_0 = 8.854187817e-12


def ideal_gas_pressure(rho, T, Z_bar, A_ion=2.5):
    m_u = 1.66053906660e-27
    a_R = 7.5657e-16


    T_clamped = float(np.clip(T, 1.0, 1e12))

    n_ion = rho / (A_ion * m_u)
    n_e = Z_bar * n_ion

    P_ion = n_ion * K_B * T_clamped
    P_e = n_e * K_B * T_clamped

    log_T = np.log(T_clamped)
    P_rad = a_R * np.exp(min(4.0 * log_T, 700.0)) / 3.0

    return P_ion + P_e + P_rad


def internal_energy(rho, T, Z_bar, A_ion=2.5):
    m_u = 1.66053906660e-27
    a_R = 7.5657e-16
    T_clamped = float(np.clip(T, 1.0, 1e12))
    epsilon_th = 1.5 * K_B * T_clamped * (1.0 + Z_bar) / (A_ion * m_u)
    log_T = np.log(T_clamped)
    epsilon_rad = a_R * np.exp(min(4.0 * log_T, 700.0)) / max(rho, 1e-30)
    return epsilon_th + epsilon_rad


def saha_ionization(rho, T, element='DT', Z_nucleus=1.0):
    T_clamped = float(np.clip(T, 1.0, 1e12))
    if T_clamped <= 100.0:
        return 0.0


    exponent = 1.5 * np.log(max(2.0 * np.pi * M_E * K_B * T_clamped / H_PLANCK**2, 1e-300))
    de_broglie = np.exp(min(exponent, 700.0))
    chi_ion = 13.6 * E_CHARGE


    arg = -chi_ion / (K_B * T_clamped)
    saha_factor = de_broglie * np.exp(max(arg, -700.0))


    m_u = 1.66053906660e-27
    n_total = rho / (2.5 * m_u)



    a_quad = float(n_total)
    b_quad = float(saha_factor)
    c_quad = -b_quad

    discriminant = b_quad**2 - 4.0 * a_quad * c_quad
    if discriminant < 0.0 or a_quad < 1e-30:
        return 0.0

    f = (-b_quad + np.sqrt(discriminant)) / (2.0 * a_quad)
    return float(np.clip(f, 0.0, Z_nucleus))


def bremsstrahlung_opacity(nu, rho, T, Z_bar):
    if T <= 0.0 or rho <= 0.0:
        return 0.0

    m_u = 1.66053906660e-27
    n_i = rho / (2.5 * m_u)
    n_e = Z_bar * n_i


    g_ff = 1.0

    prefactor = (4.0 / 3.0) * np.sqrt(2.0 * np.pi / 3.0)
    e6 = E_CHARGE**6
    denom = (4.0 * np.pi * EPSILON_0)**3 * M_E**2 * C_LIGHT * H_PLANCK * nu**3

    kappa = prefactor * e6 / denom * Z_bar**3 * n_i * n_e * g_ff
    kappa *= (1.0 - np.exp(-H_PLANCK * nu / (K_B * T)))

    return max(kappa, 0.0)


def rosseland_mean_opacity(rho, T, Z_bar, n_quad=32):
    if T <= 0.0 or rho <= 0.0:
        return 1e-20





    a_log = -10.0
    b_log = 5.0
    x_nodes, w_nodes = gauss_legendre_rule(n_quad, a_log, b_log)

    num = 0.0
    den = 0.0

    for xi, wi in zip(x_nodes, w_nodes):
        x = np.exp(xi)
        dx = x
        nu = x * K_B * T / H_PLANCK

        kappa_nu = bremsstrahlung_opacity(nu, rho, T, Z_bar)
        if kappa_nu < 1e-50:
            kappa_nu = 1e-50

        dbdt_weight = x**4 * np.exp(x) / (np.expm1(x))**2
        if not np.isfinite(dbdt_weight):
            dbdt_weight = 0.0

        num += wi * dx * dbdt_weight / kappa_nu
        den += wi * dx * dbdt_weight

    if den <= 0.0 or num <= 0.0:
        return 1e-20

    kappa_R = den / num
    return max(kappa_R, 1e-20)


def energy_flow_digraph(num_nodes, temperatures, conductivities):
    arcs = []
    for i in range(num_nodes):
        if i > 0:
            weight = 0.5 * (conductivities[i] + conductivities[i - 1])
            weight = max(weight, 1e-30)
            arcs.append((i, i - 1, weight))
        if i < num_nodes - 1:
            weight = 0.5 * (conductivities[i] + conductivities[i + 1])
            weight = max(weight, 1e-30)
            arcs.append((i, i + 1, weight))


    adj_row = [0]
    adj = []
    for i in range(num_nodes):
        neighbors = []
        for arc in arcs:
            if arc[0] == i:
                neighbors.append(arc[1])
        neighbors = sorted(list(set(neighbors)))
        adj.extend(neighbors)
        adj_row.append(len(adj))

    adj = np.array(adj, dtype=int)
    adj_row = np.array(adj_row, dtype=int)

    return arcs, adj_row, adj


def electron_thermal_conductivity(rho, T, Z_bar):
    T_clamped = float(np.clip(T, 1.0, 1e12))
    if T_clamped <= 0.0 or Z_bar <= 0.0:
        return 1e-30

    n_e = rho / (2.5 * 1.66053906660e-27) * Z_bar
    if n_e <= 0.0:
        return 1e-30


    ln_lambda = max(1.0, 23.5 - 0.5 * np.log(max(n_e * 1e-6, 1e-300)) + 1.5 * np.log(T_clamped))

    log_kappa = np.log(1.84e-5) + 2.5 * np.log(T_clamped) - np.log(Z_bar) - np.log(ln_lambda)
    kappa_spitzer = np.exp(min(log_kappa, 700.0))
    return max(float(kappa_spitzer), 1e-30)
