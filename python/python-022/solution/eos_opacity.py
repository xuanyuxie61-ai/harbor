"""
Equation of State and Opacity for ICF Plasma

Based on:
- test_digraph_arc (Project 1205): Directed graph for energy flow network
- jacobi_rule (Project 608): Gauss-Jacobi quadrature for Rosseland mean integration

Models:
- Ideal gas + radiation pressure EOS
- Saha ionization equilibrium
- Rosseland mean opacity via frequency integration
- Energy flow network coupling radiative, conductive, and convective channels
"""

import numpy as np
from quadrature_utils import integrate_gauss_jacobi, gauss_legendre_rule

# Physical constants
K_B = 1.380649e-23       # Boltzmann constant [J/K]
H_PLANCK = 6.62607015e-34  # Planck constant [J*s]
C_LIGHT = 2.99792458e8   # Speed of light [m/s]
M_E = 9.10938356e-31     # Electron mass [kg]
E_CHARGE = 1.602176634e-19  # Elementary charge [C]
EPSILON_0 = 8.854187817e-12  # Vacuum permittivity [F/m]


def ideal_gas_pressure(rho, T, Z_bar, A_ion=2.5):
    """
    Total pressure: P = P_ion + P_e + P_rad
    P_ion = rho * k_B * T / (A_ion * m_u)
    P_e = Z_bar * rho * k_B * T / (A_ion * m_u)
    P_rad = a_R * T^4 / 3
    """
    m_u = 1.66053906660e-27  # atomic mass unit [kg]
    a_R = 7.5657e-16  # radiation constant [J/m^3/K^4]

    # Clamp temperature to prevent overflow
    T_clamped = float(np.clip(T, 1.0, 1e12))

    n_ion = rho / (A_ion * m_u)
    n_e = Z_bar * n_ion

    P_ion = n_ion * K_B * T_clamped
    P_e = n_e * K_B * T_clamped
    # Use log-space for T^4 to prevent overflow
    log_T = np.log(T_clamped)
    P_rad = a_R * np.exp(min(4.0 * log_T, 700.0)) / 3.0

    return P_ion + P_e + P_rad


def internal_energy(rho, T, Z_bar, A_ion=2.5):
    """
    Specific internal energy:
    epsilon = (3/2) * k_B * T * (1 + Z_bar) / (A_ion * m_u) + a_R * T^4 / rho
    """
    m_u = 1.66053906660e-27
    a_R = 7.5657e-16
    T_clamped = float(np.clip(T, 1.0, 1e12))
    epsilon_th = 1.5 * K_B * T_clamped * (1.0 + Z_bar) / (A_ion * m_u)
    log_T = np.log(T_clamped)
    epsilon_rad = a_R * np.exp(min(4.0 * log_T, 700.0)) / max(rho, 1e-30)
    return epsilon_th + epsilon_rad


def saha_ionization(rho, T, element='DT', Z_nucleus=1.0):
    """
    Saha equation for ionization fraction in LTE:
    n_e * n_{Z+1} / n_Z = (2*pi*m_e*k_B*T/h^2)^{3/2} * (2*g_{Z+1}/g_Z) * exp(-chi/k_B*T)
    Simplified for DT plasma.
    """
    T_clamped = float(np.clip(T, 1.0, 1e12))
    if T_clamped <= 100.0:
        return 0.0

    # Prevent overflow in de_broglie
    exponent = 1.5 * np.log(max(2.0 * np.pi * M_E * K_B * T_clamped / H_PLANCK**2, 1e-300))
    de_broglie = np.exp(min(exponent, 700.0))
    chi_ion = 13.6 * E_CHARGE  # ionization energy of hydrogen-like [J]

    # Simplified: assume g factors = 1
    arg = -chi_ion / (K_B * T_clamped)
    saha_factor = de_broglie * np.exp(max(arg, -700.0))

    # Approximate electron density
    m_u = 1.66053906660e-27
    n_total = rho / (2.5 * m_u)

    # Solve quadratic for ionization fraction f: f^2 * n_total = saha_factor * (1-f)
    # f^2 * n_total + f * saha_factor - saha_factor = 0
    a_quad = float(n_total)
    b_quad = float(saha_factor)
    c_quad = -b_quad

    discriminant = b_quad**2 - 4.0 * a_quad * c_quad
    if discriminant < 0.0 or a_quad < 1e-30:
        return 0.0

    f = (-b_quad + np.sqrt(discriminant)) / (2.0 * a_quad)
    return float(np.clip(f, 0.0, Z_nucleus))


def bremsstrahlung_opacity(nu, rho, T, Z_bar):
    """
    Free-free (bremsstrahlung) absorption coefficient [m^-1] at frequency nu.
    kappa_ff = (4/3) * sqrt(2*pi/3) * (e^6 / (4*pi*eps0)^3) * (1/(m_e^2*c*h*nu^3))
               * Z^3 * n_i * n_e * g_ff * (1 - exp(-h*nu/k_B*T))
    """
    if T <= 0.0 or rho <= 0.0:
        return 0.0

    m_u = 1.66053906660e-27
    n_i = rho / (2.5 * m_u)
    n_e = Z_bar * n_i

    # Gaunt factor approximation
    g_ff = 1.0

    prefactor = (4.0 / 3.0) * np.sqrt(2.0 * np.pi / 3.0)
    e6 = E_CHARGE**6
    denom = (4.0 * np.pi * EPSILON_0)**3 * M_E**2 * C_LIGHT * H_PLANCK * nu**3

    kappa = prefactor * e6 / denom * Z_bar**3 * n_i * n_e * g_ff
    kappa *= (1.0 - np.exp(-H_PLANCK * nu / (K_B * T)))

    return max(kappa, 0.0)


def rosseland_mean_opacity(rho, T, Z_bar, n_quad=32):
    """
    Rosseland mean opacity:
    1/kappa_R = [int_0^inf (1/kappa_nu) * dB_nu/dT dnu] / [int_0^inf dB_nu/dT dnu]
    where B_nu is Planck function.

    We use Gauss-Jacobi quadrature after change of variables.
    """
    if T <= 0.0 or rho <= 0.0:
        return 1e-20

    # Dimensionless frequency: x = h*nu / (k_B*T)
    # dB/dT ~ x^4 * e^x / (e^x - 1)^2
    # integrand weight ~ x^4 * e^x / (e^x - 1)^2

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
    """
    Build directed graph for energy flow between plasma zones.
    Based on digraph_arc from Project 1205.

    Nodes represent Lagrangian zones; edges represent energy flux
    with weights proportional to conductive/radiative coupling.
    """
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

    # Build adjacency for RCM reordering
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
    """
    Spitzer electron thermal conductivity:
    kappa_e = (1.84e-5 * T^{5/2}) / (Z_bar * ln_Lambda)  [W/m/K]
    where ln_Lambda is Coulomb logarithm.
    """
    T_clamped = float(np.clip(T, 1.0, 1e12))
    if T_clamped <= 0.0 or Z_bar <= 0.0:
        return 1e-30

    n_e = rho / (2.5 * 1.66053906660e-27) * Z_bar
    if n_e <= 0.0:
        return 1e-30

    # Coulomb logarithm
    ln_lambda = max(1.0, 23.5 - 0.5 * np.log(max(n_e * 1e-6, 1e-300)) + 1.5 * np.log(T_clamped))

    log_kappa = np.log(1.84e-5) + 2.5 * np.log(T_clamped) - np.log(Z_bar) - np.log(ln_lambda)
    kappa_spitzer = np.exp(min(log_kappa, 700.0))
    return max(float(kappa_spitzer), 1e-30)
