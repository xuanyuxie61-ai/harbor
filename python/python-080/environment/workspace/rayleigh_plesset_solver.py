
import numpy as np
from scipy.integrate import solve_ivp
from utils import safe_divide, WATER_DENSITY, WATER_VISCOSITY, SURFACE_TENSION, SOUND_SPEED_WATER, VAPOR_PRESSURE, ATMOSPHERIC_PRESSURE




GAMMA_ADIABATIC = 1.4
THERMAL_DIFFUSIVITY = 1.43e-7
VAN_DER_WAALS_A = 0.00365
VAN_DER_WAALS_B = 4.27e-5
GAS_CONSTANT_MOLAR = 8.314
AMBIENT_TEMPERATURE = 293.15


def van_der_waals_pressure(n_g, V, T_g):
    pass


def gas_pressure_adiabatic(R, R0, p_g0):
    ratio = safe_divide(R0, R, default=1e6)
    return p_g0 * ratio ** (3.0 * GAMMA_ADIABATIC)


def plesset_zwick_heat_flux(R, dRdt, T_g):
    delta_T = T_g - AMBIENT_TEMPERATURE
    factor = -3.0 * (GAMMA_ADIABATIC - 1.0) / (R + 1e-15)
    factor *= dRdt * delta_T
    factor *= np.sqrt(THERMAL_DIFFUSIVITY / (np.pi * (R**2 + 1e-30)))
    return factor


def rayleigh_plesset_rhs(t, y, p_inf, sigma, rho, mu, R0, p_g0):
    R, dRdt, T_g, n_g = y
    if R <= 0:
        R = 1e-9

    V = (4.0 / 3.0) * np.pi * R**3
    p_g = van_der_waals_pressure(n_g, V, T_g)


    term1 = safe_divide(p_g - p_inf, rho)
    term2 = -4.0 * mu * dRdt / (rho * R)
    term3 = -2.0 * sigma / (rho * R)
    d2Rdt2 = safe_divide(term1 + term2 + term3 - 1.5 * dRdt**2, R)


    dTgdt = plesset_zwick_heat_flux(R, dRdt, T_g)


    D_gas = 2.0e-9
    dn_gdt = -4.0 * np.pi * R**2 * D_gas * (p_g - p_g0) / (GAS_CONSTANT_MOLAR * T_g * R0 + 1e-30)

    return [dRdt, d2Rdt2, dTgdt, dn_gdt]


def keller_miksis_rhs(t, y, p_inf, sigma, rho, mu, c, R0, p_g0):
    R, dRdt, T_g, n_g = y
    if R <= 0:
        R = 1e-9

    V = (4.0 / 3.0) * np.pi * R**3
    p_g = van_der_waals_pressure(n_g, V, T_g)


    dp_g_dt = 0.0

    lhs_coeff = 1.0 - dRdt / (c + 1e-15)
    rhs_term1 = (1.0 + dRdt / (c + 1e-15)) * (p_g - p_inf) / rho
    rhs_term2 = (R / (rho * c + 1e-30)) * dp_g_dt
    rhs_total = rhs_term1 + rhs_term2
    nonlinear_term = (1.5 - dRdt / (2.0 * c + 1e-15)) * dRdt**2

    d2Rdt2 = safe_divide(rhs_total - nonlinear_term, lhs_coeff * R)

    dTgdt = plesset_zwick_heat_flux(R, dRdt, T_g)
    D_gas = 2.0e-9
    dn_gdt = -4.0 * np.pi * R**2 * D_gas * (p_g - p_g0) / (GAS_CONSTANT_MOLAR * T_g * R0 + 1e-30)

    return [dRdt, d2Rdt2, dTgdt, dn_gdt]


def solve_rayleigh_plesset(R0, p_g0, p_inf, t_span, method='RK45', use_keller_miksis=True):
    y0 = [R0, 0.0, AMBIENT_TEMPERATURE, p_g0 * (4.0/3.0)*np.pi*R0**3 / (GAS_CONSTANT_MOLAR * AMBIENT_TEMPERATURE)]
    rho = WATER_DENSITY
    mu = WATER_VISCOSITY
    sigma = SURFACE_TENSION
    c = SOUND_SPEED_WATER

    if use_keller_miksis:
        rhs = lambda t, y: keller_miksis_rhs(t, y, p_inf, sigma, rho, mu, c, R0, p_g0)
    else:
        rhs = lambda t, y: rayleigh_plesset_rhs(t, y, p_inf, sigma, rho, mu, R0, p_g0)

    sol = solve_ivp(rhs, t_span, y0, method=method, dense_output=True, max_step=t_span[1]/1000.0)
    return sol


def critical_nucleation_radius_bisection(p_inf, p_v, sigma, rho, R_min=1e-9, R_max=1e-3, tol=1e-12):
    R0 = 1e-6

    def f(R):
        if R <= 0:
            return -1e20
        p_g = p_v * (R0 / R) ** (3.0 * GAMMA_ADIABATIC)
        return p_g - p_inf + 2.0 * sigma / R

    fa = f(R_min)
    fb = f(R_max)


    if fa * fb > 0:

        R_crit_analytical = np.sqrt(2.0 * sigma / (3.0 * max(p_inf - p_v, 1.0)))
        return R_crit_analytical


    while abs(R_max - R_min) > tol:
        R_mid = (R_min + R_max) / 2.0
        fc = f(R_mid)
        if fc == 0:
            return R_mid
        if fa * fc < 0:
            R_max = R_mid
            fb = fc
        else:
            R_min = R_mid
            fa = fc

    return (R_min + R_max) / 2.0


def nonlinear_bubble_residue(y, p_inf, sigma, rho, mu, R0, p_g0):
    R, dRdt, T_g, n_g = y
    V = (4.0 / 3.0) * np.pi * R**3
    p_g = van_der_waals_pressure(n_g, V, T_g)

    F1 = dRdt
    F2 = safe_divide(p_g - p_inf, rho) - 2.0 * sigma / (rho * R)
    F3 = T_g - AMBIENT_TEMPERATURE
    F4 = p_g - p_g0

    return np.array([F1, F2, F3, F4], dtype=float)


def nonlinear_bubble_jacobian(y, p_inf, sigma, rho, mu, R0, p_g0):
    n = len(y)
    J = np.zeros((n, n), dtype=float)
    h = 1e-8
    F0 = nonlinear_bubble_residue(y, p_inf, sigma, rho, mu, R0, p_g0)
    for j in range(n):
        yp = y.copy()
        yp[j] += h
        Fp = nonlinear_bubble_residue(yp, p_inf, sigma, rho, mu, R0, p_g0)
        J[:, j] = (Fp - F0) / h
    return J


def solve_steady_state_newton(p_inf, sigma, rho, mu, R0, p_g0, max_iter=50, tol=1e-12):
    y = np.array([R0, 0.0, AMBIENT_TEMPERATURE, p_g0 * (4.0/3.0)*np.pi*R0**3 / (GAS_CONSTANT_MOLAR * AMBIENT_TEMPERATURE)], dtype=float)

    for k in range(max_iter):
        F = nonlinear_bubble_residue(y, p_inf, sigma, rho, mu, R0, p_g0)
        if np.linalg.norm(F) < tol:
            break
        J = nonlinear_bubble_jacobian(y, p_inf, sigma, rho, mu, R0, p_g0)
        try:
            dy = np.linalg.solve(J, -F)
        except np.linalg.LinAlgError:

            dy = -np.linalg.lstsq(J, F, rcond=None)[0]
        y = y + dy

        y[0] = max(y[0], 1e-9)
        y[2] = max(y[2], 1.0)
        y[3] = max(y[3], 1e-20)
    return y
