import numpy as np


R_UNIVERSAL = 8.314462618
ATM_PA = 101325.0


DEFAULT_GAMMA = 1.4
DEFAULT_Q = 2.5e6
DEFAULT_E_A = 8.314e4
DEFAULT_A_PRE = 1.0e8
DEFAULT_T_IGN = 1500.0
DEFAULT_RHO_0 = 1.225
DEFAULT_P_0 = 101325.0
DEFAULT_T_0 = 300.0
DEFAULT_W_MOL = 0.029


def check_positive(value, name):
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be positive finite, got {value}")
    return value


def check_nonnegative(value, name):
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be nonnegative finite, got {value}")
    return value


def check_interval(a, b, name_a="a", name_b="b"):
    if not (np.isfinite(a) and np.isfinite(b)):
        raise ValueError(f"Interval endpoints must be finite: {name_a}={a}, {name_b}={b}")
    if b <= a:
        raise ValueError(f"Require {name_a} < {name_b}, got {a} >= {b}")
    return a, b


def arrhenius_rate(T, A, Ea, R=R_UNIVERSAL):


    raise NotImplementedError("Hole_1: 请实现 Arrhenius 反应速率公式")


def specific_heat_ratio_cv_cp(gamma):
    check_positive(gamma, "gamma")
    if gamma <= 1.0:
        raise ValueError("gamma must be > 1 for ideal gas")
    cv = R_UNIVERSAL / (gamma - 1.0)
    cp = gamma * cv
    return cv, cp


def sound_speed(T, gamma, W_mol=DEFAULT_W_MOL):
    check_positive(T, "Temperature T")
    check_positive(gamma, "gamma")
    check_positive(W_mol, "Molar mass W_mol")
    return np.sqrt(gamma * R_UNIVERSAL * T / W_mol)


def znd_progress_variable_derivative(lambda_var, T, A, Ea, n_order=1.0, R=R_UNIVERSAL):
    check_nonnegative(lambda_var, "lambda")
    if lambda_var > 1.0:
        lambda_var = 1.0
    check_positive(T, "Temperature T")
    k = arrhenius_rate(T, A, Ea, R)
    remain = max(1.0 - lambda_var, 0.0)
    if remain <= 0.0:
        return 0.0
    return -k * (remain ** n_order)


def rankine_hugoniot_pressure_ratio(M, gamma):
    check_positive(M, "Mach number M")
    check_positive(gamma, "gamma")
    return 1.0 + 2.0 * gamma / (gamma + 1.0) * (M * M - 1.0)


def rankine_hugoniot_density_ratio(M, gamma):
    check_positive(M, "Mach number M")
    check_positive(gamma, "gamma")
    return (gamma + 1.0) * M * M / ((gamma - 1.0) * M * M + 2.0)


def cj_detonation_velocity(gamma, Q, p0, rho0):
    check_positive(gamma, "gamma")
    check_positive(Q, "Heat release Q")
    check_positive(p0, "Pressure p0")
    check_positive(rho0, "Density rho0")
    a0_sq = gamma * p0 / rho0
    D_cj_sq = 2.0 * (gamma * gamma - 1.0) * Q + a0_sq
    if D_cj_sq <= 0.0:
        raise ValueError("CJ detonation velocity squared is non-positive")
    return np.sqrt(D_cj_sq)


def von_neumann_spike_conditions(D, gamma, p0, rho0):
    check_positive(D, "Detonation velocity D")
    a0 = sound_speed_from_prho(p0, rho0, gamma)
    M = D / a0
    p_ratio = rankine_hugoniot_pressure_ratio(M, gamma)
    rho_ratio = rankine_hugoniot_density_ratio(M, gamma)
    p_vn = p0 * p_ratio
    rho_vn = rho0 * rho_ratio
    T_vn = p_vn / (rho_vn * (R_UNIVERSAL / DEFAULT_W_MOL))
    return p_vn, rho_vn, T_vn, M


def sound_speed_from_prho(p, rho, gamma):
    check_positive(p, "Pressure p")
    check_positive(rho, "Density rho")
    check_positive(gamma, "gamma")
    return np.sqrt(gamma * p / rho)


def temperature_from_energy(e, lambda_var, Q, cv):
    check_positive(cv, "cv")
    T = (e - (1.0 - lambda_var) * Q) / cv
    if T < 0.0:

        T = max(T, 1.0e-6)
    return T


def cholesky_factor(a):
    a = np.asarray(a, dtype=float)
    if a.shape != (2, 2):
        raise ValueError("Only 2x2 matrix supported")
    if not np.allclose(a, a.T):
        raise ValueError("Matrix must be symmetric")
    if a[0, 0] <= 0.0:
        raise ValueError("Matrix not positive definite")
    L = np.zeros((2, 2))
    L[0, 0] = np.sqrt(a[0, 0])
    L[1, 0] = a[1, 0] / L[0, 0]
    diag2 = a[1, 1] - L[1, 0] ** 2
    if diag2 <= 0.0:
        raise ValueError("Matrix not positive definite")
    L[1, 1] = np.sqrt(diag2)
    return L


def solve_lower_triangular(L, b):
    x = np.zeros_like(b, dtype=float)
    x[0] = b[0] / L[0, 0]
    x[1] = (b[1] - L[1, 0] * x[0]) / L[1, 1]
    return x
