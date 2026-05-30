
import numpy as np


def chebyshev1_nodes(n, a=-1.0, b=1.0):
    i = np.arange(n, dtype=np.float64)
    x = np.cos((2.0 * i + 1.0) * np.pi / (2.0 * n))
    x_mapped = 0.5 * (b - a) * x + 0.5 * (b + a)
    return x_mapped


def barycentric_weights_cheby1(n):
    j = np.arange(n, dtype=np.float64)
    w = ((-1.0) ** j) * np.sin((2.0 * j + 1.0) * np.pi / (2.0 * n))
    return w


def barycentric_interp_1d(xd, yd, xi):
    xd = np.asarray(xd, dtype=np.float64).ravel()
    yd = np.asarray(yd, dtype=np.float64).ravel()
    xi = np.asarray(xi, dtype=np.float64).ravel()
    n = len(xd)


    wd = barycentric_weights_cheby1(n)

    numer = np.zeros(len(xi), dtype=np.float64)
    denom = np.zeros(len(xi), dtype=np.float64)
    exact = np.zeros(len(xi), dtype=np.int64) - 1

    for j in range(n):
        diff = xi - xd[j]

        mask = np.abs(diff) < 1e-14
        exact[mask] = j

        diff_safe = np.where(np.abs(diff) < 1e-14, 1.0, diff)
        t = wd[j] / diff_safe
        numer += t * yd[j]
        denom += t

    yi = numer / denom

    valid_exact = exact >= 0
    yi[valid_exact] = yd[exact[valid_exact]]
    return yi


def thermal_conductivity_interpolator(T_points, kappa_points):
    xd = np.asarray(T_points, dtype=np.float64)
    yd = np.asarray(kappa_points, dtype=np.float64)

    def interpolator(T):
        T_arr = np.asarray(T, dtype=np.float64)
        if T_arr.ndim == 0:
            T_arr = np.array([T_arr])
        result = barycentric_interp_1d(xd, yd, T_arr)

        return np.maximum(result, 0.01)

    return interpolator


def specific_heat_interpolator(T_points, Cp_points):
    xd = np.asarray(T_points, dtype=np.float64)
    yd = np.asarray(Cp_points, dtype=np.float64)

    def interpolator(T):
        T_arr = np.asarray(T, dtype=np.float64)
        if T_arr.ndim == 0:
            T_arr = np.array([T_arr])
        result = barycentric_interp_1d(xd, yd, T_arr)
        return np.maximum(result, 100.0)

    return interpolator


def density_interpolator(T_points, rho_points):
    xd = np.asarray(T_points, dtype=np.float64)
    yd = np.asarray(rho_points, dtype=np.float64)

    def interpolator(T):
        T_arr = np.asarray(T, dtype=np.float64)
        if T_arr.ndim == 0:
            T_arr = np.array([T_arr])
        result = barycentric_interp_1d(xd, yd, T_arr)
        return np.maximum(result, 10.0)

    return interpolator


def default_biomass_properties():
    T_data = np.array([300.0, 400.0, 500.0, 600.0, 700.0, 800.0, 900.0], dtype=np.float64)
    kappa_data = np.array([0.12, 0.14, 0.16, 0.18, 0.20, 0.15, 0.10], dtype=np.float64)
    Cp_data = np.array([1200.0, 1350.0, 1500.0, 1700.0, 1850.0, 1600.0, 1400.0], dtype=np.float64)
    rho_data = np.array([550.0, 480.0, 420.0, 350.0, 280.0, 220.0, 180.0], dtype=np.float64)

    kappa_interp = thermal_conductivity_interpolator(T_data, kappa_data)
    Cp_interp = specific_heat_interpolator(T_data, Cp_data)
    rho_interp = density_interpolator(T_data, rho_data)

    return {
        'T_data': T_data,
        'kappa_interp': kappa_interp,
        'Cp_interp': Cp_interp,
        'rho_interp': rho_interp
    }
