
import numpy as np
import math
from utils import gauss_legendre_1d, clamp, ensure_positive


def map_interval(xi: float, a: float, b: float) -> float:
    return 0.5 * (a + b) + 0.5 * (b - a) * xi


def integrate_3d_cube_gauss(
    func,
    x_bounds: tuple,
    y_bounds: tuple,
    z_bounds: tuple,
    order_x: int = 3,
    order_y: int = 3,
    order_z: int = 3,
) -> float:
    wx, xn = gauss_legendre_1d(order_x)
    wy, yn = gauss_legendre_1d(order_y)
    wz, zn = gauss_legendre_1d(order_z)

    ax, bx = x_bounds
    ay, by = y_bounds
    az, bz = z_bounds

    jx = 0.5 * (bx - ax)
    jy = 0.5 * (by - ay)
    jz = 0.5 * (bz - az)

    total = 0.0
    for i in range(order_x):
        x = map_interval(xn[i], ax, bx)
        for j in range(order_y):
            y = map_interval(yn[j], ay, by)
            for k in range(order_z):
                z = map_interval(zn[k], az, bz)
                total += wx[i] * wy[j] * wz[k] * func(x, y, z)

    return total * jx * jy * jz


def absorbed_energy_in_coating(
    power_density_func,
    length_x: float,
    length_y: float,
    thickness_z: float,
    order: int = 3,
) -> float:
    length_x = max(float(length_x), 1e-6)
    length_y = max(float(length_y), 1e-6)
    thickness_z = max(float(thickness_z), 1e-6)
    order = int(clamp(order, 1, 5))

    return integrate_3d_cube_gauss(
        power_density_func,
        (0.0, length_x),
        (0.0, length_y),
        (0.0, thickness_z),
        order_x=order,
        order_y=order,
        order_z=order,
    )


def multi_frequency_absorption_average(
    absorption_values: np.ndarray,
    weights: np.ndarray = None,
) -> tuple:
    absorption_values = np.asarray(absorption_values, dtype=float)
    if absorption_values.size == 0:
        return 0.0, 0.0

    if weights is None:
        mean = float(np.mean(absorption_values))
        std = float(np.std(absorption_values, ddof=1)) if absorption_values.size > 1 else 0.0
    else:
        weights = np.asarray(weights, dtype=float)
        if weights.shape != absorption_values.shape:
            raise ValueError("weights must match absorption_values shape.")
        wsum = np.sum(weights)
        if abs(wsum) < 1e-15:
            return 0.0, 0.0
        mean = float(np.sum(weights * absorption_values) / wsum)
        var = float(np.sum(weights * (absorption_values - mean) ** 2) / wsum)
        std = math.sqrt(max(var, 0.0))

    return mean, std


def absorption_efficiency(
    absorbed_power: float,
    incident_power_density: float,
    area: float,
) -> float:
    incident_power_density = max(float(incident_power_density), 1e-15)
    area = max(float(area), 1e-15)
    eta = absorbed_power / (incident_power_density * area)
    return clamp(eta, 0.0, 1.0)


def rcs_reduction_db(
    R_with_coating: float,
    R_bare_metal: float,
) -> float:
    R_with_coating = max(float(R_with_coating), 1e-15)
    R_bare_metal = max(float(R_bare_metal), 1e-15)
    reduction_db = 10.0 * math.log10(R_with_coating / R_bare_metal)
    return reduction_db


def frequency_averaged_rcs_reduction(
    R_coating_array: np.ndarray,
    R_metal_array: np.ndarray,
    frequencies: np.ndarray,
) -> float:
    R_coating_array = np.asarray(R_coating_array, dtype=float)
    R_metal_array = np.asarray(R_metal_array, dtype=float)
    if R_coating_array.size == 0 or R_metal_array.size == 0:
        return 0.0

    mean_Rc = float(np.mean(R_coating_array))
    mean_Rm = float(np.mean(R_metal_array))
    mean_Rc = max(mean_Rc, 1e-15)
    mean_Rm = max(mean_Rm, 1e-15)
    return 10.0 * math.log10(mean_Rc / mean_Rm)
