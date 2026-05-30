
import numpy as np
from typing import Tuple


def _triangle_rule_dunavant_7():
    sqrt15 = np.sqrt(15.0)
    a1 = (6.0 + sqrt15) / 21.0
    b1 = (9.0 - 2.0 * sqrt15) / 21.0
    a2 = (6.0 - sqrt15) / 21.0
    b2 = (9.0 + 2.0 * sqrt15) / 21.0

    w1 = 9.0 / 80.0
    w2 = (155.0 + sqrt15) / 2400.0
    w3 = (155.0 - sqrt15) / 2400.0


    bary = np.array([
        [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
        [a1, a1, b1],
        [a1, b1, a1],
        [b1, a1, a1],
        [a2, a2, b2],
        [a2, b2, a2],
        [b2, a2, a2],
    ])
    weights = np.array([w1, w2, w2, w2, w3, w3, w3])

    x_tri = bary[:, 0]
    y_tri = bary[:, 1]
    return x_tri, y_tri, weights


def _gauss_legendre_1d(n: int):
    x, w = np.polynomial.legendre.leggauss(n)
    return 0.5 * (x + 1.0), 0.5 * w


def prism_unit_monomial_integral(expon: Tuple[int, int, int]) -> float:
    alpha, beta, gamma = expon
    if alpha < 0 or beta < 0 or gamma < 0:
        raise ValueError("Exponents must be non-negative")

    import math
    num = math.factorial(alpha) * math.factorial(beta)
    den = (gamma + 1) * math.factorial(alpha + beta + 2)
    return num / den


def prism_witherden_rule(p: int = 5) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if p < 0 or p > 10:
        raise ValueError("Precision p must be in [0, 10]")

    if p <= 5:
        x_tri, y_tri, w_tri = _triangle_rule_dunavant_7()
        n_tri = x_tri.shape[0]

        z_1d, w_z = _gauss_legendre_1d(3)
        n_z = z_1d.shape[0]


        x = np.zeros(n_tri * n_z)
        y = np.zeros(n_tri * n_z)
        z = np.zeros(n_tri * n_z)
        w = np.zeros(n_tri * n_z)

        idx = 0
        for i in range(n_tri):
            for j in range(n_z):
                x[idx] = x_tri[i]
                y[idx] = y_tri[i]
                z[idx] = z_1d[j]
                w[idx] = w_tri[i] * w_z[j]
                idx += 1

        return x, y, z, w


    n_sub = 2
    x_base, y_base, z_base, w_base = prism_witherden_rule(p=5)
    x_all, y_all, z_all, w_all = [], [], [], []

    for i in range(n_sub):
        for j in range(n_sub):
            for k in range(n_sub):
                sx, sy, sz = i / n_sub, j / n_sub, k / n_sub
                ds = 1.0 / n_sub

                x_all.extend(sx + ds * x_base)
                y_all.extend(sy + ds * y_base)
                z_all.extend(sz + ds * z_base)
                w_all.extend((ds ** 3) * w_base)

    return np.array(x_all), np.array(y_all), np.array(z_all), np.array(w_all)


def map_prism_to_physical(x_ref: np.ndarray, y_ref: np.ndarray, z_ref: np.ndarray,
                          vertices: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    if vertices.shape != (6, 3):
        raise ValueError("vertices must have shape (6, 3)")

    v0, v1, v2 = vertices[0], vertices[1], vertices[2]
    v3, v4, v5 = vertices[3], vertices[4], vertices[5]


    z_bottom = (v0[2] + v1[2] + v2[2]) / 3.0
    z_top = (v3[2] + v4[2] + v5[2]) / 3.0
    height = z_top - z_bottom


    x_phys = v0[0] + (v1[0] - v0[0]) * x_ref + (v2[0] - v0[0]) * y_ref
    y_phys = v0[1] + (v1[1] - v0[1]) * x_ref + (v2[1] - v0[1]) * y_ref
    z_phys = z_bottom + height * z_ref



    area_base = 0.5 * abs(
        (v1[0] - v0[0]) * (v2[1] - v0[1]) - (v2[0] - v0[0]) * (v1[1] - v0[1])
    )


    detJ_xy = 2.0 * area_base
    detJ = detJ_xy * height

    return x_phys, y_phys, z_phys, detJ


def integrate_ohc_over_prism(temperature_func, vertices: np.ndarray,
                             rho0: float = 1025.0, cp: float = 3993.0) -> float:
    x_ref, y_ref, z_ref, w = prism_witherden_rule(p=5)
    x_phys, y_phys, z_phys, detJ = map_prism_to_physical(x_ref, y_ref, z_ref, vertices)




    integral = 0.0
    for i in range(x_ref.shape[0]):
        t_val = temperature_func(x_phys[i], y_phys[i], z_phys[i])
        if not np.isfinite(t_val):
            t_val = 0.0
        integral += w[i] * t_val

    ohc = rho0 * cp * integral * abs(detJ)
    return float(ohc)


def thermocline_depth_from_profile(z: np.ndarray, t: np.ndarray,
                                   t_crit: float = 20.0) -> float:
    if z.shape != t.shape or z.ndim != 1:
        raise ValueError("z and t must be 1D arrays of same shape")
    if z.shape[0] < 2:
        return float(np.nan)




    z_abs = np.abs(z)


    sort_idx = np.argsort(z_abs)
    z_sorted = z_abs[sort_idx]
    t_sorted = t[sort_idx]


    for i in range(z_sorted.shape[0] - 1):
        if (t_sorted[i] - t_crit) * (t_sorted[i + 1] - t_crit) <= 0.0:
            if abs(t_sorted[i + 1] - t_sorted[i]) < 1e-10:
                return float(z_sorted[i])
            frac = (t_crit - t_sorted[i]) / (t_sorted[i + 1] - t_sorted[i])
            return float(z_sorted[i] + frac * (z_sorted[i + 1] - z_sorted[i]))


    if np.all(t_sorted > t_crit):
        return float(z_sorted[-1])
    return float(z_sorted[0])


def warm_water_volume(thermocline_depth: np.ndarray, lon: np.ndarray, lat: np.ndarray,
                      clim_depth: np.ndarray, dx: float, dy: float) -> float:
    if thermocline_depth.shape != clim_depth.shape:
        raise ValueError("Shape mismatch")

    anomaly = thermocline_depth - clim_depth
    anomaly = np.where(anomaly > 0, anomaly, 0.0)

    cos_lat = np.cos(np.radians(lat))[None, :]
    area_element = dx * dy * cos_lat

    wwv = np.sum(anomaly * area_element)
    return float(wwv)
