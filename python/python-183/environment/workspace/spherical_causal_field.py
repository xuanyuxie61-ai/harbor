
import numpy as np
from typing import Tuple


def sphere_llq_grid_points(r: float, pc: np.ndarray,
                            lat_num: int, long_num: int) -> np.ndarray:
    if r <= 0.0:
        raise ValueError("半径 r 必须为正。")
    if lat_num < 0 or long_num < 1:
        raise ValueError("lat_num 必须 >=0, long_num 必须 >=1。")

    n_points = 2 + lat_num * long_num
    points = np.zeros((n_points, 3))
    n = 0


    theta = 0.0
    phi = 0.0
    points[n, 0] = pc[0] + r * np.sin(phi) * np.cos(theta)
    points[n, 1] = pc[1] + r * np.sin(phi) * np.sin(theta)
    points[n, 2] = pc[2] + r * np.cos(phi)
    n += 1


    for lat in range(1, lat_num + 1):
        phi = np.pi * lat / (lat_num + 1)
        for lon in range(long_num):
            theta = 2.0 * np.pi * lon / long_num
            points[n, 0] = pc[0] + r * np.sin(phi) * np.cos(theta)
            points[n, 1] = pc[1] + r * np.sin(phi) * np.sin(theta)
            points[n, 2] = pc[2] + r * np.cos(phi)
            n += 1


    theta = 0.0
    phi = np.pi
    points[n, 0] = pc[0] + r * np.sin(phi) * np.cos(theta)
    points[n, 1] = pc[1] + r * np.sin(phi) * np.sin(theta)
    points[n, 2] = pc[2] + r * np.cos(phi)
    n += 1

    return points


def associated_legendre(l_max: int, x: float) -> np.ndarray:
    if l_max < 0:
        raise ValueError("l_max 必须非负。")
    if not (-1.0 <= x <= 1.0):
        raise ValueError("x 必须在 [-1,1] 内。")

    P = np.zeros((l_max + 1, 2 * l_max + 1))

    P[0, l_max] = 1.0
    if l_max == 0:
        return P


    somx2 = np.sqrt(max(0.0, 1.0 - x * x))
    for l in range(1, l_max + 1):
        P[l, l_max + l] = - (2 * l - 1) * somx2 * P[l - 1, l_max + l - 1]
        if l < l_max:
            P[l, l_max + l + 1] = x * (2 * l + 1) * P[l, l_max + l]


    for l in range(2, l_max + 1):
        for m in range(l - 2, -1, -1):
            idx_m = l_max + m
            P[l, idx_m] = ((2 * l - 1) * x * P[l - 1, idx_m] - (l + m - 1) * P[l - 2, idx_m]) / (l - m)

    return P


def spherical_harmonic_Y(l: int, m: int, theta: float, phi: float) -> complex:
    if abs(m) > l:
        return 0.0 + 0.0j
    x = np.cos(theta)
    P_all = associated_legendre(l, x)
    P_lm = P_all[l, l + m]


    from math import factorial, sqrt
    norm = sqrt((2 * l + 1) / (4 * np.pi) * factorial(l - m) / factorial(l + m))
    return complex(norm * P_lm * np.cos(m * phi), norm * P_lm * np.sin(m * phi))


def spherical_laplacian_spectrum(l_max: int) -> np.ndarray:
    l = np.arange(l_max + 1)
    return -l * (l + 1)


def project_to_spherical_harmonics(field_values: np.ndarray,
                                    points: np.ndarray,
                                    l_max: int) -> np.ndarray:
    n_points = len(field_values)
    n_modes = (l_max + 1) ** 2
    Ymat = np.zeros((n_points, n_modes))

    for idx in range(n_points):
        x, y, z = points[idx]
        r = np.sqrt(x * x + y * y + z * z)
        if r < 1e-12:
            theta = 0.0
            phi = 0.0
        else:
            theta = np.arccos(np.clip(z / r, -1.0, 1.0))
            phi = np.arctan2(y, x)
        col = 0
        for l in range(l_max + 1):
            for m in range(-l, l + 1):
                Yval = spherical_harmonic_Y(l, m, theta, phi)
                Ymat[idx, col] = Yval.real
                col += 1


    coeffs = np.linalg.lstsq(Ymat, field_values, rcond=None)[0]
    return coeffs


def demo():
    points = sphere_llq_grid_points(r=1.0, pc=np.zeros(3), lat_num=4, long_num=8)
    print(f"[spherical_causal_field] 球面网格点数: {len(points)}")


    field = np.zeros(len(points))
    for i, pt in enumerate(points):

        r = np.linalg.norm(pt)
        if r > 1e-12:
            theta = np.arccos(np.clip(pt[2] / r, -1.0, 1.0))
            phi = np.arctan2(pt[1], pt[0])
            field[i] = np.exp(-2.0 * (theta - np.pi / 3.0) ** 2) * np.cos(2 * phi)

    coeffs = project_to_spherical_harmonics(field, points, l_max=3)
    print(f"[spherical_causal_field] 球面调和系数 (前5): {coeffs[:5].round(4)}")
    return points, coeffs


if __name__ == "__main__":
    demo()
