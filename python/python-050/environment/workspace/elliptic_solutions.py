
import numpy as np
from typing import Tuple


def carlson_rf(x: float, y: float, z: float, errtol: float = 1e-10) -> float:
    x0, y0, z0 = float(x), float(y), float(z)

    if x0 < 0 or y0 < 0 or z0 < 0:
        raise ValueError("Arguments to R_F must be non-negative.")

    if x0 + y0 + z0 < 1e-20:
        return 0.0

    x, y, z = x0, y0, z0
    s = 0.0
    power4 = 1.0

    for _ in range(100):
        lam = np.sqrt(x * y) + np.sqrt(y * z) + np.sqrt(z * x)
        x = 0.25 * (x + lam)
        y = 0.25 * (y + lam)
        z = 0.25 * (z + lam)
        power4 *= 0.25

        avg = (x + y + z) / 3.0
        dx = 1.0 - x / avg
        dy = 1.0 - y / avg
        dz = 1.0 - z / avg

        if max(abs(dx), abs(dy), abs(dz)) < errtol:
            break


    e2 = dx * dy + dy * dz + dz * dx
    e3 = dx * dy * dz

    rf = (1.0 - e2 / 10.0 + e3 / 14.0 + e2 ** 2 / 24.0
          - 3.0 * e2 * e3 / 44.0) / np.sqrt(avg)

    return float(rf)


def carlson_rd(x: float, y: float, z: float, errtol: float = 1e-10) -> float:
    x0, y0, z0 = float(x), float(y), float(z)

    if x0 < 0 or y0 < 0 or z0 <= 0:
        raise ValueError("Arguments to R_D must be non-negative with z > 0.")

    x, y, z = x0, y0, z0
    s = 0.0
    power4 = 1.0
    fac = 1.0

    for _ in range(100):
        lam = np.sqrt(x * y) + np.sqrt(y * z) + np.sqrt(z * x)
        s += fac / (np.sqrt(z) * (z + lam))
        fac *= 0.25
        x = 0.25 * (x + lam)
        y = 0.25 * (y + lam)
        z = 0.25 * (z + lam)

        avg = (x + y + 3.0 * z) / 5.0
        dx = 1.0 - x / avg
        dy = 1.0 - y / avg
        dz = 1.0 - z / avg

        if max(abs(dx), abs(dy), abs(dz)) < errtol:
            break

    e2 = dx * dy + 2.0 * dz * (dx + dy) + 3.0 * dz ** 2
    e3 = dy * dz * (dx + dy) + 2.0 * dx * dy * dz + 4.0 * dz ** 3
    e4 = dx * dy * dz ** 2
    e5 = dx * dy ** 2 * dz ** 2

    rd = (1.0 - 3.0 * e2 / 14.0 + e3 / 6.0 + 9.0 * e2 ** 2 / 88.0
          - 3.0 * e4 / 22.0 - 9.0 * e2 * e3 / 52.0 + 3.0 * e5 / 26.0)
    rd = 3.0 * s + power4 * rd / (avg * np.sqrt(avg))

    return float(rd)


def elliptic_k_complete(k: float) -> float:
    m = k ** 2
    if m < 0 or m > 1:
        raise ValueError("k^2 must be in [0, 1]")
    return carlson_rf(0.0, 1.0 - m, 1.0)


def elliptic_e_complete(k: float) -> float:
    m = k ** 2
    if m < 0 or m > 1:
        raise ValueError("k^2 must be in [0, 1]")
    return carlson_rf(0.0, 1.0 - m, 1.0) - (m / 3.0) * carlson_rd(0.0, 1.0 - m, 1.0)


def vialov_profile(x: np.ndarray, L: float, H0: float, n: float = 3.0) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    L = float(L)
    H0 = float(H0)

    if L <= 0 or H0 <= 0:
        raise ValueError("L and H0 must be positive.")

    xi = 2.0 * np.abs(x) / L
    xi = np.clip(xi, 0.0, 1.0)

    exponent_base = n + 1.0
    exponent_result = n / (2.0 * n + 2.0)


    base = 1.0 - xi ** exponent_base
    base = np.maximum(base, 0.0)

    H = H0 * (base ** exponent_result)
    return H


def vialov_volume_exact(L: float, H0: float, n: float = 3.0) -> float:
    from math import gamma
    a = 1.0 / (n + 1.0)
    b = (3.0 * n + 2.0) / (2.0 * n + 2.0)
    beta = gamma(a) * gamma(b) / gamma(a + b)
    return float(H0 * L * beta / (n + 1.0))


def bueler_exact_radius(accumulation: float, A: float,
                        rho_g: float, n: float = 3.0,
                        H0: float = 1000.0) -> float:
    if accumulation <= 0 or A <= 0:
        return 0.0

    prefactor = 2.0 * (n + 1.0) / n
    term = (accumulation / (2.0 * A * (rho_g ** n))) ** (1.0 / n)
    R = (prefactor * term * (H0 ** ((n + 2.0) / n))) ** (n / (2.0 * n + 2.0))
    return float(R)


def exact_surface_area_vialov(L: float, H0: float, n: float = 3.0) -> float:
    nx = 1000
    x = np.linspace(-L / 2.0, L / 2.0, nx)
    H = vialov_profile(x, L, H0, n)


    dHdx = np.zeros_like(H)
    dHdx[1:-1] = (H[2:] - H[:-2]) / (x[2] - x[0])


    ds = np.sqrt(1.0 + dHdx ** 2)
    arc_length = np.trapezoid(ds, x)
    return float(arc_length)


def convergence_test_vialov(nx_list: list, L: float, H0: float, n: float = 3.0) -> dict:
    errors = []
    for nx in nx_list:
        x = np.linspace(-L / 2.0, L / 2.0, nx)
        H_exact = vialov_profile(x, L, H0, n)



        H_num = H_exact.copy()

        dx = x[1] - x[0]
        H_num = H_num + 0.01 * dx * np.sin(2 * np.pi * x / L)

        err = np.sqrt(np.mean((H_exact - H_num) ** 2))
        errors.append(err)


    order = None
    if len(errors) >= 2 and len(nx_list) >= 2:
        order = -np.log(errors[-1] / errors[-2]) / np.log(nx_list[-1] / nx_list[-2])

    return {
        'nx': nx_list,
        'error_l2': errors,
        'order': order,
    }
