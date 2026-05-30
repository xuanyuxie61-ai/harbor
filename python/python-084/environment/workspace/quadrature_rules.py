# -*- coding: utf-8 -*-

import numpy as np
from typing import Tuple





def legendre_ek_compute(n: int) -> Tuple[np.ndarray, np.ndarray]:
    if n < 1:
        raise ValueError("n must be >= 1")


    beta = np.zeros(n, dtype=float)
    for i in range(1, n + 1):
        beta[i - 1] = (i * i) / (4.0 * i * i - 1.0)
    beta = np.sqrt(beta)


    J = np.diag(beta[:-1], k=1) + np.diag(beta[:-1], k=-1)


    eigvals, eigvecs = np.linalg.eigh(J)
    x = eigvals

    w = 2.0 * (eigvecs[0, :] ** 2)

    return x, w





def disk01_quadrature_rule(nr: int, nt: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    xr, wr = legendre_ek_compute(nr)

    xr = (xr + 1.0) / 2.0
    wr = wr / 2.0


    t = np.linspace(0.0, 2.0 * np.pi, nt, endpoint=False)
    w = wr / nt
    r = np.sqrt(xr)

    return w, r, t


def integrate_over_disk(
    func: callable,
    nr: int = 12,
    nt: int = 24,
    radius: float = 1.0,
) -> float:
    w, r, theta = disk01_quadrature_rule(nr, nt)




    value = 0.0
    for it in range(nt):
        for ir in range(nr):
            x = radius * r[ir] * np.cos(theta[it])
            y = radius * r[ir] * np.sin(theta[it])
            value += w[ir] * func(x, y)
    value = value * (radius ** 2) * np.pi
    return float(value)





def pyramid01_monomial_integral(expon: Tuple[int, int, int]) -> float:
    ex, ey, ez = expon
    if ex < 0 or ey < 0 or ez < 0:
        raise ValueError("Exponents must be non-negative")

    if (ex % 2 == 1) or (ey % 2 == 1):
        return 0.0

    i_hi = 2 + ex + ey
    value = 0.0
    for i in range(i_hi + 1):
        sign = -1.0 if (i % 2 == 1) else 1.0

        nchoosek = 1.0
        for j in range(1, i + 1):
            nchoosek *= (i_hi + 1 - j) / j
        value += sign * nchoosek / (i + ez + 1)

    value *= (2.0 / (ex + 1)) * (2.0 / (ey + 1))
    return float(value)


def pyramid_volume() -> float:
    return 4.0 / 3.0


def integrate_over_pyramid(
    func: callable,
    n_sample: int = 8,
) -> float:
    rng = np.random.default_rng(seed=42)
    samples = []
    weights = []
    for _ in range(n_sample ** 3):
        z = rng.random()
        max_xy = 1.0 - z
        x = rng.uniform(-max_xy, max_xy)
        y = rng.uniform(-max_xy, max_xy)
        samples.append(func(x, y, z))
        weights.append(1.0)

    samples = np.array(samples)
    volume = pyramid_volume()
    return float(volume * np.mean(samples))





def bearing_contact_force(
    pressure_func: callable,
    bearing_radius: float = 0.25,
    nr: int = 12,
    nt: int = 24,
) -> float:
    return integrate_over_disk(pressure_func, nr=nr, nt=nt, radius=bearing_radius)





def pyramid_consistent_mass(
    rho: float,
    base_area: float,
    height: float,
) -> np.ndarray:
    V = base_area * height / 3.0
    m_total = rho * V
    b = m_total / 100.0
    a = m_total / 5.0 - 4.0 * b

    M = np.full((5, 5), b, dtype=float)
    np.fill_diagonal(M, a)
    return M
