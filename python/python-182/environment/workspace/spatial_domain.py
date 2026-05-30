import math
import numpy as np


def annulus_grid(r1: float, r2: float, n_r: int, n_theta: int):
    if r1 < 0 or r2 <= r1 or n_r < 1 or n_theta < 1:
        raise ValueError("annulus_grid: invalid parameters")
    rs = np.linspace(r1, r2, n_r)
    thetas = np.linspace(0.0, 2.0 * math.pi, n_theta, endpoint=False)
    R, T = np.meshgrid(rs, thetas, indexing='ij')
    x = R.flatten() * np.cos(T.flatten())
    y = R.flatten() * np.sin(T.flatten())
    return x, y


def annulus_grid_fibonacci(r1: float, r2: float, n: int):
    if n < 1 or r1 < 0 or r2 <= r1:
        raise ValueError("annulus_grid_fibonacci: invalid parameters")
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    golden_angle = 2.0 * math.pi / (phi * phi)
    i = np.arange(1, n + 1, dtype=float)
    r = np.sqrt(r1 * r1 + (r2 * r2 - r1 * r1) * (i - 0.5) / n)
    theta = golden_angle * i
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y


def fem_basis_2d(i: int, j: int, k: int, x: float, y: float) -> float:
    if i < 0 or j < 0 or k < 0:
        raise ValueError("fem_basis_2d: indices must be non-negative")
    d = i + j + k
    if d == 0:
        return 1.0

    lijk = 1.0
    cijk = 1.0
    for p in range(i):
        lijk *= (d * x - p)
        cijk *= (i - p)
    for p in range(j):
        lijk *= (d * y - p)
        cijk *= (j - p)
    for p in range(k):
        lijk *= (d * (x + y) - (d - p))
        cijk *= ((i + j) - (d - p))

    if abs(cijk) < 1e-15:
        return 0.0
    return lijk / cijk


def fem_basis_eval_on_triangle(degree: int, x: float, y: float):
    results = []
    for ii in range(degree + 1):
        for jj in range(degree + 1 - ii):
            kk = degree - ii - jj
            val = fem_basis_2d(ii, jj, kk, x, y)
            results.append((ii, jj, kk, val))
    return results


def circle_distance_pdf(d: float) -> float:
    if d < 0.0 or d > 2.0:
        return 0.0
    denom = math.pi * math.sqrt(max(0.0, 1.0 - 0.25 * d * d))
    if denom < 1e-15:
        return 0.0
    return 1.0 / denom


def circle_distance_exact_mean() -> float:
    return 4.0 / math.pi


def circle_distance_exact_variance() -> float:
    return 2.0 - 16.0 / (math.pi * math.pi)
