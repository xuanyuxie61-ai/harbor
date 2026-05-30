
import numpy as np
from utils import cholesky_factor, hypersphere_surface_area


def uniform_in_sphere01_map(dim_num, n, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    exponent = 1.0 / dim_num
    x = np.zeros((dim_num, n), dtype=float)
    for j in range(n):
        z = rng.standard_normal(dim_num)
        z_norm = np.linalg.norm(z)
        if z_norm < 1e-15:
            z_norm = 1.0
        u = z / z_norm
        r = rng.random() ** exponent
        x[:, j] = r * u
    return x


def ellipse_sample(n, A, r, rng=None):
    A = np.array(A, dtype=float)
    if A.shape != (2, 2):
        raise ValueError("A must be 2x2")
    U = cholesky_factor(A)
    y = uniform_in_sphere01_map(2, n, rng=rng) * r

    x = np.linalg.solve(U, y)
    return x


def ellipse_area(A, r):
    det_a = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
    if det_a <= 0:
        raise ValueError("A must be positive definite")
    return np.pi * r * r / np.sqrt(det_a)


def hypersphere01_monomial_integral(dim, expon):
    expon = np.array(expon, dtype=int)
    if np.any(expon < 0):
        raise ValueError("exponents must be nonnegative")
    if np.any(expon % 2 == 1):
        return 0.0
    from math import gamma as math_gamma
    num = 2.0
    for e in expon:
        num *= math_gamma((e + 1) / 2.0)
    den = math_gamma((dim + np.sum(expon)) / 2.0)
    return float(num / den)


def hypersphere_monte_carlo_integral(dim, n_samples, func, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    area = hypersphere_surface_area(dim)

    samples = rng.standard_normal((dim, n_samples))
    norms = np.linalg.norm(samples, axis=0)
    norms = np.where(norms < 1e-15, 1.0, norms)
    samples = samples / norms
    vals = np.array([func(samples[:, i]) for i in range(n_samples)])
    return float(area * np.mean(vals)), float(area * np.std(vals) / np.sqrt(max(n_samples, 1)))


def hypercube_distance_stats(dim, n_samples, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    p1 = rng.random((dim, n_samples))
    p2 = rng.random((dim, n_samples))
    dists = np.sqrt(np.sum((p1 - p2) ** 2, axis=0))
    mu = float(np.mean(dists))
    if n_samples > 1:
        var = float(np.sum((dists - mu) ** 2) / (n_samples - 1))
    else:
        var = 0.0
    return mu, var


def antithetic_variates_integral(dim, n_pairs, func, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    u = rng.random((dim, n_pairs))
    x1 = u
    x2 = 1.0 - u
    f1 = np.array([func(x1[:, i]) for i in range(n_pairs)])
    f2 = np.array([func(x2[:, i]) for i in range(n_pairs)])
    estimates = (f1 + f2) / 2.0
    return float(np.mean(estimates)), float(np.std(estimates) / np.sqrt(max(n_pairs, 1)))
