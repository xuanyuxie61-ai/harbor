
import numpy as np
from numpy.linalg import cholesky, solve, norm




WATER_DENSITY = 998.0
WATER_VISCOSITY = 1.002e-3
SURFACE_TENSION = 0.0728
SOUND_SPEED_WATER = 1482.0
VAPOR_PRESSURE = 2338.0
ATMOSPHERIC_PRESSURE = 101325.0
GAS_CONSTANT = 8.314
BOLTZMANN = 1.380649e-23


def safe_divide(a, b, default=0.0):
    b = np.asarray(b)
    a = np.asarray(a)
    result = np.empty_like(a, dtype=float)
    mask = np.abs(b) > 1e-30
    result[mask] = a[mask] / b[mask]
    result[~mask] = default
    return result


def r8po_fa(n, a):
    a_full = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i, n):
            a_full[i, j] = a[i, j]
            a_full[j, i] = a[i, j]
    try:
        u = cholesky(a_full).T
        return u, 0
    except np.linalg.LinAlgError:
        return np.zeros((n, n)), 1


def r8po_sl(n, u, b):
    A = u.T @ u
    x = solve(A, b)
    return x


def uniform_in_sphere01_map(m, n):
    x = np.random.randn(m, n)
    norms = np.sqrt(np.sum(x**2, axis=0))
    norms = np.maximum(norms, 1e-15)
    x = x / norms
    r = np.random.uniform(0.0, 1.0, size=n) ** (1.0 / m)
    return x * r


def ellipsoid_sample(m, n, a_mat, v, r):
    u, info = r8po_fa(m, a_mat)
    if info != 0:
        raise ValueError("矩阵 A 不是正定对称矩阵")
    y = uniform_in_sphere01_map(m, n) * r
    x = np.zeros((m, n))
    for j in range(n):
        x[:, j] = r8po_sl(m, u, y[:, j])
    for i in range(m):
        x[i, :] += v[i]
    return x


def disk01_sample(n):
    x = np.random.randn(2, n)
    norms = np.sqrt(np.sum(x**2, axis=0))
    norms = np.maximum(norms, 1e-15)
    x = x / norms
    r = np.sqrt(np.random.uniform(0.0, 1.0, size=n))
    return x * r


def monomial_value(m, n, e, x):
    v = np.ones(n, dtype=float)
    for i in range(m):
        if e[i] != 0:
            v *= x[i, :] ** e[i]
    return v


def print_matrix(mat, title=""):
    if title:
        print(f"\n{title}")
    print(np.array2string(np.asarray(mat), precision=6, suppress_small=True))


def timestamp():
    from datetime import datetime
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
