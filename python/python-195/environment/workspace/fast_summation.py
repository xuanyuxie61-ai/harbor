
import numpy as np
from typing import Tuple, Optional
from utils import EPSILON_MACHINE


def toeplitz_mv(n: int, a: np.ndarray, x: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    x = np.asarray(x, dtype=float)
    if a.size < 2 * n - 1:
        raise ValueError(f"a too short: need {2*n-1}, got {a.size}")
    if x.size < n:
        raise ValueError(f"x too short: need {n}, got {x.size}")

    b = np.zeros(n, dtype=float)
    for i in range(n):

        for j in range(i, n):
            b[i] += a[j - i] * x[j]

        for j in range(i):
            b[i] += a[n + i - j - 1] * x[j]
    return b


def toeplitz_embedded_fft_mv(n: int, a: np.ndarray, x: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    x = np.asarray(x, dtype=float)
    m = 2 * n - 1



    c = np.zeros(m, dtype=complex)
    c[0] = a[0]
    if n > 1:
        c[1:n] = a[n:2*n-1]
        c[n:] = a[n-1:0:-1]

    x_padded = np.zeros(m, dtype=complex)
    x_padded[:n] = x[:n]

    y = np.fft.ifft(np.fft.fft(c) * np.fft.fft(x_padded))
    return np.real(y[:n])


def sample_unit_ball_positive(n_samples: int) -> np.ndarray:
    if n_samples <= 0:
        return np.empty((0, 3))
    v = np.random.randn(n_samples, 3)
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms = np.maximum(norms, EPSILON_MACHINE)
    u = np.abs(v) / norms
    r = np.random.rand(n_samples, 1) ** (1.0 / 3.0)
    return r * u


def sample_unit_sphere_surface(n_samples: int, dim: int = 3) -> np.ndarray:
    if n_samples <= 0:
        return np.empty((0, dim))
    v = np.random.randn(n_samples, dim)
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms = np.maximum(norms, EPSILON_MACHINE)
    return v / norms


def compute_prefix_sum_2d(particles: np.ndarray,
                          domain: Tuple[float, float, float, float],
                          nx: int, ny: int) -> np.ndarray:
    particles = np.asarray(particles, dtype=float)
    xmin, xmax, ymin, ymax = domain
    dx = (xmax - xmin) / nx
    dy = (ymax - ymin) / ny

    grid = np.zeros((nx, ny), dtype=int)
    for p in range(particles.shape[0]):
        ix = int((particles[p, 0] - xmin) / dx)
        iy = int((particles[p, 1] - ymin) / dy)
        ix = max(0, min(nx - 1, ix))
        iy = max(0, min(ny - 1, iy))
        grid[ix, iy] += 1

    prefix = np.zeros((nx + 1, ny + 1), dtype=int)
    for i in range(nx):
        for j in range(ny):
            prefix[i + 1, j + 1] = (
                grid[i, j]
                + prefix[i, j + 1]
                + prefix[i + 1, j]
                - prefix[i, j]
            )
    return prefix


def query_region_count(prefix: np.ndarray,
                       ix1: int, ix2: int,
                       iy1: int, iy2: int) -> int:
    ix1 = max(0, ix1)
    iy1 = max(0, iy1)
    ix2 = min(prefix.shape[0] - 1, ix2)
    iy2 = min(prefix.shape[1] - 1, iy2)
    return (
        prefix[ix2, iy2]
        - prefix[ix1, iy2]
        - prefix[ix2, iy1]
        + prefix[ix1, iy1]
    )


def multipole_expansion(particles: np.ndarray, charges: np.ndarray,
                        center: np.ndarray, max_order: int = 2) -> np.ndarray:
    particles = np.asarray(particles, dtype=float)
    charges = np.asarray(charges, dtype=float)
    center = np.asarray(center, dtype=float)
    d = particles.shape[1]

    result = {}

    result['monopole'] = np.sum(charges)

    if max_order >= 1:

        dipole = np.zeros(d)
        for j in range(len(charges)):
            dipole += charges[j] * (particles[j] - center)
        result['dipole'] = dipole

    if max_order >= 2:

        quadrupole = np.zeros((d, d))
        for j in range(len(charges)):
            r = particles[j] - center
            quadrupole += charges[j] * np.outer(r, r)
        result['quadrupole'] = quadrupole

    return result


def build_interaction_matrix_toeplitz(n: int, kernel_func: callable,
                                      h: float = 1.0) -> np.ndarray:
    a = np.zeros(2 * n - 1, dtype=float)

    for k in range(n):
        r = k * h
        a[k] = kernel_func(r) if r > 1e-14 else kernel_func(h * 1e-10)

    for k in range(1, n):
        r = k * h
        a[n + k - 1] = kernel_func(r)
    return a
