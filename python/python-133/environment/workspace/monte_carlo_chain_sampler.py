
import numpy as np
from typing import Tuple, Optional


def cholesky_factor(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    m = a.shape[0]
    u = np.zeros_like(a)

    for j in range(m):
        s = 0.0
        for k in range(j):
            s += u[k, j] ** 2
        diff = a[j, j] - s
        if diff <= 1.0e-14:
            diff = 1.0e-14
        u[j, j] = np.sqrt(diff)
        for i in range(j + 1, m):
            s = 0.0
            for k in range(j):
                s += u[k, i] * u[k, j]
            u[j, i] = (a[j, i] - s) / u[j, j]

    return u


def cholesky_solve(u: np.ndarray, b: np.ndarray) -> np.ndarray:
    b = np.asarray(b, dtype=float)
    m = u.shape[0]
    x = b.copy()

    for j in range(m):
        x[j] = x[j] / u[j, j]
        for i in range(j + 1, m):
            x[i] -= u[j, i] * x[j]

    return x


def sample_uniform_in_ball(m: int, n: int, radius: float = 1.0,
                           rng: Optional[np.random.Generator] = None) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng(seed=133)

    samples = np.zeros((m, n))
    for j in range(n):
        xi = rng.standard_normal(m)
        norm_xi = np.linalg.norm(xi)
        if norm_xi < 1.0e-12:
            xi[0] = 1.0
            norm_xi = 1.0
        u = rng.random()
        r = radius * (u ** (1.0 / m))
        samples[:, j] = r * xi / norm_xi

    return samples


def ellipse_sample(n: int, a_mat: np.ndarray, r: float = 1.0,
                   rng: Optional[np.random.Generator] = None) -> np.ndarray:
    a_mat = np.asarray(a_mat, dtype=float)
    m = a_mat.shape[0]
    u = cholesky_factor(a_mat)

    y_samples = sample_uniform_in_ball(m, n, radius=r, rng=rng)
    x_samples = np.zeros((m, n))
    for j in range(n):
        x_samples[:, j] = cholesky_solve(u, y_samples[:, j])

    return x_samples


def polymer_chain_gyration_tensor(chain_coords: np.ndarray) -> np.ndarray:
    coords = np.asarray(chain_coords, dtype=float)
    r_cm = np.mean(coords, axis=0)
    centered = coords - r_cm
    A = (centered.T @ centered) / coords.shape[0]
    return A


def radius_of_gyration(chain_coords: np.ndarray) -> float:
    A = polymer_chain_gyration_tensor(chain_coords)
    d = A.shape[0]
    return np.sqrt(np.trace(A) / d)


def disk_triangle_picking(n_trials: int,
                          rng: Optional[np.random.Generator] = None) -> float:
    if rng is None:
        rng = np.random.default_rng(seed=42)

    total_area = 0.0
    for _ in range(n_trials):
        theta = 2.0 * np.pi * rng.random(3)
        r = np.sqrt(rng.random(3))
        x = r * np.cos(theta)
        y = r * np.sin(theta)

        s1 = np.sqrt((x[0] - x[1]) ** 2 + (y[0] - y[1]) ** 2)
        s2 = np.sqrt((x[1] - x[2]) ** 2 + (y[1] - y[2]) ** 2)
        s3 = np.sqrt((x[2] - x[0]) ** 2 + (y[2] - y[0]) ** 2)
        s = 0.5 * (s1 + s2 + s3)


        area_sq = s * (s - s1) * (s - s2) * (s - s3)
        area_sq = max(area_sq, 0.0)
        area = np.sqrt(area_sq)
        total_area += area

    return total_area / n_trials


def mixing_efficiency_estimate(n_trials: int = 10000,
                               rng: Optional[np.random.Generator] = None) -> Tuple[float, float]:
    avg_area = disk_triangle_picking(n_trials, rng=rng)
    disk_area = np.pi
    efficiency = avg_area / disk_area

    theoretical = 35.0 / (48.0 * np.pi)
    return avg_area, efficiency, theoretical


def coarse_grained_chain_mc(n_segments: int,
                            n_samples: int,
                            kuhn_length: float = 1.0,
                            confinement_ellipsoid: Optional[np.ndarray] = None,
                            rng: Optional[np.random.Generator] = None) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng(seed=2024)

    cov = (kuhn_length ** 2 / 3.0) * n_segments * np.eye(3)
    samples = rng.multivariate_normal(mean=np.zeros(3), cov=cov, size=n_samples)

    if confinement_ellipsoid is not None:
        a_mat = np.asarray(confinement_ellipsoid, dtype=float)
        accepted = []
        for s in samples:
            if s @ a_mat @ s <= 1.0:
                accepted.append(s)
        if len(accepted) == 0:

            return np.zeros((1, 3))
        samples = np.array(accepted)

    return samples


def critical_pore_size(chain_samples: np.ndarray,
                       porosity: float = 0.4) -> float:
    Rg = np.mean(np.linalg.norm(chain_samples, axis=1))
    dc = 2.0 * Rg / ((1.0 - porosity) ** (1.0 / 3.0) + 1.0e-12)
    return dc
