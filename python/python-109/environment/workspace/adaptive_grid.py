
import numpy as np
from typing import Callable, Tuple


def cvt_energy(dim_num: int, n: int, r: np.ndarray, sample_num: int,
               sample_points: np.ndarray) -> float:
    if r.shape != (dim_num, n):
        raise ValueError("cvt_energy: r shape mismatch")
    if sample_points.shape != (dim_num, sample_num):
        raise ValueError("cvt_energy: sample_points shape mismatch")
    total = 0.0
    for k in range(sample_num):
        diff = r - sample_points[:, k:k + 1]
        dist2 = np.sum(diff ** 2, axis=0)
        total += np.min(dist2)
    return total / sample_num


def find_closest(dim_num: int, n: int, r: np.ndarray, sample_points: np.ndarray) -> np.ndarray:
    m = sample_points.shape[1]
    closest = np.zeros(m, dtype=int)
    for k in range(m):
        diff = r - sample_points[:, k:k + 1]
        dist2 = np.sum(diff ** 2, axis=0)
        closest[k] = int(np.argmin(dist2))
    return closest


def cvt_iterate(dim_num: int, n: int, sample_points: np.ndarray,
                r: np.ndarray) -> Tuple[np.ndarray, float, float]:
    m = sample_points.shape[1]
    r_new = np.zeros_like(r)
    counts = np.zeros(n, dtype=int)
    closest = find_closest(dim_num, n, r, sample_points)
    for k in range(m):
        idx = closest[k]
        r_new[:, idx] += sample_points[:, k]
        counts[idx] += 1

    for i in range(n):
        if counts[i] > 0:
            r_new[:, i] /= counts[i]
        else:
            r_new[:, i] = r[:, i]
    it_diff = np.linalg.norm(r_new - r, 'fro')
    energy = cvt_energy(dim_num, n, r_new, m, sample_points)
    return r_new, it_diff, energy


def cvt_sample_uniform(dim_num: int, n_samples: int, bounds: Tuple[float, float]) -> np.ndarray:
    low, high = bounds
    return np.random.uniform(low, high, (dim_num, n_samples))


def adaptive_cvt_grid(n_points: int, density_func: Callable,
                      domain: Tuple[float, float],
                      n_samples: int = 5000,
                      it_max: int = 30,
                      tol: float = 1e-5) -> np.ndarray:
    if n_points < 2:
        raise ValueError("adaptive_cvt_grid: n_points must be >= 2")
    x_min, x_max = domain
    dim_num = 1

    r = np.linspace(x_min, x_max, n_points).reshape(1, n_points)


    uniform = np.random.uniform(x_min, x_max, n_samples * 2)
    weights = np.maximum(density_func(uniform), 1e-12)
    weights /= np.sum(weights)

    sample_points = np.random.choice(uniform, size=n_samples, p=weights, replace=True)
    sample_points = sample_points.reshape(1, n_samples)
    for it in range(it_max):
        r_new, it_diff, _ = cvt_iterate(dim_num, n_points, sample_points, r)
        r = r_new
        if it_diff < tol:
            break
    points = r.flatten()
    points = np.sort(points)

    points = np.clip(points, x_min, x_max)
    return points


def spectral_boundary_detect(power_spectrum: np.ndarray,
                              omega: np.ndarray,
                              threshold_db: float = -30.0) -> Tuple[float, float]:
    if len(power_spectrum) != len(omega):
        raise ValueError("spectral_boundary_detect: length mismatch")
    p_max = np.max(power_spectrum)
    if p_max <= 0.0:
        return float(omega[0]), float(omega[-1])
    p_db = 10.0 * np.log10(power_spectrum / p_max + 1e-20)
    mask = p_db > threshold_db
    indices = np.where(mask)[0]
    if len(indices) == 0:
        return float(omega[0]), float(omega[-1])
    idx_left = int(indices[0])
    idx_right = int(indices[-1])

    if idx_left > 0:
        t = (threshold_db - p_db[idx_left - 1]) / (p_db[idx_left] - p_db[idx_left - 1] + 1e-20)
        omega_left = omega[idx_left - 1] + t * (omega[idx_left] - omega[idx_left - 1])
    else:
        omega_left = float(omega[idx_left])
    if idx_right < len(omega) - 1:
        t = (threshold_db - p_db[idx_right]) / (p_db[idx_right + 1] - p_db[idx_right] + 1e-20)
        omega_right = omega[idx_right] + t * (omega[idx_right + 1] - omega[idx_right])
    else:
        omega_right = float(omega[idx_right])
    return float(omega_left), float(omega_right)


def log_spaced_grid(omega_min: float, omega_max: float, n_points: int) -> np.ndarray:
    if omega_min <= 0.0 or omega_max <= omega_min:
        raise ValueError("log_spaced_grid: invalid frequency range")
    log_min = np.log10(omega_min)
    log_max = np.log10(omega_max)
    log_grid = np.linspace(log_min, log_max, n_points)
    return 10.0 ** log_grid
