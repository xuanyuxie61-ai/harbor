
import numpy as np
from math import sqrt, isclose
from typing import Callable, Tuple





def prime_count(n: int) -> int:
    if n < 2:
        return 0
    if n == 2:
        return 1
    total = 1
    for i in range(3, n + 1, 2):
        p = 1
        limit = int(sqrt(i)) + 1
        for j in range(3, limit, 2):
            if i % j == 0:
                p = 0
                break
        total += p
    return total


def generate_primes(n: int) -> np.ndarray:
    if n <= 0:
        return np.array([], dtype=int)
    primes = []
    candidate = 2
    while len(primes) < n:
        is_p = True
        limit = int(sqrt(candidate)) + 1
        for p in primes:
            if p > limit:
                break
            if candidate % p == 0:
                is_p = False
                break
        if is_p:
            primes.append(candidate)
        candidate += 1 if candidate == 2 else 2
    return np.array(primes, dtype=int)


def seeded_random(seed_prime_index: int, size: Tuple[int, ...]) -> np.ndarray:
    if seed_prime_index < 0:
        seed_prime_index = 0
    primes = generate_primes(seed_prime_index + 1)
    seed_val = int(primes[-1])
    rng = np.random.RandomState(seed=seed_val)
    return rng.rand(*size)






def legendre_gauss_nodes(n: int) -> Tuple[np.ndarray, np.ndarray]:
    if n < 1:
        raise ValueError("legendre_gauss_nodes: n 必须 >= 1")
    if n > 100:

        nodes, weights = np.polynomial.legendre.leggauss(n)
        return nodes, weights
    

    nodes, weights = np.polynomial.legendre.leggauss(n)
    return nodes, weights


def integrate_2d_gauss(
    f: Callable[[np.ndarray, np.ndarray], np.ndarray],
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    nx: int = 16,
    ny: int = 16,
) -> float:
    if nx < 1 or ny < 1:
        raise ValueError("integrate_2d_gauss: nx, ny 必须 >= 1")
    
    ax, bx = xlim
    ay, by = ylim
    
    if not (np.isfinite(ax) and np.isfinite(bx) and np.isfinite(ay) and np.isfinite(by)):
        raise ValueError("integrate_2d_gauss: 积分限必须为有限值")
    
    if isclose(bx, ax) or isclose(by, ay):
        return 0.0
    
    x_nodes, x_weights = legendre_gauss_nodes(nx)
    y_nodes, y_weights = legendre_gauss_nodes(ny)
    

    x = 0.5 * (bx - ax) * x_nodes + 0.5 * (bx + ax)
    y = 0.5 * (by - ay) * y_nodes + 0.5 * (by + ay)
    wx = 0.5 * (bx - ax) * x_weights
    wy = 0.5 * (by - ay) * y_weights
    

    X, Y = np.meshgrid(x, y, indexing='ij')
    

    with np.errstate(invalid='ignore', divide='ignore'):
        F = f(X, Y)
    

    if np.any(~np.isfinite(F)):
        F = np.where(np.isfinite(F), F, 0.0)
    

    integral = 0.0
    for i in range(nx):
        for j in range(ny):
            integral += wx[i] * wy[j] * F[i, j]
    
    return float(integral)






def safe_divide(a: np.ndarray, b: np.ndarray, fill_value: float = 0.0) -> np.ndarray:
    b = np.asarray(b)
    result = np.full_like(a, fill_value, dtype=float)
    mask = np.abs(b) > 1e-15
    result[mask] = a[mask] / b[mask]
    return result


def soft_cutoff(r: np.ndarray, rc: float, delta: float = 0.05) -> np.ndarray:
    r = np.asarray(r)
    result = np.zeros_like(r, dtype=float)
    mask_in = r < (rc - delta)
    mask_trans = (r >= (rc - delta)) & (r <= rc)
    result[mask_in] = 1.0
    if np.any(mask_trans):
        result[mask_trans] = 0.5 + 0.5 * np.cos(np.pi * (r[mask_trans] - rc + delta) / delta)
    return result


def distance_matrix_pbc(positions: np.ndarray, box: np.ndarray) -> np.ndarray:
    N = positions.shape[0]
    dim = positions.shape[1]
    if box.shape[0] != dim:
        raise ValueError("box 维度与 positions 不匹配")
    
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]

    for d in range(dim):
        if box[d] > 0:
            diff[:, :, d] -= box[d] * np.rint(diff[:, :, d] / box[d])
    
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    return dist


def mean_squared_displacement(trajectory: np.ndarray) -> np.ndarray:
    n_frames = trajectory.shape[0]
    if n_frames < 2:
        return np.zeros(n_frames)
    
    msd = np.zeros(n_frames)
    counts = np.zeros(n_frames)
    
    for dt in range(1, n_frames):
        displacements = trajectory[dt:, :, :] - trajectory[:-dt, :, :]
        sq_disp = np.sum(displacements ** 2, axis=2)
        msd[dt] = np.mean(sq_disp)
        counts[dt] = sq_disp.size
    
    msd[0] = 0.0
    counts[0] = 1.0
    return msd
