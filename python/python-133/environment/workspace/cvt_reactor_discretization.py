
import numpy as np
from typing import Tuple, Callable, Optional


def chebyshev_zero_density_1d(n: int) -> np.ndarray:
    x = np.zeros(n)
    for i in range(n):
        x[i] = np.cos((2.0 * i + 1.0) * np.pi / (2.0 * n))
    return x


def chebyshev_zero_density_2d(nx: int, ny: int) -> Tuple[np.ndarray, np.ndarray]:
    x = chebyshev_zero_density_1d(nx)
    y = chebyshev_zero_density_1d(ny)
    X, Y = np.meshgrid(x, y)
    return X, Y


def cvt_2d_lloyd(n_generators: int,
                 n_iterations: int,
                 n_samples: int,
                 density_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
                 x_min: float = -1.0,
                 x_max: float = 1.0,
                 rng: Optional[np.random.Generator] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if rng is None:
        rng = np.random.default_rng(seed=42)

    if n_generators < 3:
        raise ValueError("n_generators must be at least 3")


    g = x_min + (x_max - x_min) * rng.random((n_generators, 2))
    g_new = np.zeros_like(g)

    energy_history = np.zeros(n_iterations)
    motion_history = np.zeros(n_iterations)


    eps_margin = 1.0e-6 * (x_max - x_min)
    s_1d = np.linspace(x_min + eps_margin, x_max - eps_margin, n_samples)
    sx_mat, sy_mat = np.meshgrid(s_1d, s_1d)
    sx_vec = sx_mat.flatten()
    sy_vec = sy_mat.flatten()


    rho_mat = density_func(sx_mat, sy_mat)
    rho_mat = np.minimum(rho_mat, 10.0)

    r_mat = rho_mat ** 4
    r_vec = r_mat.flatten()

    points = np.column_stack((sx_vec, sy_vec))

    for it in range(n_iterations):



        diff = points[:, np.newaxis, :] - g[np.newaxis, :, :]
        dists_sq = np.sum(diff ** 2, axis=2)
        nearest = np.argmin(dists_sq, axis=1)


        mass = np.zeros(n_generators)
        centroid_x = np.zeros(n_generators)
        centroid_y = np.zeros(n_generators)

        for k in range(n_generators):
            mask = (nearest == k)
            mass[k] = np.sum(r_vec[mask])
            if mass[k] > 1.0e-15:
                centroid_x[k] = np.sum(r_vec[mask] * sx_vec[mask]) / mass[k]
                centroid_y[k] = np.sum(r_vec[mask] * sy_vec[mask]) / mass[k]
            else:

                centroid_x[k] = g[k, 0]
                centroid_y[k] = g[k, 1]

        g_new[:, 0] = centroid_x
        g_new[:, 1] = centroid_y


        energy = 0.0
        for idx_pt, k in enumerate(nearest):
            energy += r_vec[idx_pt] * dists_sq[idx_pt, k]
        energy_history[it] = energy / n_samples


        motion = np.mean(np.sum((g_new - g) ** 2, axis=1))
        motion_history[it] = motion

        g = g_new.copy()


        g = np.clip(g, x_min, x_max)

    return g, energy_history, motion_history


def reaction_rate_density(x: np.ndarray, y: np.ndarray,
                          peak_x: float = 0.0,
                          peak_y: float = 0.0,
                          sigma: float = 0.5,
                          amplitude: float = 2.0) -> np.ndarray:
    X = np.asarray(x)
    Y = np.asarray(y)
    rho = amplitude * np.exp(-((X - peak_x) ** 2 + (Y - peak_y) ** 2) / (2.0 * sigma ** 2)) + 0.1
    return rho


def optimal_reactor_nodes(n_nodes: int = 20,
                          n_iter: int = 30,
                          n_samples: int = 80) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    g, energy, motion = cvt_2d_lloyd(
        n_generators=n_nodes,
        n_iterations=n_iter,
        n_samples=n_samples,
        density_func=lambda x, y: reaction_rate_density(x, y),
        x_min=-1.0, x_max=1.0
    )
    return g, energy, motion
