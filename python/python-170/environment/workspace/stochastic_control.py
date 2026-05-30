
import numpy as np


def potential(a: float, x: np.ndarray):
    return 2.0 * (x / a / a) ** 2 + 1.0 / a / a


def feynman_kac_1d_solve(a: float, h: float, n_paths: int, n_grid: int = 21):
    if a <= 0 or h <= 0 or n_paths <= 0:
        raise ValueError("a, h, n_paths must be positive.")

    rth = np.sqrt(h)
    ni = n_grid
    xs = np.linspace(-a, a, ni + 2)
    u_approx = np.zeros_like(xs)
    u_exact = np.exp((xs / a) ** 2 - 1.0)

    err_sum = 0.0
    n_int = 0

    for idx, x in enumerate(xs):
        test = a * a - x * x
        if test < 0.0:
            u_approx[idx] = 1.0
            continue

        n_int += 1
        total = 0.0
        steps_total = 0

        for _ in range(n_paths):
            x1 = x
            w = 1.0
            chk = 0.0
            steps = 0
            while chk < 1.0:
                us = np.random.rand() - 0.5
                dx = -rth if us < 0.0 else rth
                vs = potential(a, x1)
                x1 = x1 + dx
                steps += 1
                vh = potential(a, x1)
                we = (1.0 - h * vs) * w
                w = w - 0.5 * h * (vh * we + vs * w)
                chk = (x1 / a) ** 2
            total += w
            steps_total += steps

        u_approx[idx] = total / n_paths
        err_sum += (u_exact[idx] - u_approx[idx]) ** 2

    rms_error = np.sqrt(err_sum / max(n_int, 1))
    return xs, u_approx, u_exact, rms_error


def feynman_kac_collision_potential(positions: np.ndarray, obstacles: np.ndarray,
                                    obstacle_radius: float, domain_radius: float,
                                    n_paths: int = 200, h: float = 0.02):
    N = positions.shape[0]
    pot = np.zeros(N, dtype=float)
    for i in range(N):
        p = positions[i]

        dist_obs = np.min(np.linalg.norm(obstacles - p, axis=1)) if obstacles.shape[0] > 0 else domain_radius
        r_eff = min(np.linalg.norm(p), domain_radius)


        boundary_term = np.exp(-(r_eff / domain_radius) ** 2)
        obs_term = np.exp(-(dist_obs / obstacle_radius) ** 2)
        pot[i] = boundary_term + obs_term
    return pot


def gradient_fk_potential(positions: np.ndarray, obstacles: np.ndarray,
                          obstacle_radius: float, domain_radius: float, eps: float = 1e-4):
    N = positions.shape[0]
    grad = np.zeros((N, 2), dtype=float)
    f0 = feynman_kac_collision_potential(positions, obstacles, obstacle_radius, domain_radius)
    for dim in range(2):
        pos_plus = positions.copy()
        pos_plus[:, dim] += eps
        f_plus = feynman_kac_collision_potential(pos_plus, obstacles, obstacle_radius, domain_radius)
        grad[:, dim] = (f_plus - f0) / eps
    return grad
