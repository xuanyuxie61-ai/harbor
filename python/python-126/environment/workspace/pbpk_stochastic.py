
import numpy as np
from typing import Callable, Tuple





def random_walk_3d_step(pos: np.ndarray, h: float) -> np.ndarray:
    direction = np.random.randint(0, 6)
    step = np.zeros(3)
    step_size = np.sqrt(3.0 * h)
    if direction == 0:
        step[0] = step_size
    elif direction == 1:
        step[0] = -step_size
    elif direction == 2:
        step[1] = step_size
    elif direction == 3:
        step[1] = -step_size
    elif direction == 4:
        step[2] = step_size
    else:
        step[2] = -step_size
    return pos + step


def inside_ellipsoid(pos: np.ndarray, a: float, b: float, c: float) -> bool:
    val = (pos[0] / a) ** 2 + (pos[1] / b) ** 2 + (pos[2] / c) ** 2
    return val <= 1.0


def feynman_kac_3d_monte_carlo(x0: float, y0: float, z0: float,
                                a: float, b: float, c: float,
                                potential: Callable[[np.ndarray], float],
                                boundary_value: Callable[[np.ndarray], float],
                                h: float = 0.01, n_trajectories: int = 10000,
                                max_steps: int = 100000) -> Tuple[float, float]:
    pos0 = np.array([x0, y0, z0])
    if not inside_ellipsoid(pos0, a, b, c):
        raise ValueError("Initial position must be inside the ellipsoid")
    if h <= 0.0 or n_trajectories <= 0:
        raise ValueError("Invalid Monte Carlo parameters")

    estimates = np.empty(n_trajectories)
    for k in range(n_trajectories):
        pos = pos0.copy()
        Y = 1.0
        steps = 0
        while steps < max_steps:
            V = potential(pos)

            Y *= np.exp(-V * h)
            pos = random_walk_3d_step(pos, h)
            steps += 1
            if not inside_ellipsoid(pos, a, b, c):
                break

        g = boundary_value(pos)
        estimates[k] = Y * g

    mean = np.mean(estimates)
    se = np.std(estimates, ddof=1) / np.sqrt(n_trajectories)
    return mean, se






def drug_absorption_probability_organ(center: np.ndarray, organ_axes: np.ndarray,
                                       D_eff: float, clearance: float,
                                       n_trajectories: int = 5000) -> Tuple[float, float]:
    if len(center) != 3 or len(organ_axes) != 3:
        raise ValueError("center and organ_axes must be 3D vectors")
    a, b, c = organ_axes
    x0, y0, z0 = center

    def potential(pos):
        return clearance / max(D_eff, 1e-20)

    def boundary_value(pos):

        return 0.0

    mean, se = feynman_kac_3d_monte_carlo(x0, y0, z0, a, b, c,
                                           potential, boundary_value,
                                           h=0.001, n_trajectories=n_trajectories,
                                           max_steps=50000)
    return mean, se


def organ_hitting_probability(source_pos: np.ndarray, target_pos: np.ndarray,
                               organ_axes: np.ndarray, D_eff: float,
                               n_trajectories: int = 2000,
                               h: float = 1e-6) -> float:
    if len(source_pos) != 3 or len(target_pos) != 3:
        raise ValueError("Positions must be 3D")
    a, b, c = organ_axes
    if not inside_ellipsoid(source_pos, a, b, c):
        raise ValueError("Source must be inside organ")
    if not inside_ellipsoid(target_pos, a, b, c):
        raise ValueError("Target must be inside organ")

    epsilon = 0.05 * min(a, b, c)
    hits = 0
    max_steps = 100000
    for _ in range(n_trajectories):
        pos = source_pos.copy()
        for _ in range(max_steps):
            pos = random_walk_3d_step(pos, h)
            dist = np.linalg.norm(pos - target_pos)
            if dist < epsilon:
                hits += 1
                break
            if not inside_ellipsoid(pos, a, b, c):
                break
    return hits / n_trajectories






def feynman_kac_1d(x0: float, L: float, V_const: float,
                    g_left: float, g_right: float,
                    h: float = 0.001, n_trajectories: int = 5000) -> float:
    if not (0.0 <= x0 <= L):
        raise ValueError("x0 must be in [0, L]")
    estimates = np.empty(n_trajectories)
    for k in range(n_trajectories):
        x = x0
        Y = 1.0
        while True:
            Y *= np.exp(-V_const * h)

            x += np.sqrt(h) if np.random.rand() < 0.5 else -np.sqrt(h)
            if x <= 0.0:
                estimates[k] = Y * g_left
                break
            if x >= L:
                estimates[k] = Y * g_right
                break
    return np.mean(estimates)






if __name__ == "__main__":

    L, V, gL, gR = 1.0, 2.0, 0.0, 1.0
    x0 = 0.5
    mc_val = feynman_kac_1d(x0, L, V, gL, gR, h=0.0005, n_trajectories=20000)

    sqrt2V = np.sqrt(2.0 * V)
    A = (gR - gL * np.cosh(sqrt2V * L)) / np.sinh(sqrt2V * L)
    B = gL
    exact = A * np.sinh(sqrt2V * x0) + B * np.cosh(sqrt2V * x0)
    print(f"1D Feynman-Kac: MC={mc_val:.6f}, Exact={exact:.6f}, RelErr={abs(mc_val-exact)/exact:.4e}")

    mean, se = feynman_kac_3d_monte_carlo(0.0, 0.0, 0.0, 1.0, 0.8, 0.6,
                                           lambda p: 1.0,
                                           lambda p: np.exp((p[0]/1.0)**2 + (p[1]/0.8)**2 + (p[2]/0.6)**2 - 1.0),
                                           h=0.01, n_trajectories=2000)
    print(f"3D Feynman-Kac estimate: {mean:.6f} ± {se:.6f}")
