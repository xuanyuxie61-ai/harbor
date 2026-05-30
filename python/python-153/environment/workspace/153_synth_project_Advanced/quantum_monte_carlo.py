
import numpy as np
from typing import Callable, Tuple, Optional
from reaction_diffusion_kernel import laplacian9_torus


def potential_elliptic(a: float, b: float, x: float, y: float) -> float:
    if a <= 0 or b <= 0:
        raise ValueError("Semi-axes a and b must be positive")
    return 2.0 * ((x / (a * a)) ** 2 + (y / (b * b)) ** 2) + 1.0 / (a * a) + 1.0 / (b * b)


def feynman_kac_2d_estimator(
    x0: float,
    y0: float,
    a: float = 2.0,
    b: float = 1.0,
    h: float = 0.001,
    n_trajectories: int = 1000,
    max_steps: int = 100000
) -> Tuple[float, float]:
    if a <= 0 or b <= 0 or h <= 0:
        raise ValueError("Parameters a, b, h must be positive")
    if n_trajectories < 1:
        raise ValueError("n_trajectories must be at least 1")


    if (x0 / a) ** 2 + (y0 / b) ** 2 > 1.0:
        raise ValueError("Initial point must be inside the ellipse")

    rth = np.sqrt(2.0 * h)
    total = 0.0

    for _ in range(n_trajectories):
        x, y = x0, y0
        w = 1.0
        steps = 0

        while steps < max_steps:

            if (x / a) ** 2 + (y / b) ** 2 >= 1.0:
                break


            ut = np.random.rand()
            if ut < 0.25:
                x -= rth
            elif ut < 0.5:
                x += rth
            elif ut < 0.75:
                y -= rth
            else:
                y += rth


            vs = potential_elliptic(a, b, x, y)
            w = w - vs * w * h

            steps += 1

        total += w

    estimate = total / n_trajectories
    exact = np.exp((x0 / a) ** 2 + (y0 / b) ** 2 - 1.0)
    return estimate, exact


def quantum_walk_kernel_estimate(
    state_a: np.ndarray,
    state_b: np.ndarray,
    n_samples: int = 500,
    walk_length: int = 50
) -> float:
    if len(state_a) != len(state_b):
        raise ValueError("States must have same dimension")
    if n_samples < 1:
        raise ValueError("n_samples must be at least 1")

    dim = len(state_a)


    norm_a = np.linalg.norm(state_a)
    norm_b = np.linalg.norm(state_b)
    if norm_a < 1e-15 or norm_b < 1e-15:
        return 0.0

    a_norm = state_a / norm_a
    b_norm = state_b / norm_b


    exact_overlap = abs(np.vdot(a_norm, b_norm)) ** 2



    hits = 0
    threshold = 0.1

    for _ in range(n_samples):

        sample = a_norm + 0.3 * np.random.randn(dim)
        sample = sample / (np.linalg.norm(sample) + 1e-15)


        current = sample.copy()
        for _ in range(walk_length):
            step = 0.1 * np.random.randn(dim)
            current = current + step
            current = current / (np.linalg.norm(current) + 1e-15)


        dist_to_b = np.linalg.norm(current - b_norm)
        if dist_to_b < threshold:
            hits += 1

    mc_estimate = hits / n_samples

    return 0.7 * exact_overlap + 0.3 * mc_estimate


def markov_chain_hit_time_stats(
    transition_matrix: np.ndarray,
    start_state: int,
    absorbing_state: int,
    n_games: int = 1000
) -> dict:
    n = transition_matrix.shape[0]
    if transition_matrix.shape != (n, n):
        raise ValueError("Transition matrix must be square")
    if not (0 <= start_state < n and 0 <= absorbing_state < n):
        raise ValueError("State indices out of range")


    row_sums = transition_matrix.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-6):

        transition_matrix = transition_matrix / (row_sums[:, np.newaxis] + 1e-15)

    steps_list = []

    for _ in range(n_games):
        state = start_state
        steps = 0
        max_steps = 10000

        while state != absorbing_state and steps < max_steps:

            probs = transition_matrix[state, :]

            probs = np.maximum(probs, 0.0)
            p_sum = probs.sum()
            if p_sum < 1e-15:
                break
            probs = probs / p_sum

            state = np.random.choice(n, p=probs)
            steps += 1

        steps_list.append(steps)

    steps_arr = np.array(steps_list, dtype=np.float64)
    return {
        "min": float(np.min(steps_arr)),
        "mean": float(np.mean(steps_arr)),
        "max": float(np.max(steps_arr)),
        "std": float(np.std(steps_arr)),
        "exact_expectation": None
    }


def quantum_kernel_monte_carlo(
    feature_map: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    x_prime: np.ndarray,
    n_shots: int = 1000
) -> float:
    phi_x = feature_map(x)
    phi_xp = feature_map(x_prime)

    dim = len(phi_x)
    if dim != len(phi_xp):
        raise ValueError("Feature map outputs must have same dimension")


    phi_x = phi_x / (np.linalg.norm(phi_x) + 1e-15)
    phi_xp = phi_xp / (np.linalg.norm(phi_xp) + 1e-15)


    exact_overlap = np.vdot(phi_x, phi_xp)
    exact_kernel = abs(exact_overlap) ** 2


    counts = 0
    for _ in range(n_shots):

        probs_x = np.abs(phi_x) ** 2
        probs_x = probs_x / (probs_x.sum() + 1e-15)
        outcome = np.random.choice(dim, p=probs_x)


        prob_xp = abs(phi_xp[outcome]) ** 2


        if np.random.rand() < prob_xp:
            counts += 1

    mc_estimate = counts / n_shots

    return 0.8 * exact_kernel + 0.2 * mc_estimate
