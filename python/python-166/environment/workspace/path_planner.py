
import numpy as np
from typing import Tuple, List, Callable


def change_dynamic(coin_values: np.ndarray, target: int) -> np.ndarray:
    if target < 1:
        return np.array([])

    INF = target + 1
    dp = np.full(target + 1, INF, dtype=int)
    dp[0] = 0

    for j in range(1, target + 1):
        for c in coin_values:
            if c <= j:
                dp[j] = min(dp[j], dp[j - c] + 1)

    return dp[1:]


def discretize_configuration_space(n_segments: int, n_angles: int,
                                   theta_max: float = np.pi / 2.0) -> np.ndarray:
    if n_angles < 2:
        n_angles = 2
    angles = np.linspace(-theta_max, theta_max, n_angles)
    return angles


def configuration_to_tip(n_segments: int, segment_length: float,
                         angles: np.ndarray) -> Tuple[float, float]:
    x, y = 0.0, 0.0
    phi = 0.0
    for i in range(n_segments):
        phi += angles[i]
        x += segment_length * np.cos(phi)
        y += segment_length * np.sin(phi)
    return x, y


def energy_cost(current: np.ndarray, next_config: np.ndarray,
                stiffness: float = 1.0,
                damping: float = 0.1) -> float:
    diff = next_config - current
    E = 0.5 * stiffness * np.sum(diff ** 2) + 0.5 * damping * np.sum(next_config ** 2)
    return E


def dp_path_planning_2d(n_segments: int, segment_length: float,
                        target: Tuple[float, float],
                        n_discrete: int = 11,
                        theta_max: float = np.pi / 3.0) -> Tuple[np.ndarray, float]:
    angles = discretize_configuration_space(n_segments, n_discrete, theta_max)
    n_a = len(angles)


    if n_segments > 5:

        return _greedy_path_planning(n_segments, segment_length, target, angles)



    INF = 1e10
    dp = np.full((n_segments, n_a), INF)
    parent = np.full((n_segments, n_a), -1, dtype=int)


    for j in range(n_a):

        dp[0, j] = energy_cost(np.array([0.0]), np.array([angles[j]]))


    for i in range(1, n_segments):
        for j in range(n_a):
            for k in range(n_a):
                prev_angles = np.full(i, angles[k])
                curr_angles = np.full(i + 1, angles[j])

                cost = dp[i - 1, k] + energy_cost(np.array([angles[k]]), np.array([angles[j]]))
                if cost < dp[i, j]:
                    dp[i, j] = cost
                    parent[i, j] = k


    best_j = -1
    best_dist = INF
    for j in range(n_a):

        test_angles = np.full(n_segments, angles[j])
        tx, ty = configuration_to_tip(n_segments, segment_length, test_angles)
        dist = (tx - target[0]) ** 2 + (ty - target[1]) ** 2
        total_cost = dp[n_segments - 1, j] + 10.0 * dist
        if total_cost < best_dist:
            best_dist = total_cost
            best_j = j


    optimal_angles = np.zeros(n_segments)
    if best_j >= 0:
        j = best_j
        optimal_angles[n_segments - 1] = angles[j]
        for i in range(n_segments - 1, 0, -1):
            j = parent[i, j]
            if j < 0:
                j = 0
            optimal_angles[i - 1] = angles[j]

    min_cost = dp[n_segments - 1, best_j] if best_j >= 0 else INF
    return optimal_angles, min_cost


def _greedy_path_planning(n_segments: int, segment_length: float,
                          target: Tuple[float, float],
                          angles: np.ndarray) -> Tuple[np.ndarray, float]:
    optimal_angles = np.zeros(n_segments)
    target_x, target_y = target


    target_angle = np.arctan2(target_y, target_x)
    avg_angle = target_angle / n_segments

    for i in range(n_segments):

        idx = np.argmin(np.abs(angles - avg_angle))
        optimal_angles[i] = angles[idx]


    tx, ty = configuration_to_tip(n_segments, segment_length, optimal_angles)
    best_dist = (tx - target_x) ** 2 + (ty - target_y) ** 2

    for _ in range(50):
        improved = False
        for i in range(n_segments):
            for a in angles:
                old = optimal_angles[i]
                optimal_angles[i] = a
                tx, ty = configuration_to_tip(n_segments, segment_length, optimal_angles)
                dist = (tx - target_x) ** 2 + (ty - target_y) ** 2
                if dist < best_dist:
                    best_dist = dist
                    improved = True
                else:
                    optimal_angles[i] = old
        if not improved:
            break

    cost = energy_cost(np.zeros(n_segments), optimal_angles)
    return optimal_angles, cost


def multi_target_path_planning(n_segments: int, segment_length: float,
                               targets: List[Tuple[float, float]],
                               n_discrete: int = 9) -> List[np.ndarray]:
    paths = []
    for idx, target in enumerate(targets):

        angles, _ = dp_path_planning_2d(n_segments, segment_length, target, n_discrete)

        tx, ty = configuration_to_tip(n_segments, segment_length, angles)
        best_dist = (tx - target[0]) ** 2 + (ty - target[1]) ** 2

        rng = np.random.RandomState(idx * 17 + 42)
        angles_test = angles.copy()
        for _ in range(20):
            i_seg = rng.randint(0, n_segments)
            angles_test[i_seg] += rng.uniform(-0.1, 0.1)
            angles_test = np.clip(angles_test, -np.pi / 3.0, np.pi / 3.0)
            tx, ty = configuration_to_tip(n_segments, segment_length, angles_test)
            dist = (tx - target[0]) ** 2 + (ty - target[1]) ** 2
            if dist < best_dist:
                best_dist = dist
                angles = angles_test.copy()
        paths.append(angles.copy())
    return paths
