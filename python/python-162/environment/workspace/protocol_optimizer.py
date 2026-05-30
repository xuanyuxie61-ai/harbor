
import numpy as np
from typing import List, Tuple, Callable
from particle_distribution import transfer_clustering, swap_clustering






def perm1_next3(p: np.ndarray) -> bool:
    n = len(p)

    d = np.ones(n, dtype=int)


    mobile = -1
    mobile_idx = -1
    for i in range(n):
        if d[i] == 1 and i < n - 1 and p[i] > p[i + 1]:
            if p[i] > mobile:
                mobile = p[i]
                mobile_idx = i
        elif d[i] == -1 and i > 0 and p[i] > p[i - 1]:
            if p[i] > mobile:
                mobile = p[i]
                mobile_idx = i
    if mobile_idx == -1:
        return False

    if d[mobile_idx] == 1:
        p[mobile_idx], p[mobile_idx + 1] = p[mobile_idx + 1], p[mobile_idx]
        d[mobile_idx], d[mobile_idx + 1] = d[mobile_idx + 1], d[mobile_idx]
    else:
        p[mobile_idx], p[mobile_idx - 1] = p[mobile_idx - 1], p[mobile_idx]
        d[mobile_idx], d[mobile_idx - 1] = d[mobile_idx - 1], d[mobile_idx]

    for i in range(n):
        if p[i] > mobile:
            d[i] = -d[i]
    return True


def all_permutations(n: int):
    p = np.arange(n, dtype=int)
    d = np.ones(n, dtype=int)
    yield p.copy()
    while True:
        mobile = -1
        mobile_idx = -1
        for i in range(n):
            if d[i] == 1 and i < n - 1 and p[i] > p[i + 1]:
                if p[i] > mobile:
                    mobile = p[i]
                    mobile_idx = i
            elif d[i] == -1 and i > 0 and p[i] > p[i - 1]:
                if p[i] > mobile:
                    mobile = p[i]
                    mobile_idx = i
        if mobile_idx == -1:
            break
        if d[mobile_idx] == 1:
            p[mobile_idx], p[mobile_idx + 1] = p[mobile_idx + 1], p[mobile_idx]
            d[mobile_idx], d[mobile_idx + 1] = d[mobile_idx + 1], d[mobile_idx]
        else:
            p[mobile_idx], p[mobile_idx - 1] = p[mobile_idx - 1], p[mobile_idx]
            d[mobile_idx], d[mobile_idx - 1] = d[mobile_idx - 1], d[mobile_idx]
        for i in range(n):
            if p[i] > mobile:
                d[i] = -d[i]
        yield p.copy()


def brute_force_protocol_search(
    current_levels: np.ndarray,
    durations: np.ndarray,
    objective: Callable[[np.ndarray, np.ndarray], float]
) -> Tuple[np.ndarray, np.ndarray, float]:
    n = len(current_levels)
    if n > 6:
        raise ValueError("Brute force limited to N <= 6")
    best_perm = np.arange(n)
    best_cost = objective(current_levels[best_perm], durations[best_perm])
    for perm in all_permutations(n):
        cost = objective(current_levels[perm], durations[perm])
        if cost < best_cost:
            best_cost = cost
            best_perm = perm.copy()
    return current_levels[best_perm], durations[best_perm], best_cost






def thermal_charge_objective(currents: np.ndarray, durations: np.ndarray,
                             thermal_model_func: Callable,
                             max_temp_limit: float = 318.15) -> float:
    total_time = np.sum(durations)


    R_int = 0.01
    q_joule = np.sum(currents ** 2 * R_int * durations)
    T_rise_est = q_joule / 100.0
    penalty = max(0.0, T_rise_est - (max_temp_limit - 298.15))
    return total_time + 1e4 * penalty






def greedy_thermal_protocol(
    target_capacity: float,
    current_options: np.ndarray,
    duration_step: float,
    thermal_model_func: Callable,
    max_temp: float = 318.15
) -> Tuple[List[float], List[float]]:
    protocol_I = []
    protocol_t = []
    remaining = target_capacity
    t_accum = 0.0
    while remaining > 1e-6:
        best_I = current_options[0]
        best_dt = duration_step
        for I in sorted(current_options, reverse=True):
            dt = min(duration_step, remaining / abs(I + 1e-18))

            T_est = 298.15 + I ** 2 * 0.01 * (t_accum + dt) / 100.0
            if T_est <= max_temp:
                best_I = I
                best_dt = dt
                break
        protocol_I.append(best_I)
        protocol_t.append(best_dt)
        remaining -= abs(best_I) * best_dt
        t_accum += best_dt
        if len(protocol_I) > 1000:
            break
    return protocol_I, protocol_t






def cluster_protocol_segments(
    currents: np.ndarray,
    n_segments: int = 4
) -> Tuple[np.ndarray, np.ndarray]:
    if len(currents) < n_segments:
        n_segments = max(1, len(currents))
    labels = np.zeros(len(currents), dtype=int)
    for k in range(n_segments):
        lo = int(k * len(currents) / n_segments)
        hi = int((k + 1) * len(currents) / n_segments)
        if hi <= lo and k > 0:
            hi = lo + 1
        labels[lo:hi] = k
    labels = transfer_clustering(currents, labels, n_segments)
    labels = swap_clustering(currents, labels, n_segments)
    centers = np.array([np.mean(currents[labels == k]) if np.any(labels == k) else 0.0
                        for k in range(n_segments)])
    return labels, centers
