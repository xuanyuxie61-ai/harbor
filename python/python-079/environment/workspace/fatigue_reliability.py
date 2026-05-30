
import numpy as np
import math
from typing import Tuple, List, Dict






def rainflow_count_cycles(
    signal: np.ndarray,
) -> List[Tuple[float, float]]:
    signal = np.asarray(signal, dtype=float)
    if len(signal) < 3:
        return []


    peaks_valleys = [signal[0]]
    for i in range(1, len(signal) - 1):
        if (signal[i] >= signal[i - 1] and signal[i] > signal[i + 1]) or \
           (signal[i] <= signal[i - 1] and signal[i] < signal[i + 1]):
            peaks_valleys.append(signal[i])
    peaks_valleys.append(signal[-1])

    cycles = []
    stack = []
    for s in peaks_valleys:
        stack.append(s)
        while len(stack) >= 3:
            s1 = stack[-3]
            s2 = stack[-2]
            s3 = stack[-1]
            delta1 = abs(s2 - s1)
            delta2 = abs(s3 - s2)
            if delta1 >= delta2:

                amp = delta1 * 0.5
                mean = (s1 + s2) * 0.5
                cycles.append((amp, mean))
                stack.pop(-2)
                stack.pop(-2)
            else:
                break

    for i in range(len(stack) - 1):
        amp = abs(stack[i + 1] - stack[i]) * 0.5
        mean = (stack[i] + stack[i + 1]) * 0.5
        cycles.append((amp, mean))
    return cycles


def rainflow_histogram(
    signal: np.ndarray, n_bins: int = 20
) -> Tuple[np.ndarray, np.ndarray]:
    cycles = rainflow_count_cycles(signal)
    if len(cycles) == 0:
        return np.zeros(n_bins + 1), np.zeros(n_bins)
    amplitudes = np.array([c[0] for c in cycles])
    min_amp = np.min(amplitudes)
    max_amp = np.max(amplitudes)
    if max_amp - min_amp < 1e-12:
        return np.zeros(n_bins + 1), np.zeros(n_bins)
    counts, edges = np.histogram(amplitudes, bins=n_bins, range=(min_amp, max_amp))
    return edges, counts






def sn_curve_cycles(
    stress_range: float,
    a: float = 1.0e12,
    m: float = 3.0,
    threshold: float = 1.0,
) -> float:
    if stress_range < threshold:
        return float('inf')
    return a * (stress_range ** (-m))


def miner_damage(
    cycles: List[Tuple[float, float]],
    a: float = 1.0e12,
    m: float = 3.0,
    threshold: float = 1.0,
) -> float:
    D = 0.0
    for amp, _ in cycles:
        S = 2.0 * amp
        N = sn_curve_cycles(S, a, m, threshold)
        if N < float('inf') and N > 0:
            D += 1.0 / N
    return D


def annual_fatigue_damage(
    stress_signal_annual: np.ndarray,
    a: float = 1.0e12,
    m: float = 3.0,
) -> float:
    cycles = rainflow_count_cycles(stress_signal_annual)
    return miner_damage(cycles, a, m)






def build_seastate_markov_chain(
    n_states: int = 8,
    transition_exponent: float = 2.0,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    np.random.seed(seed)
    P = np.zeros((n_states, n_states))
    for i in range(n_states):
        for j in range(n_states):
            dist = abs(i - j)
            if dist == 0:
                P[i, j] = 0.6
            elif dist == 1:
                P[i, j] = 0.25 / transition_exponent
            else:
                P[i, j] = 0.15 * np.exp(-dist) / (n_states - 1)

        row_sum = np.sum(P[i, :])
        if row_sum > 0:
            P[i, :] /= row_sum
        else:
            P[i, i] = 1.0


    steady = np.zeros(n_states)
    steady[0] = 1.0
    for _ in range(200):
        steady_new = steady @ P
        if np.max(np.abs(steady_new - steady)) < 1e-12:
            steady = steady_new
            break
        steady = steady_new
    steady = np.maximum(steady, 0.0)
    steady /= np.sum(steady)


    state_labels = np.linspace(0.5, 12.0, n_states)
    return P, steady, state_labels


def compute_longterm_fatigue_damage_markov(
    P: np.ndarray,
    steady_state: np.ndarray,
    state_damage_rates: np.ndarray,
) -> float:
    if P.shape[0] != len(steady_state) or len(steady_state) != len(state_damage_rates):
        raise ValueError("马尔可夫链维度不匹配")
    return float(np.dot(steady_state, state_damage_rates))


def simulate_markov_chain_trajectory(
    P: np.ndarray,
    initial_state: int,
    n_steps: int,
    seed: int = 42,
) -> np.ndarray:
    np.random.seed(seed)
    n_states = P.shape[0]
    traj = np.zeros(n_steps, dtype=int)
    state = initial_state
    traj[0] = state
    for t in range(1, n_steps):
        state = np.random.choice(n_states, p=P[state, :])
        traj[t] = state
    return traj


def markov_chain_n_step_distribution(
    P: np.ndarray,
    initial_dist: np.ndarray,
    n: int,
) -> np.ndarray:
    dist = initial_dist.copy()
    for _ in range(n):
        dist = dist @ P
    return dist






def reliability_index(
    mean_resistance: float,
    std_resistance: float,
    mean_load: float,
    std_load: float,
) -> float:
    denom = np.sqrt(std_resistance ** 2 + std_load ** 2)
    if denom < 1e-15:
        return float('inf')
    return (mean_resistance - mean_load) / denom


def failure_probability_from_beta(beta: float) -> float:
    return 0.5 * (1.0 - math.erf(beta / np.sqrt(2.0)))


def fatigue_life_prediction(
    annual_damage: float,
    design_life_years: float = 25.0,
    safety_factor: float = 10.0,
) -> Dict[str, float]:
    if annual_damage <= 1e-15:
        predicted_life = float('inf')
    else:
        predicted_life = 1.0 / annual_damage
    allowable_damage = 1.0 / safety_factor
    years_to_failure = predicted_life
    is_safe = years_to_failure > design_life_years
    return {
        "annual_damage": annual_damage,
        "predicted_life_years": predicted_life,
        "design_life_years": design_life_years,
        "safety_factor": safety_factor,
        "allowable_damage": allowable_damage,
        "years_to_failure": years_to_failure,
        "is_safe": is_safe,
    }






def stress_from_platform_response(
    surge: np.ndarray,
    heave: np.ndarray,
    pitch: np.ndarray,
    stress_factor_surge: float = 2.5e5,
    stress_factor_heave: float = 1.8e5,
    stress_factor_pitch: float = 3.2e6,
    noise_level: float = 0.02,
) -> np.ndarray:
    stress = (
        stress_factor_surge * surge
        + stress_factor_heave * heave
        + stress_factor_pitch * pitch
    )
    noise = noise_level * np.std(stress) * np.random.randn(len(stress))
    return stress + noise
