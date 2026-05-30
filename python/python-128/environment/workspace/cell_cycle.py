
import numpy as np


class CellCyclePhase:
    G1 = 0
    S = 1
    G2 = 2
    M = 3


def caesar_cycle_shift(phase_state: np.ndarray, k: int = 1):
    phase_state = np.asarray(phase_state, dtype=float)
    if phase_state.size != 4:
        raise ValueError("caesar_cycle_shift: phase_state 长度必须为 4")
    k = int(k) % 4
    if k == 0:
        return phase_state.copy()
    return np.roll(phase_state, k)


def cycle_transition_matrix(k: int = 1):
    k = int(k) % 4
    P = np.zeros((4, 4), dtype=float)
    for j in range(4):
        i = (j + k) % 4
        P[i, j] = 1.0
    return P


def chemotaxis_sensitivity_by_phase(phase_index: int,
                                    w_max: float = 1.0,
                                    w_min: float = 0.1):
    phase_index = int(phase_index) % 4
    theta = np.pi * phase_index / 2.0 - np.pi
    val = (1.0 + np.cos(theta)) / 2.0
    return float(w_min + (w_max - w_min) * val)


def advance_cell_cycle(phase_dist: np.ndarray,
                       dt: float,
                       transition_rates: np.ndarray = None):
    p = np.asarray(phase_dist, dtype=float)
    if abs(p.sum()) < 1e-15:
        raise ValueError("advance_cell_cycle: phase_dist 全为零")
    p = p / p.sum()

    if transition_rates is None:
        transition_rates = np.array([0.3, 0.5, 0.4, 0.6], dtype=float)
    else:
        transition_rates = np.asarray(transition_rates, dtype=float)

    if transition_rates.size != 4:
        raise ValueError("advance_cell_cycle: transition_rates 长度必须为 4")

    k = transition_rates

    Q = np.array([
        [-k[0], 0.0,   0.0,   k[3]],
        [ k[0], -k[1], 0.0,   0.0 ],
        [ 0.0,  k[1], -k[2], 0.0 ],
        [ 0.0,  0.0,   k[2], -k[3]]
    ], dtype=float)



    A = np.eye(4) - dt * Q
    p_new = np.linalg.solve(A, p)
    p_new = np.maximum(p_new, 0.0)
    s = p_new.sum()
    if s > 1e-15:
        p_new /= s
    return p_new


def population_weighted_chemotaxis_sensitivity(phase_dist: np.ndarray,
                                                w_max: float = 1.0,
                                                w_min: float = 0.1):
    p = np.asarray(phase_dist, dtype=float)
    if p.sum() < 1e-15:
        return 0.0
    p = p / p.sum()
    w = 0.0
    for phi in range(4):
        w += p[phi] * chemotaxis_sensitivity_by_phase(phi, w_max, w_min)
    return float(w)
