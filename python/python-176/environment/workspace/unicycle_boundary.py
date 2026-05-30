
import numpy as np


def unicycle_dynamics(state, control):
    x, y, theta = state
    v, omega = control
    return np.array([v * np.cos(theta), v * np.sin(theta), omega], dtype=float)


def unicycle_integrate_rk4(state0, control_trajectory, dt):
    N = control_trajectory.shape[0]
    states = np.zeros((N + 1, 3), dtype=float)
    states[0] = state0
    for n in range(N):
        u = control_trajectory[n]
        k1 = unicycle_dynamics(states[n], u)
        k2 = unicycle_dynamics(states[n] + 0.5 * dt * k1, u)
        k3 = unicycle_dynamics(states[n] + 0.5 * dt * k2, u)
        k4 = unicycle_dynamics(states[n] + dt * k3, u)
        states[n + 1] = states[n] + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return states


def parametric_ellipse_boundary(a, b, n_segments):
    theta = np.linspace(0.0, 2.0 * np.pi, n_segments + 1)

    dtheta = 2.0 * np.pi / n_segments
    arc_lengths = []
    for i in range(n_segments):
        th = theta[i]
        ds = np.sqrt((a * np.sin(th)) ** 2 + (b * np.cos(th)) ** 2) * dtheta
        arc_lengths.append(ds)
    return theta, np.array(arc_lengths)


def boundary_actuator_positions(a, b, n_acts, t, speeds, theta0=None):
    if theta0 is None:
        theta0 = np.linspace(0.0, 2.0 * np.pi, n_acts, endpoint=False)
    theta = (theta0 + speeds * t) % (2.0 * np.pi)
    x = a * np.cos(theta)
    y = b * np.sin(theta)
    return np.column_stack((x, y)), theta


def actuator_control_to_boundary(a, b, n_boundary_nodes, boundary_nodes_coords,
                                 actuator_positions, actuator_values,
                                 sigma=0.1):
    q = np.zeros(n_boundary_nodes, dtype=float)
    for pos, val in zip(actuator_positions, actuator_values):
        dist2 = (boundary_nodes_coords[:, 0] - pos[0]) ** 2 + (boundary_nodes_coords[:, 1] - pos[1]) ** 2
        q += val * np.exp(-dist2 / (2.0 * sigma ** 2))
    return q


def levenshtein_distance(s, t):
    m = len(s)
    n = len(t)

    prev = np.arange(n + 1, dtype=int)
    curr = np.zeros(n + 1, dtype=int)

    for i in range(1, m + 1):
        curr[0] = i
        for j in range(1, n + 1):
            cost = 0 if s[i - 1] == t[j - 1] else 1
            curr[j] = min(curr[j - 1] + 1,
                          prev[j] + 1,
                          prev[j - 1] + cost)
        prev, curr = curr, prev

    return int(prev[n])


def sequence_similarity_score(seq1, seq2):
    dist = levenshtein_distance(seq1, seq2)
    maxlen = max(len(seq1), len(seq2))
    if maxlen == 0:
        return 1.0
    return 1.0 - dist / maxlen


def rank_boundary_control_sequence(q_values, n_bins=10):
    q_min = np.min(q_values)
    q_max = np.max(q_values)
    if abs(q_max - q_min) < 1.0e-15:
        return ['0'] * len(q_values)
    bins = np.linspace(q_min, q_max, n_bins + 1)
    symbols = np.digitize(q_values, bins) - 1
    symbols = np.clip(symbols, 0, n_bins - 1)
    return [str(s) for s in symbols]


def random_unicycle_path(a, b, T, n_steps, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    dt = T / n_steps
    v = 0.5 + 0.3 * rng.random(n_steps)
    omega = 0.5 * (rng.random(n_steps) - 0.5)
    controls = np.column_stack((v, omega))
    theta0 = rng.random() * 2.0 * np.pi
    state0 = np.array([a * np.cos(theta0), b * np.sin(theta0), theta0])
    return unicycle_integrate_rk4(state0, controls, dt)
