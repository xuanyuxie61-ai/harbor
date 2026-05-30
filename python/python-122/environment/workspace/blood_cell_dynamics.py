
import numpy as np


def fahraeus_lindqvist_viscosity(diameter_um, Hct=0.45):
    d = max(diameter_um, 1e-3)
    mu_plasma = 1.2e-3
    mu_045 = 220.0 * np.exp(-1.3 * d) + 3.2 - 2.44 * np.exp(-0.06 * d ** 0.645)
    C = (0.8 + np.exp(-0.075 * d)) * (-1.0 + 1.0 / (1.0 + 10 ** (-11) * d ** 12)) + \
        1.0 / (1.0 + 10 ** (-11) * d ** 12)
    denom = (1.0 - 0.45) ** C - 1.0
    if abs(denom) < 1e-14:
        denom = 1e-14
    mu_app = mu_plasma * (1.0 + (mu_045 - 1.0) * ((1.0 - Hct) ** C - 1.0) / denom)
    return max(mu_app, mu_plasma)


def hematocrit_partition(Q_parent, Q_daughter1, Q_daughter2, Hct_parent,
                         D_parent, D_d1, D_d2):
    if Q_parent < 1e-14:
        return 0.0, 0.0
    ratio = (Q_daughter1 / (Q_daughter2 + 1e-14)) ** 1.0 * (D_d1 / (D_d2 + 1e-14)) ** 0.5
    Hct_d2 = Hct_parent * Q_parent / (Q_daughter1 * ratio + Q_daughter2 + 1e-14)
    Hct_d1 = Hct_d2 * ratio
    return Hct_d1, Hct_d2



def ppnd(p):
    p = float(p)
    a0 = 2.50662823884
    a1 = -18.61500062529
    a2 = 41.39119773534
    a3 = -25.44106049637
    b1 = -8.47351093090
    b2 = 23.08336743743
    b3 = -21.06224101826
    b4 = 3.13082909833
    c0 = -2.78718931138
    c1 = -2.29796479134
    c2 = 4.85014127135
    c3 = 2.32121276858
    d1 = 3.54388924762
    d2 = 1.63706781897
    split = 0.42

    if p <= 0.0 or p >= 1.0:
        return 0.0, 1

    if abs(p - 0.5) <= split:
        r = (p - 0.5) ** 2
        value = (p - 0.5) * (((a3 * r + a2) * r + a1) * r + a0) / \
                ((((b4 * r + b3) * r + b2) * r + b1) * r + 1.0)
        return value, 0
    else:
        if p > 0.5:
            r = np.sqrt(-np.log(1.0 - p))
        else:
            r = np.sqrt(-np.log(p))
        value = (((c3 * r + c2) * r + c1) * r + c0) / ((d2 * r + d1) * r + 1.0)
        if p < 0.5:
            value = -value
        return value, 0


def blood_flow_variability(mean_flow, std_fraction, n_samples, seed=42):
    np.random.seed(seed)
    u = np.random.uniform(1e-6, 1.0 - 1e-6, n_samples)
    z = np.array([ppnd(ui)[0] for ui in u])
    return mean_flow * (1.0 + std_fraction * z)



def jai_alai_match(strength):
    n = len(strength)
    queue = list(range(n))
    games = 0
    while len(queue) >= 2:
        p1 = queue.pop(0)
        p2 = queue.pop(0)
        games += 1
        total = strength[p1] + strength[p2]
        if total < 1e-14:
            winner = p1
        else:
            winner = p1 if np.random.rand() < strength[p1] / total else p2
        queue.append(winner)
    return queue[0], games


def blood_cell_competition_simulation(strength, n_games):
    stats = np.zeros(len(strength), dtype=int)
    for _ in range(n_games):
        winner, _ = jai_alai_match(strength)
        stats[winner] += 1
    return stats


def stochastic_pulsatile_flow(Q_mean, f_heart, t_array, amplitude=0.3, phase_noise_std=0.1):
    phi = np.random.normal(0.0, phase_noise_std)
    Q = Q_mean * (1.0 + amplitude * np.sin(2.0 * np.pi * f_heart * t_array + phi))
    noise = Q_mean * amplitude * 0.1 * np.random.randn(len(t_array))
    return np.maximum(Q + noise, 0.0)
