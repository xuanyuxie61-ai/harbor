# -*- coding: utf-8 -*-

import numpy as np
from scipy.special import betainc, gammaln


def _log_gamma(x):
    return gammaln(x)


def noncentral_beta_cdf(x, a, b, lam, error_max=1e-12):
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    if a <= 0 or b <= 0:
        raise ValueError("Shape parameters a and b must be positive.")
    if lam < 0:
        raise ValueError("Noncentrality lambda must be non-negative.")


    half_lam = lam / 2.0
    pi_val = np.exp(-half_lam)
    if pi_val == 0.0:

        return float(betainc(a + half_lam, b, x))


    beta_log = _log_gamma(a) + _log_gamma(b) - _log_gamma(a + b)


    bi = float(betainc(a, b, x))
    si = np.exp(a * np.log(x) + b * np.log(1.0 - x) - beta_log - np.log(a))

    p_sum = pi_val
    pb_sum = pi_val * bi
    i = 0

    while p_sum < 1.0 - error_max and i < 10000:
        i += 1
        pi_val = half_lam * pi_val / i
        bi = bi - si
        si = x * (a + b + i - 1) * si / (a + i)
        p_sum += pi_val
        pb_sum += pi_val * bi

        if not np.isfinite(pb_sum):
            break

    return float(np.clip(pb_sum, 0.0, 1.0))


def noncentral_beta_pdf(x, a, b, lam, dx=1e-6):
    x = np.clip(x, dx, 1.0 - dx)
    return (noncentral_beta_cdf(x + dx, a, b, lam) -
            noncentral_beta_cdf(x - dx, a, b, lam)) / (2.0 * dx)


def decay_chain_simulation(initial_state, transition_matrix, n_steps,
                           n_samples=1000, seed=None):
    if seed is not None:
        np.random.seed(seed)

    T = np.asarray(transition_matrix, dtype=float)
    n_states = T.shape[0]

    row_sums = T.sum(axis=1)
    for i in range(n_states):
        if row_sums[i] <= 0:
            T[i, i] = 1.0
        else:
            T[i, :] /= row_sums[i]


    cdf = np.cumsum(T, axis=1)
    cdf[:, -1] = 1.0

    trajectories = np.empty((n_samples, n_steps + 1), dtype=int)
    trajectories[:, 0] = initial_state

    for step in range(1, n_steps + 1):
        u = np.random.rand(n_samples)
        for samp in range(n_samples):
            state = trajectories[samp, step - 1]

            next_state = np.searchsorted(cdf[state, :], u[samp])
            next_state = min(next_state, n_states - 1)
            trajectories[samp, step] = next_state

    populations = np.zeros((n_steps + 1, n_states))
    for step in range(n_steps + 1):
        for s in range(n_states):
            populations[step, s] = np.mean(trajectories[:, step] == s)

    return populations, trajectories


def beta_decay_halflife(Z, Q_mev, Bgt=1.0, ft_const=6147.0):
    if Q_mev <= 0:
        return np.inf
    alpha = 1.0 / 137.035999084

    f_approx = (Q_mev ** 5) / 30.0 * max(1.0 - 2.0 * np.pi * alpha * Z, 0.1)
    if f_approx <= 0:
        return np.inf
    return ft_const / (f_approx * Bgt)


def q_value_beta_decay(M_parent, M_daughter, Q_ec=0.0):
    return M_parent - M_daughter + Q_ec


def neutron_drip_line_uncertainty(N_obs, Z, confidence=0.95, eff_bias=0.05):
    N_max = max(N_obs + 10, 20)
    a = N_obs + 1.0
    b = N_max - N_obs + 1.0
    lam = eff_bias * Z


    def find_quantile(p_target):
        lo, hi = 0.0, 1.0
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if noncentral_beta_cdf(mid, a, b, lam) < p_target:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    alpha = 0.5 * (1.0 - confidence)
    lower = find_quantile(alpha)
    upper = find_quantile(1.0 - alpha)

    mean = (a + 0.5 * lam) / (a + b + lam)
    mean = np.clip(mean, 0.0, 1.0)
    return lower, upper, mean
