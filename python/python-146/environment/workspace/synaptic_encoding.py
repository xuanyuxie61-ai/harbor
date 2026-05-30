
import numpy as np


class AlphaSynapse:

    def __init__(self, tau_s=2.0):
        if tau_s <= 0:
            raise ValueError("tau_s must be positive.")
        self.tau_s = tau_s

    def kernel(self, t):
        t = np.atleast_1d(t)
        K = np.zeros_like(t, dtype=float)
        mask = t > 0
        K[mask] = (t[mask] / self.tau_s) * np.exp(-t[mask] / self.tau_s)
        return K

    def convolve_spikes(self, spike_times, weights, t_grid):
        s = np.zeros_like(t_grid, dtype=float)
        for ti, wi in zip(spike_times, weights):
            dt = t_grid - ti
            s += wi * self.kernel(dt)
        return s


def polynomial_multiply_convolution(a, b):
    a = np.atleast_1d(a)
    b = np.atleast_1d(b)

    pn = len(a)
    while pn > 1 and np.isclose(a[pn - 1], 0.0):
        pn -= 1
    qn = len(b)
    while qn > 1 and np.isclose(b[qn - 1], 0.0):
        qn -= 1

    rn = pn + qn - 1
    c = np.zeros(rn)
    for i in range(pn):
        for j in range(qn):
            k = i + j
            if k < rn:
                c[k] += a[i] * b[j]
    return c


def rational_knapsack_encoding(profits, weights, budget):
    profits = np.asarray(profits, dtype=float)
    weights = np.asarray(weights, dtype=float)
    N = len(profits)
    if N != len(weights):
        raise ValueError("profits and weights must have same length.")
    if budget < 0:
        raise ValueError("budget must be non-negative.")
    if np.any(weights < 0):
        raise ValueError("weights must be non-negative.")
    if np.any(profits < 0):
        raise ValueError("profits must be non-negative.")


    safe_weights = np.where(weights == 0, 1e-12, weights)
    density = profits / safe_weights


    order = np.argsort(-density)

    x = np.zeros(N)
    mass = 0.0
    profit = 0.0

    for idx in order:
        wi = weights[idx]
        pi = profits[idx]
        if mass >= budget - 1e-12:
            x[idx] = 0.0
            continue
        if mass + wi <= budget:
            x[idx] = 1.0
            mass += wi
            profit += pi
        else:
            remaining = budget - mass
            if wi > 0:
                frac = remaining / wi
            else:
                frac = 1.0
            frac = np.clip(frac, 0.0, 1.0)
            x[idx] = frac
            mass = budget
            profit += pi * frac

    return x, mass, profit


def optimal_synaptic_weights(spike_times, signal_target, t_grid, tau_s=2.0, E_budget=10.0, sigma_noise=0.5):
    synapse = AlphaSynapse(tau_s)
    n_spikes = len(spike_times)
    if n_spikes == 0:
        return np.array([]), np.zeros_like(t_grid), 0.0


    basis = np.zeros((len(t_grid), n_spikes))
    for j, tj in enumerate(spike_times):
        basis[:, j] = synapse.kernel(t_grid - tj)



    profits = np.zeros(n_spikes)
    for j in range(n_spikes):

        proj = np.dot(basis[:, j], signal_target) / (np.dot(basis[:, j], basis[:, j]) + 1e-12)
        approx = proj * basis[:, j]
        profits[j] = np.linalg.norm(approx) ** 2


    max_profit = np.max(profits)
    if max_profit > 0:
        profits = profits / max_profit


    candidate_weights = np.ones(n_spikes)
    x, _, _ = rational_knapsack_encoding(profits, candidate_weights, E_budget)


    active = x > 0.01
    if not np.any(active):
        active[0] = True

    basis_active = basis[:, active]

    lam = 0.01
    A = basis_active.T @ basis_active + lam * np.eye(basis_active.shape[1])
    b_vec = basis_active.T @ signal_target
    try:
        w_active = np.linalg.solve(A, b_vec)
    except np.linalg.LinAlgError:
        w_active = np.linalg.lstsq(basis_active, signal_target, rcond=None)[0]


    l1_norm = np.sum(np.abs(w_active))
    if l1_norm > E_budget and l1_norm > 0:
        w_active = w_active * (E_budget / l1_norm)

    weights = np.zeros(n_spikes)
    weights[active] = w_active

    encoded_signal = basis @ weights


    var_signal = np.var(encoded_signal)
    var_noise = sigma_noise ** 2
    if var_signal > 0 and var_noise > 0:
        snr = var_signal / var_noise
        mutual_info = 0.5 * np.log2(1.0 + snr)
    else:
        mutual_info = 0.0

    return weights, encoded_signal, mutual_info


def demo_encoding():
    t_grid = np.linspace(0, 100, 1000)

    signal_target = 5.0 * np.sin(2.0 * np.pi * 0.05 * t_grid) + 0.5 * np.random.randn(len(t_grid))

    np.random.seed(7)
    rate = 0.1
    spike_times = []
    t = 0.0
    while t < 100.0:
        dt_spike = np.random.exponential(1.0 / rate)
        t += dt_spike
        if t < 100.0:
            spike_times.append(t)
    weights, encoded, mi = optimal_synaptic_weights(
        spike_times, signal_target, t_grid, tau_s=3.0, E_budget=8.0, sigma_noise=0.5
    )
    return weights, encoded, mi, spike_times
