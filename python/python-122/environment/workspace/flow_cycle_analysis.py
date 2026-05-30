
import numpy as np


def cycle_brent(f, x0):
    power = 1
    lam = 1
    tortoise = x0
    hare = f(x0)

    while tortoise != hare:
        if power == lam:
            tortoise = hare
            power *= 2
            lam = 0
        hare = f(hare)
        lam += 1

    mu = 0
    tortoise = x0
    hare = x0
    for _ in range(lam):
        hare = f(hare)

    while tortoise != hare:
        tortoise = f(tortoise)
        hare = f(hare)
        mu += 1

    return lam, mu


def cerebrovascular_autoregulation_map(cbf, params):
    MAP = params['MAP']
    MAP_ss = params['MAP_ss']
    CBF_ss = params['CBF_ss']
    k1 = params.get('k1', 0.1)
    k2 = params.get('k2', 0.2)
    k3 = params.get('k3', 2.0)
    f_heart = params.get('f_heart', 1.17)
    dt = params.get('dt', 0.01)
    n = params.get('step', 0)

    delta = k1 * (MAP - MAP_ss) / (MAP_ss + 1e-14) - k2 * (cbf - CBF_ss) / (CBF_ss + 1e-14)
    cardiac = -k3 * np.sin(2.0 * np.pi * f_heart * n * dt)
    cbf_new = cbf + delta + cardiac
    return max(cbf_new, 0.0)


def detect_hemodynamic_cycles(cbf_series, params):

    states = np.round(np.asarray(cbf_series, dtype=float) * 100).astype(int)

    if len(states) < 2:
        return None, None, []


    unique_states = list(sorted(set(states)))
    state_to_idx = {s: i for i, s in enumerate(unique_states)}
    n_unique = len(unique_states)


    transitions = {}
    for i in range(len(states) - 1):
        s = states[i]
        s_next = states[i + 1]
        if s not in transitions:
            transitions[s] = s_next

    def f(x):
        return transitions.get(x, x)

    if len(transitions) < 2:
        return None, None, states.tolist()

    x0 = states[0]
    try:
        lam, mu = cycle_brent(f, x0)
    except (RuntimeError, RecursionError):
        lam, mu = None, None

    return lam, mu, states.tolist()


def analyze_frequency_content(signal, dt):
    signal = np.asarray(signal, dtype=float)
    n = len(signal)
    if n < 2:
        return np.array([]), np.array([])
    fft_vals = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(n, d=dt)
    amps = np.abs(fft_vals) / n
    return freqs, amps


def classify_flow_regime(lam, mu, freqs, amps, dt):
    if freqs is None or len(freqs) == 0:
        return 'insufficient_data'


    if len(amps) > 1:
        dominant_idx = np.argmax(amps[1:]) + 1
        dominant_freq = freqs[dominant_idx]
        dominant_amp = amps[dominant_idx]
    else:
        dominant_freq = 0.0
        dominant_amp = 0.0

    if lam is None or mu is None:
        if 0.8 <= dominant_freq <= 1.5 and dominant_amp > 0.05 * np.max(amps):
            return 'normal_cardiac'
        return 'arrhythmic'

    period = lam * dt
    if 0.6 <= 1.0 / (period + 1e-14) <= 1.5:
        return 'normal_cardiac'
    elif 1.0 / (period + 1e-14) < 0.1:
        return 'pathological_slow_oscillation'
    elif 1.0 / (period + 1e-14) > 3.0:
        return 'pathological_fast_oscillation'
    return 'uncertain'
