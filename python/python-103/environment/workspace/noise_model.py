
import numpy as np


def brownian_motion_simulation(m, n_steps, d, t_total, seed=None):
    if seed is not None:
        np.random.seed(seed)
    if n_steps < 2:
        raise ValueError("brownian_motion_simulation: n_steps must be >= 2")
    if t_total <= 0 or d < 0:
        raise ValueError("brownian_motion_simulation: invalid physical parameters")

    dt = t_total / (n_steps - 1)
    x = np.zeros((m, n_steps))

    s = np.sqrt(2.0 * m * d * dt)
    if m == 1:
        dx = s * np.random.randn(1, n_steps - 1)
    else:
        a = np.random.randn(m, n_steps - 1)
        norms = np.sqrt(np.sum(a ** 2, axis=0))
        norms = np.where(norms < 1e-15, 1.0, norms)
        v = s / norms
        dx = a * v[None, :]

    x[:, 1:] = np.cumsum(dx, axis=1)
    return x


def generate_ase_noise(t_grid, n_sp, G, h_nu, bw, seed=None):
    if t_grid.size < 2:
        raise ValueError("generate_ase_noise: invalid time grid")
    if G <= 1.0:
        return np.zeros_like(t_grid, dtype=complex)

    if seed is not None:
        np.random.seed(seed)

    dt = t_grid[1] - t_grid[0]

    psd = n_sp * (G - 1.0) * h_nu

    P_noise = psd * bw

    sigma = np.sqrt(P_noise * dt)


    noise = sigma * (np.random.randn(t_grid.size) + 1j * np.random.randn(t_grid.size))
    return noise


def bose_einstein_distribution(n_avg, n_max=50):
    if n_avg <= 0:
        raise ValueError("bose_einstein_distribution: n_avg must be positive")

    n = np.arange(n_max + 1)

    log_probs = n * np.log(n_avg) - (n + 1) * np.log(n_avg + 1.0)
    probs = np.exp(log_probs)
    probs = probs / np.sum(probs)
    return probs, n


def photon_number_fluctuation(t_grid, pulse_power, wavelength=1550e-9, n_avg_per_mode=1.0):
    if t_grid.size < 2 or pulse_power.size != t_grid.size:
        return np.zeros_like(t_grid)

    h = 6.62607015e-34
    c = 2.99792458e8
    nu = c / wavelength
    dt = t_grid[1] - t_grid[0]


    n_photon = pulse_power * dt / (h * nu)

    var_photon = n_photon * (1.0 + n_photon / n_avg_per_mode)

    rel_fluct = np.sqrt(var_photon) / np.maximum(n_photon, 1.0)
    rel_fluct = np.clip(rel_fluct, 0.0, 1.0)

    return rel_fluct * pulse_power


def parrondo_inspired_noise_coupling(t_grid, A_signal, epsilon=0.005):
    if t_grid.size < 2 or A_signal.size != t_grid.size:
        return A_signal.copy()

    dt = t_grid[1] - t_grid[0]
    n = t_grid.size
    A_out = A_signal.copy()


    for step in range(min(20, n // 2)):

        phase_A = np.sqrt(dt) * (np.random.randn(n) * (0.5 - epsilon))
        A_out *= np.exp(1j * phase_A)


        power = np.abs(A_out) ** 2
        median_power = np.median(power)

        if median_power > 1e-30:
            state = (power > median_power).astype(float)
        else:
            state = np.zeros_like(power)

        phase_B_low = np.sqrt(dt) * (np.random.randn(n) * (0.1 - epsilon))
        phase_B_high = np.sqrt(dt) * (np.random.randn(n) * (0.75 - epsilon))
        phase_B = state * phase_B_high + (1.0 - state) * phase_B_low
        A_out *= np.exp(1j * phase_B)

    return A_out
