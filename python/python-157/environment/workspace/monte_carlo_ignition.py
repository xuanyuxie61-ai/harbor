import numpy as np
from combustion_utils import (
    check_positive, check_nonnegative, cholesky_factor, solve_lower_triangular,
    arrhenius_rate, R_UNIVERSAL, DEFAULT_E_A, DEFAULT_A_PRE, DEFAULT_T_IGN
)


def sample_ellipse(n, A_mat, r):
    check_positive(n, "n")
    check_positive(r, "r")
    A_mat = np.asarray(A_mat, dtype=float)
    if A_mat.shape != (2, 2):
        raise ValueError("A_mat must be 2x2")

    L = cholesky_factor(A_mat)


    samples = []
    max_trials = n * 20
    trial = 0
    while len(samples) < n and trial < max_trials:
        trial += 1
        y = np.random.uniform(-1.0, 1.0, size=2)
        if np.dot(y, y) <= 1.0:
            samples.append(y)
    if len(samples) < n:

        while len(samples) < n:
            angle = np.random.uniform(0.0, 2.0 * np.pi)
            rad = np.sqrt(np.random.uniform(0.0, 1.0))
            samples.append([rad * np.cos(angle), rad * np.sin(angle)])

    Y = np.array(samples[:n]).T
    Y = r * Y
    X = np.zeros_like(Y)
    for j in range(n):
        X[:, j] = solve_lower_triangular(L, Y[:, j])
    return X.T


def ignition_probability_monte_carlo(n_samples, T_mean, T_std, p_mean, p_std,
                                     phi_mean, phi_std, Ea=DEFAULT_E_A,
                                     A=DEFAULT_A_PRE, T_ign=DEFAULT_T_IGN,
                                     n_batches=5):
    check_positive(n_samples, "n_samples")
    check_positive(n_batches, "n_batches")

    k_ign = A * np.exp(-Ea / (R_UNIVERSAL * max(T_ign, 1.0e-6)))
    batch_size = n_samples // n_batches

    batch_probs = []
    for b in range(n_batches):
        count = 0
        for _ in range(batch_size):

            T = max(np.random.normal(T_mean, T_std), 200.0)
            p = max(np.random.normal(p_mean, p_std), 1000.0)
            phi = max(np.random.normal(phi_mean, phi_std), 0.1)

            phi_eff = np.exp(-0.5 * ((phi - 1.0) / 0.3) ** 2)
            k = A * np.exp(-Ea / (R_UNIVERSAL * T)) * phi_eff
            if k > k_ign:
                count += 1
        prob = count / batch_size
        batch_probs.append(prob)

    mean_prob = np.mean(batch_probs)
    std_prob = np.std(batch_probs, ddof=1)
    return mean_prob, std_prob, batch_probs


def critical_kernel_escape_time(n_grid, it_max, D_wave, gamma, Q,
                                rho0, p0, x_range=(-0.5, 0.5),
                                y_range=(-0.5, 0.5)):
    check_positive(n_grid, "n_grid")
    check_positive(it_max, "it_max")
    x_min, x_max = x_range
    y_min, y_max = y_range

    X = np.linspace(x_min, x_max, n_grid)
    Y = np.linspace(y_min, y_max, n_grid)
    escape_times = np.zeros((n_grid, n_grid), dtype=int)


    from combustion_utils import cj_detonation_velocity
    D_cj = cj_detonation_velocity(gamma, Q, p0, rho0)
    T_ref = p0 / (rho0 * (R_UNIVERSAL / 0.029))
    T_critical = 2.0 * T_ref

    for i in range(n_grid):
        for j in range(n_grid):

            z_real = X[i]
            z_imag = Y[j]
            T_local = T_ref + 50.0 * z_real

            for it in range(1, it_max + 1):

                k_val = DEFAULT_A_PRE * np.exp(-DEFAULT_E_A / (R_UNIVERSAL * max(T_local, 100.0)))
                T_local += 1.0e-7 * Q * k_val
                if T_local > T_critical:
                    escape_times[j, i] = it
                    break
            if escape_times[j, i] == 0:
                escape_times[j, i] = it_max + 1

    ignited = escape_times <= it_max
    area_fraction = np.sum(ignited) / (n_grid * n_grid)
    avg_escape_time = np.mean(escape_times[ignited]) if np.any(ignited) else it_max
    return area_fraction, avg_escape_time, escape_times
