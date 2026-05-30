import numpy as np


def add_gaussian_noise(data, alpha=0.05, beta=0.5):
    data = np.asarray(data, dtype=float)
    sigma = alpha * np.abs(data) + beta
    noise = np.random.normal(0.0, sigma)
    return data + noise


def add_spike_noise(data, level=0.02, magnitude=10.0):
    data = np.asarray(data, dtype=float)
    mask = np.random.rand(*data.shape) < level
    noise = np.random.normal(0.0, magnitude, size=data.shape)
    result = data.copy()
    result[mask] += noise[mask]
    return result


def simple_kalman_update(x_forecast, P_forecast, y_obs, H, R):
    x_f = float(x_forecast)
    P_f = float(P_forecast)
    y = float(y_obs)
    h = float(H)
    r = float(R)
    S = h ** 2 * P_f + r
    if abs(S) < 1e-14:
        S = 1e-14
    K = h * P_f / S
    x_a = x_f + K * (y - h * x_f)
    P_a = (1.0 - K * h) * P_f
    return x_a, max(P_a, 0.0)


def ensemble_assimilation(ensemble, observations, obs_variance,
                          inflation_factor=1.02):
    ens = np.asarray(ensemble, dtype=float)
    n = len(ens)
    x_mean = np.mean(ens)
    P = np.var(ens) * inflation_factor

    for y in observations:
        for i in range(n):
            ens[i], _ = simple_kalman_update(ens[i], P, y, 1.0, obs_variance)
        x_mean = np.mean(ens)
        P = np.var(ens)
    return ens
