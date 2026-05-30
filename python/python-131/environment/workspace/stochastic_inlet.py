
import numpy as np






def generate_inlet_conditions(n_samples, T_mean=523.0, T_std=5.0,
                              yCO_mean=0.30, yH2_mean=0.60, y_std=0.02,
                              Q_mean=0.01, Q_std=0.001,
                              seed=42):
    rng = np.random.default_rng(seed)

    T = T_mean + T_std * rng.standard_normal(n_samples)
    yCO_raw = yCO_mean + y_std * rng.standard_normal(n_samples)
    yH2_raw = yH2_mean + y_std * rng.standard_normal(n_samples)
    Q = Q_mean + Q_std * rng.standard_normal(n_samples)


    y_total = yCO_raw + yH2_raw
    mask = y_total > 0.95
    yCO_raw[mask] = yCO_raw[mask] * 0.95 / y_total[mask]
    yH2_raw[mask] = yH2_raw[mask] * 0.95 / y_total[mask]

    yCO = np.clip(yCO_raw, 0.05, 0.50)
    yH2 = np.clip(yH2_raw, 0.10, 0.70)

    Q = np.clip(Q, 0.5 * Q_mean, 1.5 * Q_mean)
    T = np.clip(T, T_mean - 3 * T_std, T_mean + 3 * T_std)

    time = np.arange(n_samples, dtype=float)

    return {
        'T': T,
        'yCO': yCO,
        'yH2': yH2,
        'Q': Q,
        'time': time,
        'statistics': {
            'T_mean': float(np.mean(T)),
            'T_std': float(np.std(T)),
            'yCO_mean': float(np.mean(yCO)),
            'yH2_mean': float(np.mean(yH2)),
            'Q_mean': float(np.mean(Q)),
            'Q_cv': float(np.std(Q) / np.mean(Q)),
        }
    }


def generate_perturbed_profile(base_profile, mu_perturb=0.0, sigma_perturb=0.05,
                               seed=42):
    rng = np.random.default_rng(seed)
    base = np.asarray(base_profile, dtype=float)
    noise = rng.normal(mu_perturb, sigma_perturb, size=base.shape)
    perturbed = base * (1.0 + noise)
    return np.clip(perturbed, 0.0, None)
