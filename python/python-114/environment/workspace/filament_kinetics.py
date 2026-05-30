
import numpy as np


def henon_filament_map(theta: float, kappa: float,
                       c: float = 0.95, a: float = 0.3) -> tuple:
    if c < -1.0 or c > 1.0:
        raise ValueError("c must be in [-1, 1]")
    s = np.sqrt(max(1.0 - c * c, 0.0))

    theta_new = theta * c - (kappa - a * theta ** 2) * s
    kappa_new = theta * s + (kappa - a * theta ** 2) * c


    theta_new = np.clip(theta_new, 0.0, 1.0)
    return theta_new, kappa_new


def simulate_filament_assembly(num_monomers: int = 50,
                               num_realizations: int = 100,
                               c: float = 0.95,
                               a: float = 0.3,
                               theta_init: float = 0.05) -> dict:
    if num_monomers <= 0 or num_realizations <= 0:
        raise ValueError("Counts must be positive")

    theta_traj = np.zeros((num_realizations, num_monomers), dtype=float)
    kappa_traj = np.zeros((num_realizations, num_monomers), dtype=float)

    for r in range(num_realizations):
        theta = theta_init + 0.02 * np.random.randn()
        theta = np.clip(theta, 0.0, 1.0)
        kappa = 0.1 + 0.05 * np.random.randn()

        theta_traj[r, 0] = theta
        kappa_traj[r, 0] = kappa

        for n in range(1, num_monomers):
            theta, kappa = henon_filament_map(theta, kappa, c, a)
            theta_traj[r, n] = theta
            kappa_traj[r, n] = kappa

    return {
        'theta_mean': np.mean(theta_traj, axis=0),
        'theta_std': np.std(theta_traj, axis=0),
        'kappa_mean': np.mean(kappa_traj, axis=0),
        'kappa_std': np.std(kappa_traj, axis=0),
        'coverage_final': np.mean(theta_traj[:, -1]),
        'cooperativity': a
    }


def compute_filament_stability_energy(coverage: float,
                                       bending_modulus: float = 200.0,
                                       binding_energy_per_monomer: float = -35.0) -> float:
    if coverage < 0 or coverage > 1:
        raise ValueError("coverage must be in [0, 1]")
    if bending_modulus < 0:
        raise ValueError("bending modulus must be non-negative")

    kappa_0 = 1.0
    kappa = kappa_0 * (1.0 - coverage)
    e_bend = 0.5 * bending_modulus * kappa ** 2
    e_bind = coverage * binding_energy_per_monomer
    return e_bind + e_bend


def cooperativity_index_from_trajectory(theta_mean: np.ndarray) -> float:
    if len(theta_mean) < 2:
        return 1.0

    idx_10 = -1
    idx_90 = -1
    for i, th in enumerate(theta_mean):
        if idx_10 < 0 and th >= 0.1:
            idx_10 = i
        if th >= 0.9:
            idx_90 = i
            break

    if idx_10 < 0 or idx_90 < 0 or idx_90 == idx_10:
        return 1.0


    ratio = (idx_90 - idx_10) / len(theta_mean)
    n_hill = max(1.0, 5.0 * (1.0 - ratio))
    return n_hill


def logistic_growth_model(t: np.ndarray, K: float = 1.0,
                          r: float = 0.1, t0: float = 50.0) -> np.ndarray:
    t = np.asarray(t, dtype=float)
    return K / (1.0 + np.exp(-r * (t - t0)))
