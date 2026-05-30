
import numpy as np
from typing import Tuple
from potential_models import total_potential_lj


def heat_capacity_cv(energies: np.ndarray, temperature: float,
                     n_particles: int, dim: int = 2,
                     k_boltzmann: float = 1.0) -> Tuple[float, float]:


    raise NotImplementedError("Hole_3: 请补全 heat_capacity_cv 的热力学公式实现")


def elastic_constants_from_fluctuations(stress_history: np.ndarray,
                                        volume: float,
                                        temperature: float,
                                        k_boltzmann: float = 1.0) -> np.ndarray:
    if len(stress_history) < 2:
        return np.eye(2)



    if stress_history.ndim == 1:

        mean_p = np.mean(stress_history)
        var_p = np.var(stress_history, ddof=1)

        bulk_modulus = volume * var_p / (k_boltzmann * temperature) if temperature > 1e-12 else 0.0
        C = np.array([[bulk_modulus, 0.0],
                      [0.0, bulk_modulus]])
    else:

        n_steps = stress_history.shape[0]
        sigma = np.mean(stress_history, axis=0)

        delta_sigma = stress_history - sigma

        C = np.zeros((2, 2, 2, 2))
        for alpha in range(2):
            for beta in range(2):
                for gamma in range(2):
                    for delta in range(2):
                        cov = np.mean(delta_sigma[:, alpha, beta] *
                                       delta_sigma[:, gamma, delta])
                        if temperature > 1e-12:
                            C[alpha, beta, gamma, delta] = cov * volume / (k_boltzmann * temperature)

        C_voigt = np.zeros((2, 2))
        C_voigt[0, 0] = C[0, 0, 0, 0]
        C_voigt[1, 1] = C[1, 1, 1, 1]
        C_voigt[0, 1] = C[0, 0, 1, 1]
        C_voigt[1, 0] = C_voigt[0, 1]
        C = C_voigt
    return C


def elastic_constants_strain_derivative(positions: np.ndarray,
                                         epsilon: float = 1.0,
                                         sigma: float = 1.0,
                                         rcut: float = 2.5,
                                         volume: float = 1.0,
                                         strain_perturbation: float = 1e-4) -> np.ndarray:
    n, d = positions.shape
    C = np.zeros((d, d))
    eps = strain_perturbation


    e0 = total_potential_lj(positions, epsilon, sigma, rcut)

    for alpha in range(d):

        pos_plus = positions.copy()
        pos_minus = positions.copy()
        for i in range(n):
            pos_plus[i, alpha] *= (1.0 + eps)
            pos_minus[i, alpha] *= (1.0 - eps)

        e_plus = total_potential_lj(pos_plus, epsilon, sigma, rcut)
        e_minus = total_potential_lj(pos_minus, epsilon, sigma, rcut)

        C[alpha, alpha] = (e_plus - 2.0 * e0 + e_minus) / (eps ** 2 * volume)

    return C


def radial_distribution_function(positions: np.ndarray,
                                  box_size: float,
                                  n_bins: int = 50,
                                  rcut: float = None) -> Tuple[np.ndarray, np.ndarray]:
    n, d = positions.shape
    if rcut is None:
        rcut = box_size / 2.0

    dr = rcut / n_bins
    r_bins = np.linspace(0.5 * dr, rcut - 0.5 * dr, n_bins)
    hist = np.zeros(n_bins)

    for i in range(n):
        for j in range(i + 1, n):
            rij = positions[j] - positions[i]

            rij -= box_size * np.round(rij / box_size)
            r = np.linalg.norm(rij)
            if r < rcut:
                bin_idx = int(r / dr)
                if 0 <= bin_idx < n_bins:
                    hist[bin_idx] += 2.0


    volume = box_size ** d
    shell_volumes = np.zeros(n_bins)
    for k in range(n_bins):
        r_inner = k * dr
        r_outer = (k + 1) * dr
        if d == 2:
            shell_volumes[k] = np.pi * (r_outer ** 2 - r_inner ** 2)
        elif d == 3:
            shell_volumes[k] = (4.0 / 3.0) * np.pi * (r_outer ** 3 - r_inner ** 3)
        else:
            shell_volumes[k] = 2.0 * (r_outer - r_inner)


    norm = volume / (n * (n - 1)) if n > 1 else 1.0
    g_r = norm * hist / (shell_volumes + 1e-30)
    return r_bins, g_r


def thermal_expansion_estimate(lengths: np.ndarray,
                                temperatures: np.ndarray) -> float:
    if len(lengths) < 2 or len(temperatures) < 2:
        return 0.0

    T_mean = np.mean(temperatures)
    L_mean = np.mean(lengths)
    numerator = np.sum((temperatures - T_mean) * (lengths - L_mean))
    denominator = np.sum((temperatures - T_mean) ** 2)
    if abs(denominator) < 1e-30:
        return 0.0
    dL_dT = numerator / denominator
    L0 = L_mean - dL_dT * T_mean
    if abs(L0) < 1e-30:
        L0 = 1e-30
    return dL_dT / L0


def entropy_from_energy_distribution(energies: np.ndarray,
                                      temperature: float,
                                      n_bins: int = 20,
                                      k_boltzmann: float = 1.0) -> float:
    if len(energies) < 2:
        return 0.0
    e_min, e_max = np.min(energies), np.max(energies)
    if abs(e_max - e_min) < 1e-30:
        return 0.0

    counts, _ = np.histogram(energies, bins=n_bins, range=(e_min, e_max))
    probs = counts / len(energies)

    probs = probs[probs > 1e-30]
    entropy = -k_boltzmann * np.sum(probs * np.log(probs))
    return entropy
