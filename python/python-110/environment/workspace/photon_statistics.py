
import numpy as np
from typing import Tuple, Dict
from utils import validate_array_1d


def disk_unit_sample(n: int = 1) -> np.ndarray:
    theta = 2.0 * np.pi * np.random.rand(n)
    r = np.sqrt(np.random.rand(n))
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.vstack([x, y])


def disk_distance_stats(n_samples: int = 10000) -> Tuple[float, float]:
    distances = np.zeros(n_samples, dtype=float)
    for i in range(n_samples):
        p = disk_unit_sample(1).ravel()
        q = disk_unit_sample(1).ravel()
        distances[i] = np.linalg.norm(p - q)
    mu = float(np.mean(distances))
    if n_samples > 1:
        var = float(np.var(distances, ddof=1))
    else:
        var = 0.0
    return mu, var


def second_order_correlation_weak_coupling(
    tau: np.ndarray,
    gamma_dot: float,
    kappa: float,
    g_coupling: float,
) -> np.ndarray:
    tau = np.asarray(tau, dtype=float)
    if gamma_dot < 0 or kappa < 0:
        raise ValueError("Decay rates must be non-negative")
    Gamma = gamma_dot + kappa
    if Gamma < 1e-15:
        Gamma = 1e-15
    g2 = 1.0 - np.exp(-Gamma * np.abs(tau))

    if g_coupling > 0.01 * Gamma:
        Omega = np.sqrt(np.maximum(g_coupling ** 2 - 0.25 * Gamma ** 2, 0.0))
        if Omega > 1e-15:
            g2 += 0.2 * (g_coupling / Gamma) ** 2 * np.cos(Omega * tau) * np.exp(-0.5 * Gamma * np.abs(tau))

    g2 = np.maximum(g2, 0.0)
    return g2


def antibunching_parameter(g2_0: float) -> str:
    if g2_0 < 0.5:
        return "strong_antibunching"
    elif g2_0 < 1.0:
        return "weak_antibunching"
    else:
        return "bunching_or_poissonian"


def monte_carlo_photon_detection(
    emission_rate: float,
    detection_efficiency: float,
    gate_time: float,
    n_trials: int = 10000,
) -> Dict[str, float]:
    if emission_rate < 0 or detection_efficiency < 0 or gate_time < 0:
        raise ValueError("Physical parameters must be non-negative")
    if detection_efficiency > 1.0:
        detection_efficiency = 1.0
    mu = emission_rate * gate_time * detection_efficiency
    counts = np.random.poisson(mu, size=n_trials)
    mean_counts = float(np.mean(counts))
    var_counts = float(np.var(counts, ddof=1))
    p_zero = float(np.mean(counts == 0))
    p_single = float(np.mean(counts == 1))
    p_multi = float(np.mean(counts >= 2))
    return {
        "mean_counts": mean_counts,
        "variance": var_counts,
        "p_zero": p_zero,
        "p_single": p_single,
        "p_multi": p_multi,
        "singles_fraction": p_single / (1.0 - p_zero + 1e-15),
    }


def simulate_hanbury_brown_twiss(
    gamma_dot: float,
    kappa: float,
    g_coupling: float,
    measurement_time: float,
    n_bins: int = 200,
) -> Tuple[np.ndarray, np.ndarray]:
    if measurement_time <= 0:
        raise ValueError("measurement_time must be positive")
    tau_vals = np.linspace(-measurement_time, measurement_time, n_bins)
    g2_vals = second_order_correlation_weak_coupling(tau_vals, gamma_dot, kappa, g_coupling)

    noise = np.random.normal(0.0, 0.02 * np.max(g2_vals), size=n_bins)
    counts = np.maximum(g2_vals + noise, 0.0)
    return tau_vals, counts


def photon_indistinguishability_homodyne(
    pure_state_overlap: float,
    dephasing_rate: float,
    pulse_duration: float,
) -> float:
    if pure_state_overlap < 0 or pure_state_overlap > 1:
        raise ValueError("Overlap must be in [0, 1]")
    if dephasing_rate < 0 or pulse_duration < 0:
        raise ValueError("Rates and durations must be non-negative")
    V = (pure_state_overlap ** 2) * np.exp(-2.0 * dephasing_rate * pulse_duration)
    return float(V)


def detection_area_efficiency(
    detector_radius: float,
    emission_waist: float,
    distance: float,
) -> float:
    if detector_radius <= 0 or emission_waist <= 0 or distance < 0:
        raise ValueError("Geometric parameters must be positive")

    beam_radius = emission_waist * np.sqrt(1.0 + (distance / (np.pi * emission_waist ** 2 / 780e-9)) ** 2)
    eta = (detector_radius ** 2) / (beam_radius ** 2)
    return float(min(eta, 1.0))
