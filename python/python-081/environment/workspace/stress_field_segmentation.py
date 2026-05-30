
import numpy as np
from typing import Tuple, List


def single_threshold_segmentation(sigma_vm: np.ndarray,
                                   threshold: float) -> np.ndarray:
    labels = np.where(sigma_vm > threshold, 1, 0)
    return labels


def dual_threshold_segmentation(sigma_vm: np.ndarray,
                                 theta_low: float,
                                 theta_high: float) -> np.ndarray:
    labels = np.zeros(len(sigma_vm), dtype=np.int32)
    labels[(sigma_vm > theta_low) & (sigma_vm <= theta_high)] = 1
    labels[sigma_vm > theta_high] = 2
    return labels


def otsu_threshold(sigma_vm: np.ndarray, n_bins: int = 256) -> float:
    if len(sigma_vm) == 0:
        return 0.0
    data = sigma_vm.flatten()
    vmin, vmax = np.min(data), np.max(data)
    if vmax - vmin < 1e-12:
        return vmin


    bins = np.linspace(vmin, vmax, n_bins + 1)
    hist, _ = np.histogram(data, bins=bins)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    total = len(data)
    total_mean = np.mean(data)
    max_variance = -1.0
    optimal_threshold = vmin

    omega_0 = 0
    sum_0 = 0.0
    for i in range(n_bins):
        omega_0 += hist[i]
        sum_0 += hist[i] * bin_centers[i]
        if omega_0 == 0 or omega_0 == total:
            continue
        omega_1 = total - omega_0
        mu_0 = sum_0 / omega_0
        mu_1 = (total * total_mean - sum_0) / omega_1
        variance = (omega_0 * omega_1 / (total ** 2)) * (mu_0 - mu_1) ** 2
        if variance > max_variance:
            max_variance = variance
            optimal_threshold = bin_centers[i]

    return float(optimal_threshold)


def compute_damage_zone_statistics(sigma_vm: np.ndarray,
                                    labels: np.ndarray) -> dict:
    unique_labels = np.unique(labels)
    stats = {}
    for lbl in unique_labels:
        mask = labels == lbl
        zone_stresses = sigma_vm[mask]
        stats[f"zone_{lbl}"] = {
            "count": int(np.sum(mask)),
            "mean_stress": float(np.mean(zone_stresses)) if len(zone_stresses) > 0 else 0.0,
            "max_stress": float(np.max(zone_stresses)) if len(zone_stresses) > 0 else 0.0,
            "min_stress": float(np.min(zone_stresses)) if len(zone_stresses) > 0 else 0.0,
        }
    return stats


def identify_critical_elements(sigma_vm: np.ndarray,
                                elements: np.ndarray,
                                top_percentile: float = 5.0) -> np.ndarray:
    threshold = np.percentile(sigma_vm, 100.0 - top_percentile)
    critical = np.where(sigma_vm >= threshold)[0]
    return critical
