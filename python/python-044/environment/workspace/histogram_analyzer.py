
import numpy as np


def histogramize(data, bin_min, bin_max, bin_num):
    data = np.asarray(data).flatten()
    if bin_num < 1:
        raise ValueError("bin_num must be positive.")
    if bin_max <= bin_min:
        raise ValueError("bin_max must be greater than bin_min.")

    bin_edges = np.linspace(bin_min, bin_max, bin_num + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    bin_counts = np.zeros(bin_num, dtype=int)
    for i in range(bin_num):
        if i < bin_num - 1:
            mask = (data >= bin_edges[i]) & (data < bin_edges[i + 1])
        else:
            mask = (data >= bin_edges[i]) & (data <= bin_edges[i + 1])
        bin_counts[i] = int(np.sum(mask))

    return bin_centers, bin_counts, bin_edges


def energy_spectrum_bins(pressure, displacement, material, bin_num=20):
    n = len(pressure)


    E_fluid = 0.5 * (1.0 / material.M) * pressure ** 2


    eps_mag = np.linalg.norm(displacement, axis=1)


    E_strain = 0.5 * (material.lam + 2.0 * material.mu) * eps_mag ** 2


    E_coupling = 0.5 * material.alpha * pressure * eps_mag

    E_total = E_fluid + E_strain + np.abs(E_coupling)

    e_min = float(np.min(E_total))
    e_max = float(np.max(E_total))
    if e_max <= e_min:
        e_max = e_min + 1.0

    centers, counts, edges = histogramize(E_total, e_min, e_max, bin_num)

    stats = {
        "energy_mean": float(np.mean(E_total)),
        "energy_std": float(np.std(E_total)),
        "energy_min": e_min,
        "energy_max": e_max,
        "energy_median": float(np.median(E_total)),
        "skewness": float(_compute_skewness(E_total)),
        "kurtosis": float(_compute_kurtosis(E_total)),
        "bin_centers": centers,
        "bin_counts": counts,
        "bin_edges": edges,
        "total_energy": float(np.sum(E_total)),
    }
    return stats


def _compute_skewness(x):
    x = np.asarray(x).flatten()
    n = len(x)
    if n < 3:
        return 0.0
    m = np.mean(x)
    s = np.std(x, ddof=1)
    if s < 1e-14:
        return 0.0
    return (np.sum((x - m) ** 3) / n) / (s ** 3)


def _compute_kurtosis(x):
    x = np.asarray(x).flatten()
    n = len(x)
    if n < 4:
        return 0.0
    m = np.mean(x)
    s = np.std(x, ddof=1)
    if s < 1e-14:
        return 0.0
    return (np.sum((x - m) ** 4) / n) / (s ** 4) - 3.0


def analyze_wave_front_histogram(pressure_history, time_array, bin_num=20):
    n_steps = pressure_history.shape[0]
    max_p = np.max(pressure_history, axis=1)
    min_p = np.min(pressure_history, axis=1)

    p_range = max(max_p.max(), abs(min_p.min()))
    if p_range < 1e-14:
        p_range = 1.0

    centers, counts, edges = histogramize(max_p, -p_range, p_range, bin_num)

    stats = {
        "time": time_array,
        "max_pressure": max_p,
        "min_pressure": min_p,
        "mean_pressure": np.mean(pressure_history, axis=1),
        "std_pressure": np.std(pressure_history, axis=1),
        "bin_centers": centers,
        "bin_counts": counts,
    }
    return stats
