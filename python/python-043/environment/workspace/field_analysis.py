
import numpy as np
from typing import List, Tuple, Dict






def svd_field_modes(coeffs_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    coeffs_matrix = np.asarray(coeffs_matrix, dtype=float)

    mean = np.mean(coeffs_matrix, axis=0)
    Anom = coeffs_matrix - mean[np.newaxis, :]
    U, S, Vt = np.linalg.svd(Anom, full_matrices=False)
    return U, S, Vt





def extract_dipole_parameters(coeffs: Dict[Tuple[int, int], complex]) -> Dict[str, float]:









    raise NotImplementedError("Hole_3: 偶极子参数提取待实现")

    return {
        "g10": 0.0,
        "g11": 0.0,
        "h11": 0.0,
        "inclination": 0.0,
        "declination": 0.0,
        "dipole_moment_norm": 0.0,
        "dipole_tilt": 0.0,
    }






def detect_reversals(g10_series: np.ndarray, time_series: np.ndarray,
                     threshold_ratio: float = 0.3) -> List[Dict[str, float]]:
    g10 = np.asarray(g10_series, dtype=float)
    t = np.asarray(time_series, dtype=float)
    n = len(g10)
    if n < 3:
        return []


    g10_smooth = g10.copy()
    for i in range(1, n - 1):
        g10_smooth[i] = (g10[i - 1] + g10[i] + g10[i + 1]) / 3.0

    max_amp = np.max(np.abs(g10_smooth))
    threshold = threshold_ratio * max_amp
    if threshold < 1e-30:
        return []

    reversals = []
    in_reversal = False
    reversal_start = 0.0
    polarity_before = 0

    for i in range(1, n):
        prev = g10_smooth[i - 1]
        curr = g10_smooth[i]

        if prev * curr < 0:
            if not in_reversal:

                if abs(prev) >= threshold:
                    in_reversal = True
                    reversal_start = t[i - 1]
                    polarity_before = 1 if prev > 0 else -1
            else:

                if abs(curr) >= threshold:
                    reversal_end = t[i]
                    duration = reversal_end - reversal_start
                    polarity_after = 1 if curr > 0 else -1
                    reversals.append({
                        "time_start": reversal_start,
                        "time_end": reversal_end,
                        "duration": duration,
                        "polarity_before": polarity_before,
                        "polarity_after": polarity_after,
                    })
                    in_reversal = False

    return reversals





def magnetic_energy_spectrum(coeffs_time: List[Dict[Tuple[int, int], complex]],
                              l_max: int) -> Tuple[np.ndarray, np.ndarray]:
    n_times = len(coeffs_time)
    if n_times == 0:
        return np.array([]), np.array([])

    E_all = np.zeros((n_times, l_max + 1), dtype=float)
    for it, coeffs in enumerate(coeffs_time):
        for l in range(l_max + 1):
            energy = 0.0
            for m in range(-l, l + 1):
                c = coeffs.get((l, m), 0.0)
                energy += abs(c) ** 2
            E_all[it, l] = (l + 1.0) * energy

    E_mean = np.mean(E_all, axis=0)
    return np.arange(l_max + 1), E_mean





def reversal_statistics(reversals: List[Dict[str, float]],
                        total_time: float) -> Dict[str, float]:
    if not reversals:
        return {
            "reversal_rate": 0.0,
            "mean_duration": 0.0,
            "std_duration": 0.0,
            "total_reversal_time_ratio": 0.0,
        }

    durations = np.array([ev["duration"] for ev in reversals], dtype=float)
    total_rev_time = np.sum(durations)
    myr = 1e6 * 365.25 * 24 * 3600.0

    return {
        "reversal_rate": len(reversals) / (total_time / myr),
        "mean_duration": float(np.mean(durations)),
        "std_duration": float(np.std(durations)),
        "total_reversal_time_ratio": total_rev_time / total_time,
    }





def generate_field_report(coeffs_history: List[Dict[Tuple[int, int], complex]],
                          time_history: np.ndarray,
                          l_max: int) -> Dict[str, any]:
    g10_series = np.array([extract_dipole_parameters(c)["g10"] for c in coeffs_history])
    reversals = detect_reversals(g10_series, time_history)
    stats = reversal_statistics(reversals, time_history[-1] - time_history[0])
    l_vals, E_mean = magnetic_energy_spectrum(coeffs_history, l_max)


    latest_dipole = extract_dipole_parameters(coeffs_history[-1])

    return {
        "dipole_latest": latest_dipole,
        "reversals": reversals,
        "statistics": stats,
        "energy_spectrum_l": l_vals,
        "energy_spectrum_E": E_mean,
    }





def _self_test():

    t = np.linspace(0.0, 1e6 * 365.25 * 24 * 3600, 1000)
    g10 = np.sin(2.0 * np.pi * t / (2.5e5 * 365.25 * 24 * 3600))
    revs = detect_reversals(g10, t)
    assert len(revs) >= 3


    coeffs = {(1, 0): 1.0 + 0.0j, (1, 1): 0.5 + 0.3j, (2, 0): 0.1 + 0.0j}
    dp = extract_dipole_parameters(coeffs)
    assert abs(dp["g10"] - 1.0) < 1e-10
    assert dp["dipole_moment_norm"] > 0.0


    mat = np.random.randn(20, 5)
    U, S, Vt = svd_field_modes(mat)
    assert len(S) <= 5

    print("field_analysis: self-test passed.")


if __name__ == "__main__":
    _self_test()
