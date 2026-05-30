
import numpy as np
from typing import Tuple, List, Dict






def modulation_curve(
    t_days: np.ndarray,
    s0: float,
    sm: float,
    t0: float = 152.0,
    period: float = 365.25,
) -> np.ndarray:
    return s0 + sm * np.cos(2.0 * np.pi * (t_days - t0) / period)


def modulation_curve_lissajous(
    t_days: np.ndarray,
    s0: float,
    sm: float,
    t0: float = 152.0,
    period: float = 365.25,
    phase_shift: float = np.pi / 2.0,
) -> Tuple[np.ndarray, np.ndarray]:
    omega = 2.0 * np.pi / period
    phase = omega * (t_days - t0)
    X = s0 + sm * np.cos(phase)
    Y = s0 + sm * np.cos(phase + phase_shift)
    return X, Y






def bin_events_by_time(
    events: List[Dict],
    n_bins: int = 12,
    period: float = 365.25,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    counts = np.zeros(n_bins)
    bin_edges = np.linspace(0.0, period, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    for ev in events:
        t = ev.get("time_day", 0.0) % period
        idx = int(np.clip(np.floor(t / period * n_bins), 0, n_bins - 1))
        counts[idx] += 1.0

    errors = np.sqrt(np.where(counts > 0.0, counts, 1.0))
    return bin_centers, counts, errors


def fit_modulation_amplitude(
    t_bins: np.ndarray,
    counts: np.ndarray,
    errors: np.ndarray,
    period: float = 365.25,
) -> Tuple[float, float, float, float]:
    t_bins = np.asarray(t_bins)
    counts = np.asarray(counts)
    errors = np.asarray(errors)


    t0 = 152.0
    theta = 2.0 * np.pi * (t_bins - t0) / period


    w = 1.0 / np.where(errors > 0.0, errors, 1.0)
    A_mat = np.column_stack([np.ones_like(t_bins), np.cos(theta), np.sin(theta)])
    Aw = A_mat * w[:, None]
    cw = counts * w


    coeffs, residuals, rank, s = np.linalg.lstsq(Aw, cw, rcond=None)
    s0_fit = coeffs[0]
    A_fit = coeffs[1]
    B_fit = coeffs[2]

    sm_fit = np.sqrt(A_fit ** 2 + B_fit ** 2)
    phase_fit = np.arctan2(B_fit, A_fit)


    model = s0_fit + A_fit * np.cos(theta) + B_fit * np.sin(theta)
    chi2 = np.sum(((counts - model) / errors) ** 2)

    return float(s0_fit), float(sm_fit), float(phase_fit), float(chi2)


def modulation_significance(
    s0: float,
    sm: float,
    total_counts: float,
    n_bins: int = 12,
) -> float:
    if s0 <= 0.0 or total_counts <= 0.0:
        return 0.0
    modulation_fraction = sm / s0
    sigma = modulation_fraction * np.sqrt(total_counts)
    return float(sigma)






def analyze_modulation_by_energy_bins(
    events: List[Dict],
    energy_edges: np.ndarray,
    n_time_bins: int = 12,
) -> List[Dict]:
    results = []
    n_ebins = len(energy_edges) - 1

    for i in range(n_ebins):
        e_low = energy_edges[i]
        e_high = energy_edges[i + 1]
        selected = [ev for ev in events if e_low <= ev.get("energy_obs", 0.0) < e_high]

        if len(selected) < n_time_bins * 2:
            results.append({
                "energy_low": e_low,
                "energy_high": e_high,
                "n_events": len(selected),
                "s0": None,
                "sm": None,
                "significance": 0.0,
            })
            continue

        t_bins, counts, errors = bin_events_by_time(selected, n_time_bins)
        s0, sm, phase, chi2 = fit_modulation_amplitude(t_bins, counts, errors)
        sig = modulation_significance(s0, sm, float(len(selected)), n_time_bins)

        results.append({
            "energy_low": e_low,
            "energy_high": e_high,
            "n_events": len(selected),
            "s0": s0,
            "sm": sm,
            "phase": phase,
            "chi2": chi2,
            "significance": sig,
        })

    return results






if __name__ == "__main__":

    t = np.linspace(0.0, 365.25, 100)
    s = modulation_curve(t, s0=100.0, sm=5.0)
    assert abs(np.mean(s) - 100.0) < 0.1, "调制曲线平均值异常"
    assert abs(np.max(s) - 105.0) < 0.1, "调制曲线最大值异常"
    assert abs(np.min(s) - 95.0) < 0.1, "调制曲线最小值异常"


    X, Y = modulation_curve_lissajous(t, 100.0, 5.0)

    assert abs(X[0] - X[-1]) < 1e-10, "Lissajous 曲线未闭合"


    np.random.seed(0)
    true_s0, true_sm = 100.0, 5.0
    t_bins = np.linspace(15.0, 350.0, 12)
    counts = modulation_curve(t_bins, true_s0, true_sm) + np.random.normal(0.0, 3.0, size=12)
    errors = np.sqrt(np.where(counts > 0, counts, 1.0))
    s0_fit, sm_fit, phase_fit, chi2 = fit_modulation_amplitude(t_bins, counts, errors)
    assert abs(s0_fit - true_s0) / true_s0 < 0.1, "S0 拟合偏差过大"
    assert abs(sm_fit - true_sm) / true_sm < 0.3, "Sm 拟合偏差过大"


    sig = modulation_significance(100.0, 5.0, 10000.0)
    assert sig > 0.0, "显著性应为正"

    print("annual_modulation.py: 所有自测通过")
