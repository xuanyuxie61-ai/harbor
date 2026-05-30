
import numpy as np
from typing import Tuple, Optional


def cube_distance_pdf_exact(d: np.ndarray) -> np.ndarray:
    d = np.asarray(d, dtype=float)
    pdf = np.zeros_like(d)
    sqrt2 = np.sqrt(2.0)
    sqrt3 = np.sqrt(3.0)


    mask1 = (d >= 0.0) & (d <= 1.0)
    dm = d[mask1]
    pdf[mask1] = (-dm ** 2) * ((dm - 8.0) * dm ** 2 + np.pi * (6.0 * dm - 4.0))


    mask2 = (d > 1.0) & (d <= sqrt2)
    dm = d[mask2]
    t = dm ** 2 - 1.0
    t = np.maximum(t, 0.0)
    sqrt_term = np.sqrt(t)
    asec_dm = np.arccos(np.clip(1.0 / dm, -1.0, 1.0))
    pdf[mask2] = 2.0 * dm * (
        (dm ** 2 - 8.0 * sqrt_term + 3.0) * dm ** 2
        - 4.0 * sqrt_term
        + 12.0 * dm ** 2 * asec_dm
        + np.pi * (3.0 - 4.0 * dm) - 0.5
    )


    mask3 = (d > sqrt2) & (d <= sqrt3)
    dm = d[mask3]
    t2 = dm ** 2 - 2.0
    t2 = np.maximum(t2, 0.0)
    sqrt_term2 = np.sqrt(t2)

    arg_acsc = np.sqrt(np.maximum(2.0 - 2.0 / (dm ** 2), 0.0))
    arg_acsc = np.clip(arg_acsc, 1e-12, 1e12)
    acsc_val = np.arcsin(np.clip(1.0 / arg_acsc, -1.0, 1.0))
    atan1 = np.arctan(dm * sqrt_term2)
    atan2 = np.arctan(sqrt_term2)
    pdf[mask3] = dm * (
        (1.0 + dm ** 2) * (6.0 * np.pi + 8.0 * sqrt_term2 - 5.0 - dm ** 2)
        - 16.0 * dm * acsc_val
        + 16.0 * dm * atan1
        - 24.0 * (dm ** 2 + 1.0) * atan2
    )

    pdf = np.maximum(pdf, 0.0)
    return pdf


def cube_distance_stats_monte_carlo(n_samples: int = 10000,
                                    seed: Optional[int] = None) -> Tuple[float, float]:
    if seed is not None:
        np.random.seed(seed)
    p = np.random.rand(n_samples, 3)
    q = np.random.rand(n_samples, 3)
    t = np.linalg.norm(p - q, axis=1)
    mu = float(np.mean(t))
    if n_samples > 1:
        var = float(np.sum((t - mu) ** 2) / (n_samples - 1))
    else:
        var = 0.0
    return mu, var


def histogram_2d_uniformity(points: np.ndarray,
                            x_range: Tuple[float, float] = (-1.0, 1.0),
                            y_range: Tuple[float, float] = (-1.0, 1.0),
                            bins: int = 10) -> dict:
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points 必须为 (N, 2) 数组")

    H, xedges, yedges = np.histogram2d(
        points[:, 0], points[:, 1],
        bins=bins, range=[x_range, y_range]
    )
    N = points.shape[0]
    expected = N / (bins * bins)
    if expected < 1e-12:
        return {'chi2': 0.0, 'p_uniformity': 0.0, 'max_deviation': 0.0}

    chi2 = np.sum((H - expected) ** 2 / expected)

    df = bins * bins - 1

    from stochastic_channel import alnorm
    p_uniformity = 1.0 - alnorm((chi2 - df) / np.sqrt(2.0 * df), False) if df > 0 else 1.0
    max_deviation = float(np.max(np.abs(H - expected)) / expected)

    return {
        'chi2': float(chi2),
        'p_uniformity': float(p_uniformity),
        'max_deviation': float(max_deviation),
        'histogram': H,
        'xedges': xedges,
        'yedges': yedges,
    }


def compute_array_pattern_metric(pattern_db: np.ndarray,
                                 mainlobe_indices: np.ndarray,
                                 sidelobe_threshold_db: float = -13.0) -> dict:
    pattern_lin = 10.0 ** (pattern_db / 10.0)
    total_power = np.sum(pattern_lin)
    mainlobe_power = np.sum(pattern_lin[mainlobe_indices])

    sidelobe_mask = np.ones(len(pattern_db), dtype=bool)
    sidelobe_mask[mainlobe_indices] = False
    sidelobe_power = np.sum(pattern_lin[sidelobe_mask])

    peak_sidelobe = float(np.max(pattern_db[sidelobe_mask])) if np.any(sidelobe_mask) else -100.0
    islr = 10.0 * np.log10(sidelobe_power / max(mainlobe_power, 1e-18))
    beamwidth_eff = mainlobe_power / max(total_power, 1e-18)

    return {
        'peak_sidelobe_level_db': peak_sidelobe,
        'integrated_sidelobe_ratio_db': float(islr),
        'beamwidth_efficiency': float(beamwidth_eff),
    }


def convergence_rate_residual(residual_history: np.ndarray) -> float:
    if residual_history.size < 2:
        return 0.0
    k = np.arange(residual_history.size)
    log_r = np.log(np.maximum(residual_history, 1e-18))

    A = np.vstack([k, np.ones_like(k)]).T
    m, _ = np.linalg.lstsq(A, log_r, rcond=None)[0]
    return float(np.exp(m))
