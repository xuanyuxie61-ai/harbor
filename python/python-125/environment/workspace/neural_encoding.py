
import numpy as np
from typing import Tuple






def trig_interp_basis(x: np.ndarray, k: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    result = np.zeros_like(x)
    

    eps = 1e-14
    
    for i in range(len(x)):
        xi = x[i]
        if abs(xi) < eps:
            result[i] = 1.0
            continue
        
        denom = np.sin(np.pi * xi / 2.0)
        if abs(denom) < eps:



            result[i] = 1.0 if k % 2 == 1 else 0.0
            continue
        
        numer = np.sin(k * np.pi * xi / 2.0)
        
        if k % 2 == 1:
            result[i] = numer / (k * denom)
        else:
            tan_val = np.tan(np.pi * xi / 2.0)
            if abs(tan_val) < eps:
                result[i] = 0.0
            else:
                result[i] = numer / (k * tan_val)
    
    return result


def trig_interpolate_spike_pattern(
    spike_times: np.ndarray,
    spike_values: np.ndarray,
    t_eval: np.ndarray,
    period: float = 2.0
) -> np.ndarray:
    N = len(spike_times)
    if N == 0:
        return np.zeros_like(t_eval)
    

    x_nodes = 2.0 * spike_times / period - 1.0
    x_eval = 2.0 * t_eval / period - 1.0
    
    result = np.zeros_like(t_eval)
    for j in range(N):

        shift = x_eval - x_nodes[j]

        shift = ((shift + 1.0) % 2.0) - 1.0
        result += spike_values[j] * trig_interp_basis(shift, N)
    
    return result






def horner_polynomial_eval(coeffs: np.ndarray, x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    coeffs = np.asarray(coeffs, dtype=np.float64)
    m = len(coeffs) - 1
    
    if m < 0:
        return np.zeros_like(x)
    

    p = np.full_like(x, coeffs[m])
    for k in range(m - 1, -1, -1):
        p = p * x + coeffs[k]
    
    return p


def compute_tuning_curve(
    stimulus_values: np.ndarray,
    coeffs: np.ndarray
) -> np.ndarray:
    return horner_polynomial_eval(coeffs, stimulus_values)






def analyze_spike_train(
    spike_times: np.ndarray,
    t_max: float,
    n_bins: int = 100
) -> dict:
    n_spikes = len(spike_times)
    if n_spikes < 2:
        return {
            'mean_rate': n_spikes / t_max if t_max > 0 else 0.0,
            'cv_isi': 0.0,
            'isi_mean': 0.0,
            'isi_std': 0.0,
            'rate_histogram': np.zeros(n_bins),
            'bin_edges': np.linspace(0, t_max, n_bins + 1),
        }
    

    mean_rate = n_spikes / t_max
    

    isi = np.diff(spike_times)
    isi_mean = float(np.mean(isi))
    isi_std = float(np.std(isi))
    cv_isi = isi_std / isi_mean if isi_mean > 1e-14 else 0.0
    

    hist, bin_edges = np.histogram(spike_times, bins=n_bins, range=(0, t_max))
    bin_width = t_max / n_bins
    rate_histogram = hist / bin_width
    
    return {
        'mean_rate': mean_rate,
        'cv_isi': cv_isi,
        'isi_mean': isi_mean,
        'isi_std': isi_std,
        'rate_histogram': rate_histogram,
        'bin_edges': bin_edges,
        'n_spikes': n_spikes,
    }


def encoding_efficiency(
    spike_train: np.ndarray,
    stimulus: np.ndarray,
    dt: float
) -> dict:
    T = len(spike_train)
    if T == 0:
        return {'info_rate': 0.0, 'snr': 0.0, 'correlation': 0.0}
    

    stimulus_mean = np.mean(stimulus)
    spike_mean = np.mean(spike_train)
    
    cov = np.mean((stimulus - stimulus_mean) * (spike_train - spike_mean))
    var_s = np.var(stimulus)
    var_r = np.var(spike_train)
    
    if var_s < 1e-14 or var_r < 1e-14:
        correlation = 0.0
        snr = 0.0
    else:
        correlation = cov / np.sqrt(var_s * var_r)
        snr = correlation ** 2 / (1.0 - correlation ** 2 + 1e-14)
    

    info_rate = 0.5 * np.log2(1.0 + snr) if snr > 0 else 0.0
    
    return {
        'info_rate_bits': float(info_rate),
        'snr': float(snr),
        'correlation': float(correlation),
        'mean_stimulus': float(stimulus_mean),
        'mean_spike_rate': float(spike_mean / dt),
    }
