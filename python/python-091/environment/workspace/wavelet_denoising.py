
import numpy as np
from typing import Tuple, List


def haar_step_1d(signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n = len(signal)
    if n % 2 != 0:

        signal = signal[:-1]
        n -= 1
    
    if n < 2:
        return signal.copy(), np.zeros_like(signal)
    
    half = n // 2
    approx = np.zeros(half)
    detail = np.zeros(half)
    
    for k in range(half):
        approx[k] = (signal[2*k] + signal[2*k+1]) / np.sqrt(2.0)
        detail[k] = (signal[2*k] - signal[2*k+1]) / np.sqrt(2.0)
    
    return approx, detail


def haar_1d(signal: np.ndarray, n_levels: int = None) -> Tuple[np.ndarray, List[np.ndarray]]:
    n = len(signal)
    max_levels = int(np.floor(np.log2(n)))
    
    if n_levels is None:
        n_levels = max_levels
    else:
        n_levels = min(n_levels, max_levels)
    
    if n_levels < 1:
        return signal.copy(), []
    
    details = []
    current = signal.astype(float)
    
    for _ in range(n_levels):
        approx, detail = haar_step_1d(current)
        details.append(detail)
        current = approx
    

    coeffs = [current]
    for detail in reversed(details):
        coeffs.append(detail)
    
    return np.concatenate(coeffs), details


def haar_1d_inverse(coeffs: np.ndarray, n_levels: int) -> np.ndarray:
    if n_levels < 1:
        return coeffs.copy()
    



    n_total = len(coeffs)
    approx_len = n_total // (2**n_levels)
    

    current = coeffs[:approx_len].copy()
    idx = approx_len
    
    for level in range(n_levels):
        detail_len = len(current)
        detail = coeffs[idx:idx + detail_len]
        idx += detail_len
        

        reconstructed = np.zeros(2 * detail_len)
        for k in range(detail_len):
            reconstructed[2*k] = (current[k] + detail[k]) / np.sqrt(2.0)
            reconstructed[2*k+1] = (current[k] - detail[k]) / np.sqrt(2.0)
        
        current = reconstructed
    
    return current


def universal_threshold(details: List[np.ndarray], sigma: float = None) -> float:

    finest_detail = details[-1]
    N = sum(len(d) for d in details) + len(details[0]) if details else len(finest_detail)
    
    if sigma is None:

        median_abs = np.median(np.abs(finest_detail))
        sigma = median_abs / 0.6745
        if sigma < 1e-14:
            sigma = 1e-14
    
    threshold = sigma * np.sqrt(2.0 * np.log(N))
    return threshold


def soft_threshold(coeffs: np.ndarray, threshold: float) -> np.ndarray:
    return np.sign(coeffs) * np.maximum(np.abs(coeffs) - threshold, 0.0)


def hard_threshold(coeffs: np.ndarray, threshold: float) -> np.ndarray:
    result = coeffs.copy()
    result[np.abs(result) <= threshold] = 0.0
    return result


def denoise_ultrasound_signal(signal: np.ndarray, n_levels: int = None,
                              threshold_mode: str = 'soft') -> Tuple[np.ndarray, dict]:

    coeffs, details = haar_1d(signal, n_levels)
    

    threshold = universal_threshold(details)
    


    n_approx = len(details[0]) if details else len(coeffs) // 2
    

    approx_len = len(coeffs)
    for d in details:
        approx_len -= len(d)
    
    approx_coeffs = coeffs[:approx_len].copy()
    detail_coeffs = coeffs[approx_len:].copy()
    

    if threshold_mode == 'soft':
        detail_coeffs = soft_threshold(detail_coeffs, threshold)
    else:
        detail_coeffs = hard_threshold(detail_coeffs, threshold)
    

    denoised_coeffs = np.concatenate([approx_coeffs, detail_coeffs])
    denoised = haar_1d_inverse(denoised_coeffs, len(details))
    

    denoised = denoised[:len(signal)]
    
    info = {
        'n_levels': len(details),
        'threshold': float(threshold),
        'noise_estimate': float(threshold / np.sqrt(2.0 * np.log(len(signal)))),
        'threshold_mode': threshold_mode,
        'original_energy': float(np.sum(signal**2)),
        'denoised_energy': float(np.sum(denoised**2))
    }
    
    return denoised, info


def extract_multiscale_features(signal: np.ndarray, n_levels: int = 4) -> dict:
    coeffs, details = haar_1d(signal, n_levels)
    
    features = {}
    total_energy = np.sum(signal**2)
    
    for i, detail in enumerate(details):
        level = len(details) - i
        energy = np.sum(detail**2)
        

        abs_coeffs = np.abs(detail)
        sum_abs = np.sum(abs_coeffs)
        if sum_abs > 1e-14:
            p = abs_coeffs / sum_abs
            entropy = -np.sum(p * np.log(p + 1e-14))
        else:
            entropy = 0.0
        
        features[f'level_{level}_energy'] = float(energy)
        features[f'level_{level}_energy_ratio'] = float(energy / (total_energy + 1e-14))
        features[f'level_{level}_entropy'] = float(entropy)
        features[f'level_{level}_max_coeff'] = float(np.max(np.abs(detail)))
        features[f'level_{level}_mean_coeff'] = float(np.mean(np.abs(detail)))
    
    return features
