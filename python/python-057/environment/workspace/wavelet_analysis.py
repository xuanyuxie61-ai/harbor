
import numpy as np


def haar_1d_transform(signal):
    v = np.asarray(signal, dtype=float)
    n = len(v)
    

    m = 2**int(np.floor(np.log2(n)))
    v = v[:m]
    
    coeffs = []
    energies = []
    current = v.copy()
    
    while len(current) > 1:
        k = len(current) // 2
        

        avg = (current[0::2] + current[1::2]) / np.sqrt(2.0)

        diff = (current[0::2] - current[1::2]) / np.sqrt(2.0)
        
        energies.append(np.sum(diff**2))
        coeffs.append(diff)
        current = avg
    
    coeffs.append(current)
    coeffs.reverse()
    energies.reverse()
    
    return coeffs, energies


def haar_1d_inverse(coeffs):
    approx = coeffs[0].copy()
    
    for j in range(1, len(coeffs)):
        detail = coeffs[j]
        n = len(approx)
        
        reconstructed = np.zeros(2 * n)
        reconstructed[0::2] = (approx + detail) / np.sqrt(2.0)
        reconstructed[1::2] = (approx - detail) / np.sqrt(2.0)
        approx = reconstructed
    
    return approx


def haar_2d_transform(field):
    u = np.asarray(field, dtype=float)
    rows, cols = u.shape
    

    m_r = 2**int(np.floor(np.log2(rows)))
    m_c = 2**int(np.floor(np.log2(cols)))
    u = u[:m_r, :m_c]
    

    col_approx = np.zeros((m_r // 2, m_c))
    col_detail = np.zeros((m_r // 2, m_c))
    
    for j in range(m_c):
        c, _ = haar_1d_transform(u[:, j])
        if len(c) >= 2:
            col_approx[:, j] = c[0][:m_r//2]

            col_detail[:, j] = c[-1][:m_r//2]
    

    LL = np.zeros((m_r // 2, m_c // 2))
    LH = np.zeros((m_r // 2, m_c // 2))
    HL = np.zeros((m_r // 2, m_c // 2))
    HH = np.zeros((m_r // 2, m_c // 2))
    
    for i in range(m_r // 2):
        c_a, _ = haar_1d_transform(col_approx[i, :])
        c_d, _ = haar_1d_transform(col_detail[i, :])
        
        if len(c_a) >= 2 and len(c_d) >= 2:
            LL[i, :] = c_a[0][:m_c//2]
            LH[i, :] = c_a[-1][:m_c//2] if len(c_a) > 1 else 0.0
            HL[i, :] = c_d[0][:m_c//2] if len(c_d) > 1 else 0.0
            HH[i, :] = c_d[-1][:m_c//2] if len(c_d) > 1 else 0.0
    
    return LL, LH, HL, HH


def detect_breaking_events(signal, threshold_factor=3.0):
    coeffs, energies = haar_1d_transform(signal)
    

    if len(coeffs) >= 2:
        detail = coeffs[-1]
    else:
        detail = np.zeros(1)
    

    window = max(4, len(detail) // 32)
    wavelet_energy = np.zeros(len(detail))
    
    for i in range(len(detail)):
        start = max(0, i - window // 2)
        end = min(len(detail), i + window // 2 + 1)
        wavelet_energy[i] = np.sum(detail[start:end]**2)
    

    mean_energy = np.mean(wavelet_energy)
    std_energy = np.std(wavelet_energy)
    threshold = mean_energy + threshold_factor * std_energy
    
    breaking_indices = np.where(wavelet_energy > threshold)[0]
    
    return breaking_indices, wavelet_energy


def multi_scale_spectrum(signal):
    coeffs, energies = haar_1d_transform(signal)
    

    scales = np.arange(1, len(energies) + 1)
    spectrum = np.array(energies)
    

    if np.sum(spectrum) > 1.0e-12:
        spectrum = spectrum / np.sum(spectrum)
    
    return scales, spectrum
