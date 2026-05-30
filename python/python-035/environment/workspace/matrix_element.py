import numpy as np
from constants import M_HIGGS, M_Z, GAMMA_Z, GAMMA_H, G_F, ALPHA_EM, TINY
from utils import horner_eval, lu_factor_scaled, lu_solve, safe_divide




def z_propagator(s, m_z=M_Z, gamma_z=GAMMA_Z):
    denom = s - m_z ** 2 + 1j * m_z * gamma_z
    if abs(denom) < TINY:
        return 0.0 + 0.0j
    return 1.0 / denom


def higgs_propagator(s, m_h=M_HIGGS, gamma_h=GAMMA_H):
    denom = s - m_h ** 2 + 1j * m_h * gamma_h
    if abs(denom) < TINY:
        return 0.0 + 0.0j
    return 1.0 / denom





def g_hzz_coupling():
    v = 1.0 / np.sqrt(np.sqrt(2.0) * G_F)
    return 2.0 * M_Z ** 2 / v


def g_zff_coupling(g_v, g_a):
    return g_v ** 2 + g_a ** 2





def matrix_element_squared_hzz4l(m_z1, m_z2, m_higgs=M_HIGGS):











    raise NotImplementedError("HOLE 1: 请实现 matrix_element_squared_hzz4l")






def vandermonde_quadrature_weights(n, a, b, nodes):
    nodes = np.asarray(nodes, dtype=float)
    vander = np.zeros((n, n))
    rhs = np.zeros(n)
    
    for k in range(n):
        for i in range(n):
            vander[k, i] = nodes[i] ** k
        rhs[k] = (b ** (k + 1) - a ** (k + 1)) / (k + 1.0)
    

    lu, pivot, iflag = lu_factor_scaled(vander)
    if iflag != 0:

        weights = np.linalg.lstsq(vander, rhs, rcond=None)[0]
    else:
        weights = lu_solve(lu, pivot, rhs)
    

    weights = np.maximum(weights, 0.0)
    
    return weights





def fit_amplitude_polynomial(m_z1_vals, m_z2_vals, amplitude_vals, degree=5):
    from utils import cooley_tukey_fft
    
    m_z1_vals = np.asarray(m_z1_vals, dtype=float)
    m_z2_vals = np.asarray(m_z2_vals, dtype=float)
    amp = np.asarray(amplitude_vals, dtype=float)
    

    if len(m_z1_vals) < degree + 1 or len(m_z2_vals) < degree + 1:
        degree = min(len(m_z1_vals), len(m_z2_vals)) - 1
    


    n1 = min(len(m_z1_vals), degree + 1)
    n2 = min(len(m_z2_vals), degree + 1)
    

    idx1 = np.linspace(0, len(m_z1_vals) - 1, n1, dtype=int)
    idx2 = np.linspace(0, len(m_z2_vals) - 1, n2, dtype=int)
    

    amp_proj_1 = np.zeros(n1)
    for i, ii in enumerate(idx1):
        mask = np.abs(m_z2_vals - m_z2_vals[len(m_z2_vals)//2]) < 10.0
        if np.any(mask):
            amp_proj_1[i] = np.mean(amp[ii, mask]) if amp.ndim > 1 else amp[ii]
        else:
            amp_proj_1[i] = amp[ii] if amp.ndim == 1 else amp[ii, 0]
    

    x_norm = 2.0 * (m_z1_vals[idx1] - np.min(m_z1_vals[idx1])) / (np.max(m_z1_vals[idx1]) - np.min(m_z1_vals[idx1]) + TINY) - 1.0
    coeffs = np.polyfit(x_norm, amp_proj_1, min(degree, len(x_norm)-1))
    
    return coeffs


def eval_amplitude_polynomial(coeffs, x):
    return horner_eval(coeffs, x)





def helicity_amplitude_zzstar(m_z1, m_z2, cos_theta, phi, m_higgs=M_HIGGS):
    if abs(cos_theta) > 1.0:
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
    

    pz1_sq = abs(z_propagator(m_z1 ** 2)) ** 2
    pz2_sq = abs(z_propagator(m_z2 ** 2)) ** 2
    

    angular = 1.0 + cos_theta ** 2
    

    norm = 1.0 / (m_higgs ** 4)
    
    return float(pz1_sq * pz2_sq * angular * norm)
