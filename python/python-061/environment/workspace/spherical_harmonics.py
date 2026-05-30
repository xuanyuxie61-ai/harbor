
import numpy as np
from scipy.special import lpmv, factorial


EARTH_RADIUS = 6.371e6


def associated_legendre(l, m, x):
    x = np.atleast_1d(x)

    return lpmv(m, l, x)


def spherical_harmonic_y(l, m, theta, phi):
    x = np.cos(theta)
    

    if m >= 0:
        norm_sq = (2.0 * l + 1.0) / (4.0 * np.pi) * factorial(l - m) / factorial(l + m)
    else:

        m_pos = -m
        norm_sq = (2.0 * l + 1.0) / (4.0 * np.pi) * factorial(l - m_pos) / factorial(l + m_pos)
    
    norm = np.sqrt(norm_sq)
    

    p_lm = associated_legendre(l, abs(m), x)
    

    if m >= 0:
        y_val = norm * p_lm * np.exp(1j * m * phi)
    else:
        y_val = (-1)**abs(m) * norm * p_lm * np.exp(1j * m * phi)
    
    return y_val


def compute_spectral_coefficients_1d(theta, values, L_max=20):
    n = len(theta)
    coeffs = np.zeros(L_max + 1, dtype=complex)
    

    dtheta = np.diff(theta)
    
    for l in range(L_max + 1):
        y_l0 = spherical_harmonic_y(l, 0, theta, 0.0)
        integrand = values * np.conj(y_l0) * np.sin(theta)
        

        integral = 0.0
        for i in range(n - 1):
            integral += 0.5 * (integrand[i] + integrand[i + 1]) * dtheta[i]
        
        coeffs[l] = integral
    
    return coeffs


def reconstruct_from_spectral_1d(theta, coeffs):
    L_max = len(coeffs) - 1
    values = np.zeros(len(theta), dtype=complex)
    
    for l in range(L_max + 1):
        y_l0 = spherical_harmonic_y(l, 0, theta, 0.0)
        values += coeffs[l] * y_l0
    
    return np.real(values)


def spectral_laplacian_1d(coeffs, R=EARTH_RADIUS):
    L_max = len(coeffs) - 1
    lap_coeffs = np.zeros_like(coeffs)
    
    for l in range(L_max + 1):
        lap_coeffs[l] = -l * (l + 1.0) / (R**2) * coeffs[l]
    
    return lap_coeffs


def chebyshev_spectral_filter(coeffs, order=4):
    L_max = len(coeffs) - 1
    if L_max <= 0:
        return coeffs.copy()
    
    alpha = 1.0
    filtered = np.zeros_like(coeffs)
    
    for l in range(L_max + 1):
        damping = 1.0 / (1.0 + alpha * (l / L_max)**(2 * order))
        filtered[l] = coeffs[l] * damping
    
    return filtered



def spectral_variance_spectrum(coeffs):
    L_max = len(coeffs) - 1
    energy = np.zeros(L_max + 1)
    wavenumbers = np.arange(L_max + 1)
    
    for l in range(L_max + 1):
        energy[l] = l * (l + 1.0) * np.abs(coeffs[l])**2
    
    return energy, wavenumbers
