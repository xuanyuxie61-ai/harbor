
import numpy as np
import math
from scipy.special import gamma as sp_gamma


def source_spectrum_gaussian(k, k0, delta_k):
    k = np.asarray(k, dtype=float)
    if delta_k <= 0:
        raise ValueError("delta_k must be positive.")
    S = np.exp(-0.5 * ((k - k0) / delta_k) ** 2)

    norm = delta_k * np.sqrt(2.0 * np.pi)
    if norm > 0:
        S = S / norm
    return S


def dispersion_phase(k, k0, phi_coeffs):
    k = np.asarray(k, dtype=float)
    phi_coeffs = np.asarray(phi_coeffs, dtype=float)
    dk = k - k0
    phi = np.zeros_like(k)
    for m, coeff in enumerate(phi_coeffs):
        if m == 0:
            phi += coeff
        else:
            phi += coeff / math.factorial(m) * (dk ** m)
    return phi


def oct_interferogram_fd(k, z, k0, delta_k, phi_coeffs,
                         reflectivity_sample, reflectivity_reference):
    k = np.asarray(k, dtype=float)
    if reflectivity_sample < 0 or reflectivity_reference < 0:
        raise ValueError("Reflectivities must be non-negative.")
    S = source_spectrum_gaussian(k, k0, delta_k)
    phi = dispersion_phase(k, k0, phi_coeffs)
    dc_term = reflectivity_reference + reflectivity_sample
    ac_term = 2.0 * np.sqrt(reflectivity_reference * reflectivity_sample) * np.cos(2.0 * k * z + phi)
    I = S * (dc_term + ac_term)
    return I


def henvey_greenstein_phase_function(cos_theta, g):
    cos_theta = np.asarray(cos_theta, dtype=float)
    if not (-1.0 <= g <= 1.0):
        raise ValueError("Anisotropy factor g must be in [-1, 1].")
    denom = (1.0 + g * g - 2.0 * g * cos_theta) ** 1.5

    denom = np.where(denom < 1e-14, 1e-14, denom)
    p = 0.5 * (1.0 - g * g) / denom
    return p


def mie_scattering_cross_section(radius, n_particle, n_medium, wavelength):
    if radius < 0 or wavelength <= 0:
        raise ValueError("radius and wavelength must be positive.")
    m = n_particle / n_medium
    k = 2.0 * np.pi * n_medium / wavelength
    x = k * radius


    if x < 0.1 and abs(m - 1.0) < 0.5:
        factor = abs((m * m - 1.0) / (m * m + 2.0)) ** 2
        sigma_s = (8.0 * np.pi / 3.0) * (k ** 4) * (radius ** 6) * factor
        sigma_a = 4.0 * np.pi * radius ** 3 * k * np.imag((m * m - 1.0) / (m * m + 2.0))
        sigma_a = max(sigma_a, 0.0)
    elif x < 50.0:

        factor = abs((m * m - 1.0) / (m * m + 2.0)) ** 2
        sigma_s = 2.0 * np.pi * (radius ** 2) * (x ** 2) * factor / (1.0 + x ** 2)
        sigma_a = 0.01 * sigma_s
    else:

        sigma_s = np.pi * radius * radius
        sigma_a = 0.05 * sigma_s
    return sigma_s, sigma_a


def scattering_coefficients(volume_fraction, radius, n_particle, n_medium, wavelength):
    if volume_fraction < 0 or volume_fraction > 1:
        raise ValueError("volume_fraction must be in [0, 1].")
    if radius <= 0:
        raise ValueError("radius must be positive.")
    sigma_s, sigma_a = mie_scattering_cross_section(radius, n_particle, n_medium, wavelength)
    rho = 3.0 * volume_fraction / (4.0 * np.pi * radius ** 3)
    mu_s = rho * sigma_s
    mu_a = rho * sigma_a

    m = n_particle / n_medium
    g = 1.0 - 0.5 / (m * m) if m > 1.0 else 0.0
    g = np.clip(g, -0.99, 0.99)
    return mu_s, mu_a, g


def diffusion_coefficient(mu_s, mu_a, g):


    raise NotImplementedError("Hole 1: diffusion_coefficient needs to be implemented.")


def transport_length(mu_s, mu_a, g):
    mu_s_prime = (1.0 - g) * mu_s
    denom = mu_s_prime + mu_a
    if denom <= 0:
        raise ValueError("Invalid optical properties: denominator <= 0.")
    return 1.0 / denom


def coherence_length_gaussian(delta_lambda, lambda0):
    if delta_lambda <= 0 or lambda0 <= 0:
        raise ValueError("Wavelengths must be positive.")
    l_c = (2.0 * np.log(2.0) / np.pi) * (lambda0 ** 2 / delta_lambda)
    return l_c


def signal_to_noise_ratio_oct(reflectivity, shot_noise_power, detector_noise_power):
    if reflectivity < 0:
        raise ValueError("reflectivity must be non-negative.")
    denominator = shot_noise_power + detector_noise_power
    if denominator <= 0:
        return 1e6
    snr_linear = reflectivity / denominator
    snr_db = 10.0 * np.log10(max(snr_linear, 1e-12))
    return snr_db


def speckle_contrast(intensity_array):
    arr = np.asarray(intensity_array, dtype=float)
    if arr.size == 0:
        return 0.0
    mean_i = np.mean(arr)
    if mean_i <= 1e-14:
        return 0.0
    std_i = np.std(arr, ddof=1)
    C = std_i / mean_i
    return C


def doppler_phase_shift(v_flow, n_medium, lambda0, theta=0.0):
    if lambda0 <= 0:
        raise ValueError("lambda0 must be positive.")
    delta_f = 2.0 * n_medium * v_flow * np.cos(theta) / lambda0
    return delta_f
