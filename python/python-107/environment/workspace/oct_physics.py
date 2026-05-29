"""
oct_physics.py

Core physical models for Spectral-Domain Optical Coherence Tomography (SD-OCT).
Implements interferometry, dispersion, scattering theory, and coherence physics.

Scientific foundations:
- Wiener-Khinchin theorem for spectral interferometry
- Mie scattering theory for spherical particles
- Henyey-Greenstein phase function for anisotropic scattering
- Diffusion approximation of radiative transfer equation
- Dispersion compensation via Taylor expansion of phase
"""

import numpy as np
import math
from scipy.special import gamma as sp_gamma


def source_spectrum_gaussian(k, k0, delta_k):
    """
    Gaussian spectral density of the low-coherence light source.

    S(k) = S_0 * exp( - (k - k0)^2 / (2 * delta_k^2) )

    Parameters
    ----------
    k : array_like
        Wavenumber (rad/micron).
    k0 : float
        Central wavenumber.
    delta_k : float
        Spectral bandwidth (standard deviation).

    Returns
    -------
    S : ndarray
        Normalized spectral density.
    """
    k = np.asarray(k, dtype=float)
    if delta_k <= 0:
        raise ValueError("delta_k must be positive.")
    S = np.exp(-0.5 * ((k - k0) / delta_k) ** 2)
    # Normalize so that integral over k is 1 (approximate)
    norm = delta_k * np.sqrt(2.0 * np.pi)
    if norm > 0:
        S = S / norm
    return S


def dispersion_phase(k, k0, phi_coeffs):
    """
    Taylor-expanded dispersion phase accumulated in the sample arm.

    phi(k) = sum_{m=0}^{M} phi_m / m! * (k - k0)^m

    Parameters
    ----------
    k : array_like
        Wavenumber.
    k0 : float
        Central wavenumber.
    phi_coeffs : array_like
        Coefficients [phi_0, phi_1, phi_2, ...].

    Returns
    -------
    phi : ndarray
        Phase at each wavenumber.
    """
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
    """
    Spectral-domain OCT interferogram for a single reflector at depth z.

    I(k) = S(k) * [ R_R + R_S + 2 * sqrt(R_R * R_S) * cos(2 k z + phi(k)) ]

    Parameters
    ----------
    k : array_like
        Wavenumber array.
    z : float
        Depth of the sample reflector (micron).
    k0, delta_k : float
        Source spectrum parameters.
    phi_coeffs : array_like
        Dispersion coefficients.
    reflectivity_sample : float
        Sample arm reflectivity R_S.
    reflectivity_reference : float
        Reference arm reflectivity R_R.

    Returns
    -------
    I : ndarray
        Interferogram spectral intensity.
    """
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
    """
    Henyey-Greenstein scattering phase function for anisotropic media.

    p(cos_theta) = (1 - g^2) / [ 2 * (1 + g^2 - 2 g cos_theta)^{3/2} ]

    This is the normalized probability density for scattering angle cosine.

    Parameters
    ----------
    cos_theta : array_like
        Cosine of scattering angle.
    g : float
        Anisotropy factor, -1 <= g <= 1.

    Returns
    -------
    p : ndarray
        Phase function values.
    """
    cos_theta = np.asarray(cos_theta, dtype=float)
    if not (-1.0 <= g <= 1.0):
        raise ValueError("Anisotropy factor g must be in [-1, 1].")
    denom = (1.0 + g * g - 2.0 * g * cos_theta) ** 1.5
    # Guard against division by zero
    denom = np.where(denom < 1e-14, 1e-14, denom)
    p = 0.5 * (1.0 - g * g) / denom
    return p


def mie_scattering_cross_section(radius, n_particle, n_medium, wavelength):
    """
    Approximate Mie scattering cross section for spherical particles
    using the Rayleigh-Debye-Gans approximation when |m-1| << 1 and
    2 k a |m-1| << 1, otherwise fall back to geometric optics scaling.

    For Rayleigh regime (small particles):
    sigma_s = (8 pi / 3) * k^4 * a^6 * |(m^2 - 1)/(m^2 + 2)|^2

    where m = n_p / n_m, k = 2 pi n_m / lambda.

    Parameters
    ----------
    radius : float
        Particle radius (micron).
    n_particle : float
        Refractive index of particle.
    n_medium : float
        Refractive index of medium.
    wavelength : float
        Vacuum wavelength (micron).

    Returns
    -------
    sigma_s : float
        Scattering cross section (micron^2).
    sigma_a : float
        Absorption cross section (micron^2), approximate.
    """
    if radius < 0 or wavelength <= 0:
        raise ValueError("radius and wavelength must be positive.")
    m = n_particle / n_medium
    k = 2.0 * np.pi * n_medium / wavelength
    x = k * radius  # Size parameter

    # Rayleigh regime criterion
    if x < 0.1 and abs(m - 1.0) < 0.5:
        factor = abs((m * m - 1.0) / (m * m + 2.0)) ** 2
        sigma_s = (8.0 * np.pi / 3.0) * (k ** 4) * (radius ** 6) * factor
        sigma_a = 4.0 * np.pi * radius ** 3 * k * np.imag((m * m - 1.0) / (m * m + 2.0))
        sigma_a = max(sigma_a, 0.0)
    elif x < 50.0:
        # Intermediate: use approximate scaling
        factor = abs((m * m - 1.0) / (m * m + 2.0)) ** 2
        sigma_s = 2.0 * np.pi * (radius ** 2) * (x ** 2) * factor / (1.0 + x ** 2)
        sigma_a = 0.01 * sigma_s  # heuristic
    else:
        # Large particles: geometric optics limit
        sigma_s = np.pi * radius * radius
        sigma_a = 0.05 * sigma_s
    return sigma_s, sigma_a


def scattering_coefficients(volume_fraction, radius, n_particle, n_medium, wavelength):
    """
    Compute bulk scattering and absorption coefficients from particle properties.

    mu_s = rho * sigma_s,  mu_a = rho * sigma_a
    where rho = 3 * phi_v / (4 pi a^3) is number density.

    Parameters
    ----------
    volume_fraction : float
        Particle volume fraction phi_v.
    radius : float
        Particle radius.
    n_particle, n_medium : float
        Refractive indices.
    wavelength : float
        Wavelength.

    Returns
    -------
    mu_s, mu_a, g : float
        Scattering coefficient, absorption coefficient, anisotropy.
    """
    if volume_fraction < 0 or volume_fraction > 1:
        raise ValueError("volume_fraction must be in [0, 1].")
    if radius <= 0:
        raise ValueError("radius must be positive.")
    sigma_s, sigma_a = mie_scattering_cross_section(radius, n_particle, n_medium, wavelength)
    rho = 3.0 * volume_fraction / (4.0 * np.pi * radius ** 3)
    mu_s = rho * sigma_s
    mu_a = rho * sigma_a
    # Anisotropy factor g for HG: approximate using forward scattering lobe
    m = n_particle / n_medium
    g = 1.0 - 0.5 / (m * m) if m > 1.0 else 0.0
    g = np.clip(g, -0.99, 0.99)
    return mu_s, mu_a, g


def diffusion_coefficient(mu_s, mu_a, g):
    """
    Diffusion coefficient in the diffusion approximation of RTE.

    D = 1 / (3 * (mu_s' + mu_a))
    where mu_s' = (1 - g) * mu_s is the reduced scattering coefficient.

    Parameters
    ----------
    mu_s, mu_a, g : float
        Scattering, absorption, anisotropy.

    Returns
    -------
    D : float
        Diffusion coefficient.
    """
    # TODO: Implement the diffusion coefficient formula for the RTE diffusion approximation.
    # Hint: reduced scattering coefficient mu_s' = (1 - g) * mu_s, then D = 1 / (3 * (mu_s' + mu_a)).
    raise NotImplementedError("Hole 1: diffusion_coefficient needs to be implemented.")


def transport_length(mu_s, mu_a, g):
    """
    Transport mean free path.

    l_tr = 1 / (mu_s' + mu_a)

    Returns
    -------
    l_tr : float
    """
    mu_s_prime = (1.0 - g) * mu_s
    denom = mu_s_prime + mu_a
    if denom <= 0:
        raise ValueError("Invalid optical properties: denominator <= 0.")
    return 1.0 / denom


def coherence_length_gaussian(delta_lambda, lambda0):
    """
    Axial coherence length (FWHM) for Gaussian spectrum.

    l_c = (2 ln 2 / pi) * (lambda0^2 / delta_lambda)

    Parameters
    ----------
    delta_lambda : float
        Spectral bandwidth (FWHM) in vacuum wavelength.
    lambda0 : float
        Central wavelength.

    Returns
    -------
    l_c : float
        Coherence length.
    """
    if delta_lambda <= 0 or lambda0 <= 0:
        raise ValueError("Wavelengths must be positive.")
    l_c = (2.0 * np.log(2.0) / np.pi) * (lambda0 ** 2 / delta_lambda)
    return l_c


def signal_to_noise_ratio_oct(reflectivity, shot_noise_power, detector_noise_power):
    """
    OCT SNR model for shot-noise-limited detection.

    SNR = (eta e / (h nu) * P_s)^2 / (2 e^2 B (eta / (h nu)) (P_R + P_S) + N_det)

    Simplified: SNR ~ R_S / (shot_noise_power + detector_noise_power)

    Parameters
    ----------
    reflectivity : float
        Sample reflectivity.
    shot_noise_power, detector_noise_power : float
        Noise powers.

    Returns
    -------
    snr : float
        Signal-to-noise ratio in dB.
    """
    if reflectivity < 0:
        raise ValueError("reflectivity must be non-negative.")
    denominator = shot_noise_power + detector_noise_power
    if denominator <= 0:
        return 1e6
    snr_linear = reflectivity / denominator
    snr_db = 10.0 * np.log10(max(snr_linear, 1e-12))
    return snr_db


def speckle_contrast(intensity_array):
    """
    Speckle contrast C = sigma_I / <I> for fully developed speckle.
    For ideal fully developed polarized speckle, C -> 1.

    Parameters
    ----------
    intensity_array : array_like
        Intensity values.

    Returns
    -------
    C : float
        Speckle contrast.
    """
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
    """
    Doppler frequency shift in phase-resolved OCT.

    Delta_phi = (4 pi n_medium v_flow cos(theta) / lambda0) * tau

    where tau is the integration time.

    Parameters
    ----------
    v_flow : float
        Flow velocity (micron/s).
    n_medium : float
        Refractive index.
    lambda0 : float
        Central wavelength.
    theta : float
        Angle between flow direction and beam axis.

    Returns
    -------
    delta_f : float
        Doppler frequency shift (Hz) per unit time.
    """
    if lambda0 <= 0:
        raise ValueError("lambda0 must be positive.")
    delta_f = 2.0 * n_medium * v_flow * np.cos(theta) / lambda0
    return delta_f
