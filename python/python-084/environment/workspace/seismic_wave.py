# -*- coding: utf-8 -*-
"""
seismic_wave.py
===============
Synthetic ground-motion generation and processing for time-history analysis.

Incorporates ideas from three seed projects:
  - 171_chirikov_iteration:   Chirikov standard map for chaotic phase perturbation
  - 045_asa159:               Random contingency table for phase-angle constrained sampling
  - 584_image_rgb_to_gray:    Weighted multi-channel fusion of horizontal components

Physical model:
  Ground acceleration is synthesized via the stochastic method (Boore, 2003):
    a_g(t) = I(t) * (1/N) * sum_k A(omega_k) * cos(omega_k * t + phi_k)
  where
    I(t)       = envelope function (Shamoto & Iwasaki, ARI)
    A(omega)   = sqrt( S(omega) * Delta_omega )   (Fourier amplitude)
    S(omega)   = Kanai-Tajimi power spectral density
    phi_k      = random phases, perturbed by Chirikov map for chaos realism

The multi-component ground motion (x, y, z) is fused via luminance-like
weights (0.7 horizontal, 0.2 horizontal transverse, 0.1 vertical),
adapted from the RGB-to-gray weighted average concept.
"""

import numpy as np
from typing import Tuple, Optional


# ---------------------------------------------------------------------- #
# Envelope function
# ---------------------------------------------------------------------- #
def envelope_ari(t: np.ndarray, t_rise: float = 2.0, t_flat: float = 8.0, t_decay: float = 4.0) -> np.ndarray:
    """
    ARI (Arias Intensity) envelope function for ground-motion shaping.
    
       I(t) = (t / t_rise)^2           for 0 <= t <= t_rise
       I(t) = 1.0                      for t_rise < t <= t_rise + t_flat
       I(t) = exp( -c * (t - t_rise - t_flat) )   for t > t_rise + t_flat
    
    where c is chosen so I(t_total) ~ 0.01.
    """
    t = np.asarray(t, dtype=float)
    env = np.zeros_like(t)
    t_total = t_rise + t_flat + t_decay

    # Rise segment
    mask_rise = (t >= 0) & (t <= t_rise)
    env[mask_rise] = (t[mask_rise] / t_rise) ** 2

    # Flat segment
    mask_flat = (t > t_rise) & (t <= t_rise + t_flat)
    env[mask_flat] = 1.0

    # Decay segment
    mask_decay = t > t_rise + t_flat
    c = np.log(100.0) / t_decay   # so at end of decay, env ~ 0.01
    env[mask_decay] = np.exp(-c * (t[mask_decay] - t_rise - t_flat))

    # Boundary protection: force zero for negative t
    env[t < 0] = 0.0
    return env


# ---------------------------------------------------------------------- #
# Kanai-Tajimi power spectral density
# ---------------------------------------------------------------------- #
def kanai_tajimi_psd(omega: np.ndarray, omega_g: float = 15.0, zeta_g: float = 0.6, S0: float = 0.01) -> np.ndarray:
    """
    Kanai-Tajimi power spectral density for firm ground:
    
               S0 * (omega_g^4 + 4 * zeta_g^2 * omega_g^2 * omega^2)
      S(omega) = ---------------------------------------------------
               (omega_g^2 - omega^2)^2 + 4 * zeta_g^2 * omega_g^2 * omega^2
    
    Parameters
    ----------
    omega : np.ndarray
        Circular frequency [rad/s].
    omega_g : float
        Dominant ground frequency [rad/s].
    zeta_g : float
        Ground damping ratio.
    S0 : float
        White noise intensity factor [m^2 / s^3].
    """
    omega = np.asarray(omega, dtype=float)
    # Protect against omega = 0 division issues
    omega_safe = np.where(omega == 0, 1e-12, omega)

    num = S0 * (omega_g ** 4 + 4.0 * zeta_g ** 2 * omega_g ** 2 * omega_safe ** 2)
    den = (omega_g ** 2 - omega_safe ** 2) ** 2 + 4.0 * zeta_g ** 2 * omega_g ** 2 * omega_safe ** 2

    psd = np.where(den > 0, num / den, 0.0)
    return psd


# ---------------------------------------------------------------------- #
# Chirikov standard map for phase perturbation (from 171_chirikov_iteration)
# ---------------------------------------------------------------------- #
def chirikov_map_perturbation(phases: np.ndarray, K: float = 0.8, iterations: int = 3) -> np.ndarray:
    """
    Apply the Chirikov standard map iteratively to perturb phase angles,
    introducing deterministic chaos that mimics the nonlinear phase
    interactions present in real earthquake rupture processes.
    
    Standard map:
      I_{n+1} = I_n + K * sin(theta_n)
      theta_{n+1} = theta_n + I_{n+1}   (mod 2*pi)
    
    We use the action I as a proxy for phase perturbation strength.
    """
    theta = np.asarray(phases, dtype=float).copy()
    N = len(theta)
    I = np.zeros(N, dtype=float)

    for _ in range(iterations):
        I = I + K * np.sin(theta)
        theta = theta + I
        theta = np.mod(theta, 2.0 * np.pi)

    return theta


# ---------------------------------------------------------------------- #
# Random phase constrained sampling (from 045_asa159 idea)
# ---------------------------------------------------------------------- #
def constrained_random_phases(n: int, seed: int = 42) -> np.ndarray:
    """
    Generate random phase angles with a statistical uniformity constraint
    inspired by the random contingency table (ASA 159) seed.
    
    The contingency table enforces row/column sum constraints; here we
    enforce that the phase distribution is approximately uniform in
    [0, 2*pi] by rejection-sampling any cluster that deviates > 20%.
    """
    rng = np.random.default_rng(seed)
    phases = rng.uniform(0.0, 2.0 * np.pi, size=n)

    # Bin phases into 8 angular bins and check uniformity
    n_bins = 8
    max_iter = 100
    for _iter in range(max_iter):
        hist, _ = np.histogram(phases, bins=n_bins, range=(0.0, 2.0 * np.pi))
        expected = n / n_bins
        max_dev = np.max(np.abs(hist - expected)) / expected
        if max_dev < 0.20:
            break
        # Resample the most over-populated bin
        max_bin = np.argmax(hist)
        bin_mask = (
            (phases >= max_bin * 2.0 * np.pi / n_bins)
            & (phases < (max_bin + 1) * 2.0 * np.pi / n_bins)
        )
        over_count = int(hist[max_bin] - expected)
        if over_count > 0:
            idx_in_bin = np.where(bin_mask)[0]
            replace_idx = rng.choice(idx_in_bin, size=over_count, replace=False)
            # Move to under-populated bin
            min_bin = np.argmin(hist)
            new_low = min_bin * 2.0 * np.pi / n_bins
            new_high = (min_bin + 1) * 2.0 * np.pi / n_bins
            phases[replace_idx] = rng.uniform(new_low, new_high, size=over_count)

    return phases


# ---------------------------------------------------------------------- #
# Ground motion synthesis
# ---------------------------------------------------------------------- #
def synthesize_ground_motion(
    dt: float = 0.01,
    t_max: float = 20.0,
    omega_g: float = 15.0,
    zeta_g: float = 0.6,
    S0: float = 0.02,
    K_chirikov: float = 0.5,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Synthesize a synthetic ground acceleration time history.
    
    Parameters
    ----------
    dt : float
        Time step [s].
    t_max : float
        Total duration [s].
    omega_g, zeta_g, S0 : float
        Kanai-Tajimi PSD parameters.
    K_chirikov : float
        Chirikov map perturbation strength.
    seed : int
        Random seed.
    
    Returns
    -------
    t : np.ndarray
        Time vector [s].
    a_g : np.ndarray
        Ground acceleration [m/s^2].
    """
    t = np.arange(0.0, t_max + dt, dt)
    N = len(t)
    # Nyquist frequency
    omega_max = np.pi / dt
    # Frequency spacing
    domega = 2.0 * np.pi / t_max
    n_freq = int(omega_max / domega)
    omega_k = np.arange(1, n_freq + 1) * domega

    # Kanai-Tajimi PSD and Fourier amplitudes
    S_omega = kanai_tajimi_psd(omega_k, omega_g, zeta_g, S0)
    A_k = np.sqrt(2.0 * S_omega * domega)

    # Constrained random phases + Chirikov perturbation
    phi_k = constrained_random_phases(len(omega_k), seed=seed)
    phi_k = chirikov_map_perturbation(phi_k, K=K_chirikov, iterations=2)

    # Envelope
    env = envelope_ari(t, t_rise=2.0, t_flat=8.0, t_decay=6.0)

    # Sum of cosines (inverse Fourier transform via direct summation)
    a_g = np.zeros(N, dtype=float)
    for k in range(len(omega_k)):
        a_g += A_k[k] * np.cos(omega_k[k] * t + phi_k[k])

    # Apply envelope
    a_g = env * a_g

    # Zero-mean correction (remove any DC drift)
    a_g = a_g - np.mean(a_g)

    # Peak ground acceleration (PGA) normalization to ~0.3 g (strong motion)
    pga_target = 0.3 * 9.81   # 0.3 g ~ 2.94 m/s^2
    pga_current = np.max(np.abs(a_g))
    if pga_current > 1e-12:
        a_g = a_g * (pga_target / pga_current)

    return t, a_g


# ---------------------------------------------------------------------- #
# Multi-component fusion (from 584_image_rgb_to_gray weighted average)
# ---------------------------------------------------------------------- #
def fuse_components(
    a_x: np.ndarray,
    a_y: np.ndarray,
    a_z: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Fuse three-component ground motion into a single representative
    acceleration record using weighted averaging (luminance analogy).
    
    Default weights: [0.70, 0.20, 0.10] for (major horizontal,
    transverse horizontal, vertical), consistent with engineering practice
    where the major horizontal component dominates seismic demand.
    """
    if weights is None:
        weights = np.array([0.70, 0.20, 0.10], dtype=float)
    weights = np.asarray(weights, dtype=float)
    weights = weights / np.sum(weights)

    return weights[0] * a_x + weights[1] * a_y + weights[2] * a_z


# ---------------------------------------------------------------------- #
# High-level seismic wave generator class
# ---------------------------------------------------------------------- #
class SeismicWaveGenerator:
    """
    Generates synthetic multi-component ground motions for time-history
    analysis of base-isolated structures.
    """

    def __init__(self, dt: float = 0.01, t_max: float = 20.0, seed: int = 84):
        self.dt = dt
        self.t_max = t_max
        self.seed = seed
        self.t = np.arange(0.0, t_max + dt, dt)

    def generate(self, n_components: int = 3) -> np.ndarray:
        """
        Generate ground acceleration array of shape (n_time, n_components).
        Each component uses a different random seed derived from base seed.
        """
        acc = np.zeros((len(self.t), n_components), dtype=float)
        for i in range(n_components):
            _, a = synthesize_ground_motion(
                dt=self.dt,
                t_max=self.t_max,
                seed=self.seed + i * 1000,
                omega_g=15.0 + i * 2.0,   # Slightly different soil freq per component
            )
            acc[:, i] = a
        return acc

    def get_fused_record(self, weights: Optional[np.ndarray] = None) -> np.ndarray:
        """Return the fused single-component record."""
        acc = self.generate(n_components=3)
        return fuse_components(acc[:, 0], acc[:, 1], acc[:, 2], weights=weights)
