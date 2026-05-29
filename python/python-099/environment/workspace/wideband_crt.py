"""
wideband_crt.py
---------------
Wideband stealth coating design using the Chinese Remainder Theorem (CRT)
to encode multiple discrete absorption bands into a unified frequency
parameter.  This enables multi-band plasma resonance engineering.

Incorporates core ideas from:
  - 170_chinese_remainder_theorem  (CRT reconstruction)

Physics Motivation
------------------
A plasma coating can be tuned to absorb strongly at specific frequencies
by matching the local plasma frequency to the target frequency.
If we want stealth simultaneously at frequencies f_1, f_2, ..., f_K,
the CRT provides a mathematical framework to construct a composite
frequency parameter F such that:

    F mod M_i = R_i

where M_i are pairwise-coprime frequency-bin moduli and R_i are the
encoded target bands.  This is used here as a design-space encoding
tool rather than a direct physical law.
"""

import numpy as np
import math
from utils import clamp


def extended_gcd(a: int, b: int) -> tuple:
    """
    Extended Euclidean algorithm.
    Returns (g, x, y) such that a*x + b*y = g = gcd(a, b).
    """
    if b == 0:
        return (abs(a), 1 if a >= 0 else -1, 0)
    g, x1, y1 = extended_gcd(b, a % b)
    x = y1
    y = x1 - (a // b) * y1
    return g, x, y


def mod_inverse(a: int, m: int) -> int:
    """
    Compute the modular multiplicative inverse of a modulo m.
    Raises ValueError if inverse does not exist.
    """
    g, x, _ = extended_gcd(a, m)
    if g != 1:
        raise ValueError(f"No modular inverse for {a} mod {m}.")
    return x % m


def chinese_remainder_theorem(
    remainders: np.ndarray,
    moduli: np.ndarray,
) -> int:
    """
    Solve the simultaneous congruences using the Chinese Remainder Theorem.

        x ≡ r_i (mod m_i)    for i = 0..K-1

    where m_i are pairwise coprime.

    Parameters
    ----------
    remainders : (K,) ndarray of int
    moduli : (K,) ndarray of int

    Returns
    -------
    x : int
        The unique solution modulo M = prod(m_i).
    """
    remainders = np.asarray(remainders, dtype=int)
    moduli = np.asarray(moduli, dtype=int)
    if remainders.shape != moduli.shape:
        raise ValueError("remainders and moduli must have the same shape.")

    K = remainders.size
    # Verify pairwise coprimality
    for i in range(K):
        for j in range(i + 1, K):
            g, _, _ = extended_gcd(int(moduli[i]), int(moduli[j]))
            if g != 1:
                raise ValueError(
                    f"Moduli {moduli[i]} and {moduli[j]} are not coprime (gcd={g})."
                )

    M = int(np.prod(moduli))
    x = 0
    for i in range(K):
        Mi = M // int(moduli[i])
        inv = mod_inverse(Mi, int(moduli[i]))
        x += int(remainders[i]) * Mi * inv

    return x % M


def encode_frequency_bands(
    target_frequencies_hz: np.ndarray,
    bin_width_hz: float,
) -> tuple:
    """
    Encode a set of target frequencies into CRT-compatible remainders and moduli.

    Each frequency is mapped to a bin index:
        r_i = round(f_i / bin_width)

    Moduli are chosen as the first K pairwise-coprime integers >= 10.

    Parameters
    ----------
    target_frequencies_hz : (K,) ndarray
        Target absorption frequencies [Hz].
    bin_width_hz : float
        Frequency bin width [Hz].

    Returns
    -------
    (remainders, moduli, composite_parameter)
    """
    target_frequencies_hz = np.asarray(target_frequencies_hz, dtype=float)
    bin_width_hz = max(float(bin_width_hz), 1.0)

    K = target_frequencies_hz.size
    remainders = np.round(target_frequencies_hz / bin_width_hz).astype(int)
    remainders = np.abs(remainders)  # positive bins only
    max_r = int(np.max(remainders))

    # Choose moduli larger than all remainders to satisfy CRT bounds
    candidate = max(max_r + 1, 100)
    if candidate % 2 == 0:
        candidate += 1
    moduli_list = []
    while len(moduli_list) < K:
        is_coprime = all(
            extended_gcd(candidate, int(m))[0] == 1
            for m in moduli_list
        )
        if is_coprime:
            moduli_list.append(candidate)
        candidate += 2

    moduli = np.array(moduli_list, dtype=int)
    composite = chinese_remainder_theorem(remainders, moduli)
    return remainders, moduli, composite


def decode_frequency_bands(
    composite: int,
    moduli: np.ndarray,
    bin_width_hz: float,
) -> np.ndarray:
    """
    Decode the target frequencies from a composite CRT parameter.

    Parameters
    ----------
    composite : int
        The CRT composite parameter.
    moduli : (K,) ndarray of int
    bin_width_hz : float

    Returns
    -------
    frequencies : (K,) ndarray
        Reconstructed frequencies [Hz].
    """
    moduli = np.asarray(moduli, dtype=int)
    bin_width_hz = max(float(bin_width_hz), 1.0)
    K = moduli.size
    freqs = np.zeros(K, dtype=float)
    for i in range(K):
        r_i = composite % int(moduli[i])
        freqs[i] = r_i * bin_width_hz
    return freqs


def design_multiband_coating_parameters(
    band_frequencies: np.ndarray,
    band_widths: np.ndarray,
    base_plasma_freq: float,
) -> dict:
    """
    Design plasma coating parameters for multi-band absorption using CRT encoding.

    For each target band, the local plasma frequency is tuned to match:
        omega_p,j = 2*pi * f_j

    The CRT composite parameter encodes all target bands into a single
    design index that can be used to generate a multi-resonance density profile.

    Parameters
    ----------
    band_frequencies : (K,) ndarray
        Target frequencies [Hz].
    band_widths : (K,) ndarray
        Absorption bandwidths [Hz].
    base_plasma_freq : float
        Base plasma angular frequency [rad/s] for scaling.

    Returns
    -------
    design : dict
        {
            "composite_index": int,
            "moduli": ndarray,
            "remainders": ndarray,
            "target_omega_p": ndarray,
            "band_widths": ndarray,
        }
    """
    band_frequencies = np.asarray(band_frequencies, dtype=float)
    band_widths = np.asarray(band_widths, dtype=float)

    min_freq = np.min(band_frequencies)
    bin_width = max(min_freq / 100.0, 1e6)

    remainders, moduli, composite = encode_frequency_bands(band_frequencies, bin_width)
    target_omega_p = 2.0 * math.pi * band_frequencies

    return {
        "composite_index": composite,
        "moduli": moduli,
        "remainders": remainders,
        "target_omega_p": target_omega_p,
        "band_widths": band_widths,
        "bin_width_hz": bin_width,
    }
