
import numpy as np
import math
from utils import clamp


def extended_gcd(a: int, b: int) -> tuple:
    if b == 0:
        return (abs(a), 1 if a >= 0 else -1, 0)
    g, x1, y1 = extended_gcd(b, a % b)
    x = y1
    y = x1 - (a // b) * y1
    return g, x, y


def mod_inverse(a: int, m: int) -> int:
    g, x, _ = extended_gcd(a, m)
    if g != 1:
        raise ValueError(f"No modular inverse for {a} mod {m}.")
    return x % m


def chinese_remainder_theorem(
    remainders: np.ndarray,
    moduli: np.ndarray,
) -> int:
    remainders = np.asarray(remainders, dtype=int)
    moduli = np.asarray(moduli, dtype=int)
    if remainders.shape != moduli.shape:
        raise ValueError("remainders and moduli must have the same shape.")

    K = remainders.size

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
    target_frequencies_hz = np.asarray(target_frequencies_hz, dtype=float)
    bin_width_hz = max(float(bin_width_hz), 1.0)

    K = target_frequencies_hz.size
    remainders = np.round(target_frequencies_hz / bin_width_hz).astype(int)
    remainders = np.abs(remainders)
    max_r = int(np.max(remainders))


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
